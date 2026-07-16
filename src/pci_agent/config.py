"""
Agent configuration.

Environment overrides follow the 12-factor pattern used elsewhere in the
codebase (see ``__main__.py``): the pydantic models are the source of
truth, and :meth:`AgentConfig.from_env` folds in the ``PCI_*`` variables at
process start.
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field

Backend = Literal["ollama", "llamacpp"]


class LLMConfig(BaseModel):
    """Configuration for the local language model.

    ``backend`` selects between the Ollama HTTP daemon (recommended) and the
    in-process ``llama-cpp-python`` GGUF loader. Historical behaviour is
    preserved: constructing ``AgentConfig()`` with no arguments still
    produces a config with ``model_path=None`` and no LLM loaded — the
    Ollama backend is opted into either explicitly or via
    ``PCI_LLM_BACKEND=ollama``.
    """

    backend: Backend = "llamacpp"

    # --- llama-cpp-python (in-process) --------------------------------
    model_path: str | None = None

    # --- Ollama (HTTP daemon) ----------------------------------------
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_tier: str = "default"
    ollama_model: str | None = None
    ollama_timeout_seconds: float = Field(default=120.0, gt=0.0)

    # --- Shared knobs -------------------------------------------------
    context_length: int = Field(default=4096, ge=512, le=1_000_000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=8192)
    n_gpu_layers: int = Field(default=0, ge=0)
    n_threads: int | None = None


class ContextConfig(BaseModel):
    """Configuration for context store connection"""

    endpoint: str = "http://localhost:8080"
    timeout_seconds: float = 30.0


class AgentConfig(BaseModel):
    """Main agent configuration"""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)

    # Agent behavior
    max_context_items: int = Field(default=10, ge=1, le=100)
    audit_logging: bool = True

    @classmethod
    def from_env(cls) -> AgentConfig:
        """Build an :class:`AgentConfig` from ``PCI_*`` environment variables.

        Unknown / unset variables fall back to model defaults.
        """
        backend_env = os.environ.get("PCI_LLM_BACKEND")
        backend: Backend = (
            backend_env  # type: ignore[assignment]
            if backend_env in ("ollama", "llamacpp")
            else "llamacpp"
        )

        timeout_seconds = 120.0
        if (timeout_env := os.environ.get("PCI_OLLAMA_TIMEOUT")) is not None:
            try:
                timeout_seconds = float(timeout_env)
            except ValueError as exc:
                raise ValueError("PCI_OLLAMA_TIMEOUT must be a float value") from exc

        return cls(
            llm=LLMConfig(
                backend=backend,
                model_path=os.environ.get("PCI_LLM_MODEL_PATH"),
                ollama_base_url=os.environ.get("PCI_OLLAMA_URL", "http://127.0.0.1:11434"),
                ollama_tier=os.environ.get("PCI_LLM_TIER", "default"),
                ollama_model=os.environ.get("PCI_OLLAMA_MODEL"),
                ollama_timeout_seconds=timeout_seconds,
            )
        )
