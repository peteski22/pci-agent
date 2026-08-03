"""
Agent configuration.

Environment overrides follow the 12-factor pattern used elsewhere in the
codebase (see ``__main__.py``): the pydantic models are the source of
truth, and :meth:`AgentConfig.from_env` folds in the ``PCI_*`` variables at
process start.
"""

from __future__ import annotations

import math
import os
from typing import Literal

from pydantic import BaseModel, Field

from pci_agent.coordination import ApprovalMode

Backend = Literal["ollama", "openai", "llamacpp"]


def _parse_env_timeout(var_name: str, default: float) -> float:
    """Parse a per-request timeout env var, rejecting non-finite values.

    A non-finite timeout (e.g. ``inf``) disables the httpx2 deadline, so it is
    rejected here rather than silently dropped. Non-positive values are left
    for the pydantic ``Field`` guard to reject with its own message.
    """
    raw = os.environ.get(var_name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{var_name} must be a float value") from exc
    if not math.isfinite(value):
        raise ValueError(f"{var_name} must be a finite value")
    return value


class LLMConfig(BaseModel):
    """Configuration for the local language model.

    ``backend`` selects between the Ollama HTTP daemon (recommended), an
    OpenAI-compatible HTTP endpoint, and the in-process ``llama-cpp-python``
    GGUF loader. Historical behaviour is preserved: constructing
    ``AgentConfig()`` with no arguments still produces a config with
    ``model_path=None`` and no LLM loaded — an HTTP backend is opted into
    either explicitly or via ``PCI_LLM_BACKEND``.
    """

    backend: Backend = "llamacpp"

    # --- llama-cpp-python (in-process) --------------------------------
    model_path: str | None = None

    # --- Ollama (HTTP daemon) ----------------------------------------
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_tier: str = "default"
    ollama_model: str | None = None
    ollama_timeout_seconds: float = Field(default=120.0, gt=0.0, allow_inf_nan=False)

    # --- OpenAI-compatible (HTTP daemon) -----------------------------
    openai_base_url: str = "http://127.0.0.1:8000/v1"
    openai_tier: str = "default"
    openai_model: str | None = None
    openai_api_key: str | None = None
    openai_timeout_seconds: float = Field(default=120.0, gt=0.0, allow_inf_nan=False)

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


class ApprovalConfig(BaseModel):
    """Autonomous-approval behavior."""

    mode: ApprovalMode = ApprovalMode.MANUAL


class AgentConfig(BaseModel):
    """Main agent configuration"""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    approval: ApprovalConfig = Field(default_factory=ApprovalConfig)

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
            if backend_env in ("ollama", "openai", "llamacpp")
            else "llamacpp"
        )

        # PCI_LLM_TIER is shared across the HTTP backends; PCI_OPENAI_* mirror
        # the PCI_OLLAMA_* family for the OpenAI-compatible endpoint.
        tier = os.environ.get("PCI_LLM_TIER", "default")

        mode_env = os.environ.get("PCI_APPROVAL_MODE")
        mode = ApprovalMode(mode_env) if mode_env in tuple(ApprovalMode) else ApprovalMode.MANUAL

        return cls(
            llm=LLMConfig(
                backend=backend,
                model_path=os.environ.get("PCI_LLM_MODEL_PATH"),
                ollama_base_url=os.environ.get("PCI_OLLAMA_URL", "http://127.0.0.1:11434"),
                ollama_tier=tier,
                ollama_model=os.environ.get("PCI_OLLAMA_MODEL"),
                ollama_timeout_seconds=_parse_env_timeout("PCI_OLLAMA_TIMEOUT", 120.0),
                openai_base_url=os.environ.get("PCI_OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
                openai_tier=tier,
                openai_model=os.environ.get("PCI_OPENAI_MODEL"),
                openai_api_key=os.environ.get("PCI_OPENAI_API_KEY"),
                openai_timeout_seconds=_parse_env_timeout("PCI_OPENAI_TIMEOUT", 120.0),
            ),
            approval=ApprovalConfig(mode=mode),
        )
