"""
In-process ``llama-cpp-python`` backend (GGUF file loader).

This is the "no daemon" fallback path. Loading the model requires the
optional ``llm`` extra to be installed (``uv sync --extra llm``); if
``llama-cpp-python`` is missing, :meth:`LocalLLM.load` raises
:class:`ImportError` so callers can degrade or surface the error.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pci_agent.config import LLMConfig
from pci_agent.models.backend import LLMResponse

_MISSING_DEP_MESSAGE = (
    "llama-cpp-python is not installed. Install the optional 'llm' extra: "
    "'uv sync --extra llm' — or switch to the Ollama backend via "
    "LLMConfig(backend='ollama')."
)


class LocalLLM:
    """Thin async wrapper over ``llama_cpp.Llama``.

    The underlying library is synchronous; both :meth:`load` and
    :meth:`generate` off-load their calls onto the default executor so the
    asyncio loop is never blocked.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._model: Any = None

    async def load(self) -> None:
        """Instantiate the underlying ``Llama`` model in a worker thread."""
        try:
            from llama_cpp import Llama  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised via mock
            raise ImportError(_MISSING_DEP_MESSAGE) from exc

        if not self.config.model_path:
            raise ValueError("LLMConfig.model_path must be set to load LocalLLM")

        loop = asyncio.get_running_loop()

        def _build() -> Any:
            return Llama(
                model_path=self.config.model_path,
                n_ctx=self.config.context_length,
                n_gpu_layers=self.config.n_gpu_layers,
                n_threads=self.config.n_threads,
                verbose=False,
            )

        self._model = await loop.run_in_executor(None, _build)

    async def unload(self) -> None:
        """Release native model resources.

        ``llama-cpp-python`` >= 0.3.0 exposes ``Llama.close()`` for
        deterministic cleanup of native CPU/GPU allocations; earlier
        versions rely on garbage collection, so we tolerate a missing
        ``close`` attribute.
        """
        model, self._model = self._model, None
        if model is None:
            return
        close = getattr(model, "close", None)
        if close is None:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, close)

    async def aclose(self) -> None:
        """Alias for :meth:`unload` matching the ``LLMBackend`` protocol."""
        await self.unload()

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Run a single completion request."""
        if self._model is None:
            raise RuntimeError("LocalLLM.load() must be awaited before generate()")

        loop = asyncio.get_running_loop()
        effective_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        effective_temperature = temperature if temperature is not None else self.config.temperature

        def _run() -> Any:
            return self._model(
                prompt,
                max_tokens=effective_max_tokens,
                temperature=effective_temperature,
            )

        raw = await loop.run_in_executor(None, _run)

        # llama.cpp returns an OpenAI-shaped completion dict; be defensive
        # about the exact shape so mocks and future upstream tweaks don't
        # break us.
        text = ""
        finish_reason = "stop"
        tokens_used = 0
        choices = raw.get("choices") if isinstance(raw, dict) else None
        if choices:
            first = choices[0]
            text = str(first.get("text", ""))
            finish_reason = str(first.get("finish_reason") or "stop")
        usage = raw.get("usage") if isinstance(raw, dict) else None
        if isinstance(usage, dict):
            tokens_used = int(usage.get("total_tokens", 0) or 0)

        return LLMResponse(
            text=text,
            tokens_used=tokens_used,
            finish_reason=finish_reason,
        )
