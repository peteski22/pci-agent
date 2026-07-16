"""
Ollama HTTP backend.

Talks to a local Ollama daemon (default ``http://127.0.0.1:11434``) over the
``/api/generate`` endpoint. Structured output uses Ollama's ``format``
parameter, which forwards a JSON schema to llama.cpp's built-in
schema-to-GBNF constraint — the model can only emit tokens that keep the
output conformant.

Environment variables (12-factor style, matching ``__main__.py``):

- ``PCI_LLM_BACKEND`` — ``ollama`` (default) or ``llamacpp``.
- ``PCI_LLM_TIER`` — ``default`` (Qwen3.6-27B), ``small`` (Phi-4-mini) or
  ``on-device`` (Bonsai-27B 1-bit).
- ``PCI_OLLAMA_MODEL`` — explicit model tag; overrides tier when set.
- ``PCI_OLLAMA_URL`` — base URL of the Ollama daemon.
- ``PCI_OLLAMA_TIMEOUT`` — per-request timeout in seconds (float).
"""

from __future__ import annotations

import json
import os
from types import TracebackType
from typing import Any

import httpx

from pci_agent.models.backend import LLMResponse, StructuredResponse

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_RETRIES = 2

# Named model tiers.
#
# ``on-device`` intentionally selects the 1-bit Bonsai-27B tag: the 1-bit
# (Q1_0) kernels are mainlined into upstream llama.cpp and load through
# Ollama's bundled build. The ternary (Q2_0) variant currently requires the
# PrismML llama.cpp fork and is *not* the default — see the runtime report.
MODEL_TIERS: dict[str, str] = {
    "default": "qwen3.6:27b",
    "small": "phi4-mini:3.8b",
    "on-device": "bonsai:27b-q1_0",
}
DEFAULT_TIER = "default"


class OllamaError(Exception):
    """Base class for Ollama backend errors."""


class OllamaTransportError(OllamaError):
    """Raised when the HTTP call itself failed (connection, non-2xx, etc.)."""


class OllamaTimeoutError(OllamaError):
    """Raised when the daemon did not respond within the configured timeout."""


class OllamaSchemaError(OllamaError):
    """Raised when the model returned text that failed schema decoding."""


class OllamaModelRefusalError(OllamaError):
    """Raised when the model returned an empty/refusal response."""


def resolve_model_for_tier(tier: str) -> str:
    """Return the Ollama tag associated with a named tier.

    Raises ``ValueError`` on an unknown tier so misconfiguration surfaces at
    startup rather than at first inference request.
    """
    try:
        return MODEL_TIERS[tier]
    except KeyError as exc:
        known = ", ".join(sorted(MODEL_TIERS))
        raise ValueError(f"Unknown LLM tier '{tier}' — known tiers: {known}") from exc


def _resolve_env_model(
    *,
    explicit_model: str | None,
    tier: str | None,
) -> str:
    """Model selection precedence: explicit arg > env override > tier default."""
    if explicit_model:
        return explicit_model
    env_model = os.environ.get("PCI_OLLAMA_MODEL")
    if env_model:
        return env_model
    resolved_tier = tier or os.environ.get("PCI_LLM_TIER", DEFAULT_TIER)
    return resolve_model_for_tier(resolved_tier)


class OllamaBackend:
    """Async HTTP client for a local Ollama daemon.

    Instances own an :class:`httpx.AsyncClient`. Call :meth:`aclose` when
    done (or use the backend as an async context manager). The client is
    created lazily so the backend is safe to construct from sync code.
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str | None = None,
        tier: str | None = None,
        request_timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        temperature: float = 0.2,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = _resolve_env_model(explicit_model=model, tier=tier)
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    # ---------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> OllamaBackend:
        """Build a backend using ``PCI_OLLAMA_*`` / ``PCI_LLM_TIER`` env vars."""
        base_url = os.environ.get("PCI_OLLAMA_URL", DEFAULT_OLLAMA_URL)
        try:
            timeout = float(
                os.environ.get("PCI_OLLAMA_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS))
            )
        except ValueError as exc:
            raise ValueError("PCI_OLLAMA_TIMEOUT must be a float value") from exc

        return cls(
            base_url=base_url,
            model=None,  # let _resolve_env_model consult PCI_OLLAMA_MODEL / tier
            tier=None,
            request_timeout=timeout,
            transport=transport,
        )

    # ---------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.request_timeout,
                transport=self._transport,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> OllamaBackend:
        self._get_client()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Free-form text generation. Returns unparsed text."""
        payload = self._build_payload(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            schema=None,
        )
        raw = await self._post_generate(payload)
        return LLMResponse(
            text=str(raw.get("response", "")),
            tokens_used=int(raw.get("eval_count", 0) or 0),
            finish_reason=_finish_reason(raw),
        )

    async def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> StructuredResponse:
        """Schema-constrained generation.

        Passes ``schema`` to Ollama's ``format`` field so the model can only
        emit conformant JSON. Callers are still expected to validate the
        result against their own pydantic model at the trust boundary — the
        schema guarantees shape but not semantic correctness.
        """
        payload = self._build_payload(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            schema=schema,
        )
        raw = await self._post_generate(payload)

        text = str(raw.get("response", "")).strip()
        if not text:
            raise OllamaModelRefusalError(
                f"Ollama model '{payload['model']}' returned an empty response"
            )
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise OllamaSchemaError(
                f"Model output was not valid JSON: {exc.msg}"
            ) from exc
        if not isinstance(data, dict):
            raise OllamaSchemaError(
                f"Model output must decode to a JSON object, got {type(data).__name__}"
            )

        return StructuredResponse(
            data=data,
            raw_text=text,
            model=str(payload["model"]),
            tokens_used=int(raw.get("eval_count", 0) or 0),
            finish_reason=_finish_reason(raw),
            metadata={
                "prompt_eval_count": int(raw.get("prompt_eval_count", 0) or 0),
                "total_duration_ns": int(raw.get("total_duration", 0) or 0),
            },
        )

    # ---------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------

    def _build_payload(
        self,
        *,
        prompt: str,
        model: str | None,
        max_tokens: int | None,
        temperature: float | None,
        schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        payload: dict[str, Any] = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if schema is not None:
            payload["format"] = schema
        return payload

    async def _post_generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        client = self._get_client()
        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await client.post("/api/generate", json=payload)
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    raise OllamaTimeoutError(
                        f"Ollama request timed out after {self.request_timeout}s"
                    ) from exc
                continue
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    raise OllamaTransportError(
                        f"Ollama transport error: {exc}"
                    ) from exc
                continue

            if response.status_code >= 500 and attempt < self.max_retries:
                # Retry transient 5xx once
                last_exc = OllamaTransportError(
                    f"Ollama returned {response.status_code}"
                )
                continue
            if response.status_code != 200:
                raise OllamaTransportError(
                    f"Ollama returned HTTP {response.status_code}: {response.text[:500]}"
                )

            try:
                data = response.json()
            except ValueError as exc:
                raise OllamaTransportError(
                    "Ollama response was not valid JSON"
                ) from exc
            if not isinstance(data, dict):
                raise OllamaTransportError(
                    f"Ollama response must be an object, got {type(data).__name__}"
                )
            return data

        # Loop exhausted without returning — should be unreachable because
        # the final iteration raises. Kept as a defensive fallback.
        raise OllamaTransportError(
            f"Ollama request failed after {self.max_retries + 1} attempts: {last_exc}"
        )


def _finish_reason(raw: dict[str, Any]) -> str:
    if raw.get("done") is True:
        reason = raw.get("done_reason")
        if isinstance(reason, str) and reason:
            return reason
        return "stop"
    return "length"
