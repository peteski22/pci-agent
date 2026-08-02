"""
OpenAI-compatible HTTP backend.

Talks to any daemon that speaks the OpenAI ``/chat/completions`` protocol —
vLLM, SGLang, LM Studio, ``llama-cpp-python``'s ``server.py``, oobabooga and
most hosted providers. This is the de-facto interoperability contract across
the local-LLM ecosystem, so pci-demo can bundle whichever daemon it likes
without pci-agent caring which one is running.

Structured output uses OpenAI's ``response_format`` with a
``json_schema`` block, which constrains the model to emit only conformant
JSON. The error taxonomy mirrors :mod:`pci_agent.models.ollama` so callers
(e.g. the S-PAL bridge) can stay backend-agnostic.

Environment variables (12-factor style, matching ``__main__.py``):

- ``PCI_LLM_BACKEND`` — ``openai`` selects this backend.
- ``PCI_LLM_TIER`` — ``default`` (Qwen3.6-27B) or ``small`` (Phi-4-mini).
- ``PCI_OPENAI_MODEL`` — explicit model name; overrides tier when set.
- ``PCI_OPENAI_BASE_URL`` — base URL of the OpenAI-compatible endpoint,
  including the ``/v1`` root (e.g. ``http://127.0.0.1:8000/v1``).
- ``PCI_OPENAI_API_KEY`` — optional; sent as ``Authorization: Bearer …``.
- ``PCI_OPENAI_TIMEOUT`` — per-request timeout in seconds (float).
"""

from __future__ import annotations

import json
import math
import os
import warnings
from types import TracebackType
from typing import Any
from urllib.parse import urlsplit

import httpx2

from pci_agent.models.backend import LLMResponse, StructuredResponse
from pci_agent.models.ollama import DEFAULT_TIER, resolve_model_for_tier

DEFAULT_OPENAI_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_RETRIES = 2

# Name attached to the JSON-schema block sent via ``response_format``. The
# OpenAI protocol requires a name; its value is not otherwise significant.
_SCHEMA_NAME = "response"

# Hosts for which a plaintext (``http://``) endpoint carries no network
# exposure, so an API key sent over them does not leave the machine.
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _warn_on_insecure_credentials(base_url: str, api_key: str | None) -> None:
    """Warn if an API key would be sent in cleartext to a remote endpoint.

    A bearer token over plaintext ``http://`` to anything but a loopback host
    is exposed on the wire. The loopback default is safe; this only fires on
    non-default remote configuration.
    """
    if not api_key:
        return
    parts = urlsplit(base_url)
    if parts.scheme == "https" or (parts.hostname or "") in _LOOPBACK_HOSTS:
        return
    warnings.warn(
        "An API key is set but the endpoint is not HTTPS and not a loopback "
        "host; the bearer token will be sent in cleartext.",
        stacklevel=3,
    )


class OpenAIError(Exception):
    """Base class for OpenAI-compatible backend errors."""


class OpenAITransportError(OpenAIError):
    """Raised when the HTTP call itself failed (connection, non-2xx, etc.)."""


class OpenAITimeoutError(OpenAIError):
    """Raised when the server did not respond within the configured timeout."""


class OpenAISchemaError(OpenAIError):
    """Raised when the model returned text that failed schema decoding."""


class OpenAIModelRefusalError(OpenAIError):
    """Raised when the model refused or returned an empty response."""


def _resolve_env_model(
    *,
    explicit_model: str | None,
    tier: str | None,
) -> str:
    """Model selection precedence: explicit arg > env override > tier default."""
    if explicit_model:
        return explicit_model
    env_model = os.environ.get("PCI_OPENAI_MODEL")
    if env_model:
        return env_model
    resolved_tier = tier or os.environ.get("PCI_LLM_TIER", DEFAULT_TIER)
    return resolve_model_for_tier(resolved_tier)


class OpenAICompatBackend:
    """Async HTTP client for an OpenAI-compatible chat endpoint.

    Instances own an :class:`httpx2.AsyncClient`. Call :meth:`aclose` when
    done (or use the backend as an async context manager). The client is
    created lazily so the backend is safe to construct from sync code.
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        model: str | None = None,
        tier: str | None = None,
        api_key: str | None = None,
        request_timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        temperature: float = 0.2,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = _resolve_env_model(explicit_model=model, tier=tier)
        self.api_key = api_key
        _warn_on_insecure_credentials(self.base_url, api_key)
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self._transport = transport
        self._client: httpx2.AsyncClient | None = None

    # ---------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        *,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> OpenAICompatBackend:
        """Build a backend using ``PCI_OPENAI_*`` / ``PCI_LLM_TIER`` env vars."""
        base_url = os.environ.get("PCI_OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)
        try:
            timeout = float(os.environ.get("PCI_OPENAI_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS)))
        except ValueError as exc:
            raise ValueError("PCI_OPENAI_TIMEOUT must be a float value") from exc
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("PCI_OPENAI_TIMEOUT must be a positive finite value")

        return cls(
            base_url=base_url,
            model=None,  # let _resolve_env_model consult PCI_OPENAI_MODEL / tier.
            tier=None,
            api_key=os.environ.get("PCI_OPENAI_API_KEY"),
            request_timeout=timeout,
            transport=transport,
        )

    # ---------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------

    def _get_client(self) -> httpx2.AsyncClient:
        if self._client is None:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
            self._client = httpx2.AsyncClient(
                base_url=self.base_url,
                timeout=self.request_timeout,
                transport=self._transport,
                headers=headers,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> OpenAICompatBackend:
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
        raw = await self._post_chat(payload)
        message, finish_reason = self._extract_message(raw)
        return LLMResponse(
            text=str(message.get("content") or ""),
            tokens_used=_completion_tokens(raw),
            finish_reason=finish_reason,
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

        Passes ``schema`` to the server via ``response_format`` so the model
        can only emit conformant JSON. Callers are still expected to validate
        the result against their own pydantic model at the trust boundary —
        the schema guarantees shape but not semantic correctness.

        Raises:
            OpenAIModelRefusalError: If the model refused the request or
                returned an empty response.
            OpenAISchemaError: If the response was not a JSON object.
            OpenAIError: On transport, timeout or protocol failures (see the
                subclasses raised by the underlying request).
        """
        payload = self._build_payload(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            schema=schema,
        )
        raw = await self._post_chat(payload)
        message, finish_reason = self._extract_message(raw)

        refusal = message.get("refusal")
        if isinstance(refusal, str) and refusal.strip():
            raise OpenAIModelRefusalError(
                f"Model '{payload['model']}' refused the request: {refusal.strip()}"
            )

        text = str(message.get("content") or "").strip()
        if not text:
            raise OpenAIModelRefusalError(f"Model '{payload['model']}' returned an empty response")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise OpenAISchemaError(f"Model output was not valid JSON: {exc.msg}") from exc
        if not isinstance(data, dict):
            raise OpenAISchemaError(
                f"Model output must decode to a JSON object, got {type(data).__name__}"
            )

        usage = raw.get("usage") or {}
        return StructuredResponse(
            data=data,
            raw_text=text,
            model=str(payload["model"]),
            tokens_used=_completion_tokens(raw),
            finish_reason=finish_reason,
            metadata={
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
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
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature if temperature is not None else self.temperature,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": _SCHEMA_NAME,
                    "schema": schema,
                    "strict": True,
                },
            }
        return payload

    async def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        client = self._get_client()
        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await client.post("/chat/completions", json=payload)
            except httpx2.TimeoutException as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    raise OpenAITimeoutError(
                        f"OpenAI-compatible request timed out after {self.request_timeout}s"
                    ) from exc
                continue
            except httpx2.HTTPError as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    raise OpenAITransportError(f"OpenAI-compatible transport error: {exc}") from exc
                continue

            if response.status_code >= 500 and attempt < self.max_retries:
                # Retry transient 5xx.
                last_exc = OpenAITransportError(
                    f"OpenAI-compatible endpoint returned {response.status_code}"
                )
                continue
            if response.status_code != 200:
                raise OpenAITransportError(
                    f"OpenAI-compatible endpoint returned HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )

            try:
                data = response.json()
            except ValueError as exc:
                raise OpenAITransportError("OpenAI-compatible response was not valid JSON") from exc
            if not isinstance(data, dict):
                raise OpenAITransportError(
                    f"OpenAI-compatible response must be an object, got {type(data).__name__}"
                )
            return data

        # Loop exhausted without returning — should be unreachable because
        # the final iteration raises. Kept as a defensive fallback.
        raise OpenAITransportError(
            f"OpenAI-compatible request failed after {self.max_retries + 1} attempts: {last_exc}"
        )

    @staticmethod
    def _extract_message(raw: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """Pull the first choice's message and finish reason from a response."""
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            raise OpenAITransportError("OpenAI-compatible response contained no choices")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise OpenAITransportError("OpenAI-compatible choice must be an object")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise OpenAITransportError("OpenAI-compatible choice contained no message")
        finish_reason = str(choice.get("finish_reason") or "stop")
        return message, finish_reason


def _completion_tokens(raw: dict[str, Any]) -> int:
    usage = raw.get("usage") or {}
    return int(usage.get("completion_tokens", 0) or 0)
