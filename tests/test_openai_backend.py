"""Tests for the OpenAI-compatible HTTP backend.

Uses ``httpx.MockTransport`` so no daemon is required. Live-daemon smoke
tests live in ``test_openai_integration.py`` behind the ``openai`` marker.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable

import httpx
import pytest

from pci_agent.models.backend import LLMResponse, StructuredResponse
from pci_agent.models.ollama import MODEL_TIERS
from pci_agent.models.openai import (
    DEFAULT_OPENAI_BASE_URL,
    OpenAICompatBackend,
    OpenAIModelRefusalError,
    OpenAISchemaError,
    OpenAITimeoutError,
    OpenAITransportError,
)

# --- helpers ---


def _mock_transport(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def _chat(content: str, *, finish_reason: str = "stop", completion_tokens: int = 12) -> dict:
    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": completion_tokens, "total_tokens": 17},
    }


def _ok(body: dict) -> httpx.Response:
    return httpx.Response(200, json=body)


# --- model / env resolution ---


class TestModelResolution:
    def test_reuses_shared_tiers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PCI_LLM_TIER", "small")
        monkeypatch.delenv("PCI_OPENAI_MODEL", raising=False)
        backend = OpenAICompatBackend()
        assert backend.model == MODEL_TIERS["small"]

    def test_explicit_env_model_overrides_tier(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PCI_LLM_TIER", "small")
        monkeypatch.setenv("PCI_OPENAI_MODEL", "my-served-model")
        backend = OpenAICompatBackend()
        assert backend.model == "my-served-model"

    def test_explicit_arg_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PCI_OPENAI_MODEL", "env-model")
        backend = OpenAICompatBackend(model="arg-model")
        assert backend.model == "arg-model"

    def test_unknown_tier_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PCI_LLM_TIER", "gigantic")
        monkeypatch.delenv("PCI_OPENAI_MODEL", raising=False)
        with pytest.raises(ValueError, match="Unknown LLM tier"):
            OpenAICompatBackend()

    def test_from_env_uses_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in (
            "PCI_LLM_TIER",
            "PCI_OPENAI_MODEL",
            "PCI_OPENAI_BASE_URL",
            "PCI_OPENAI_TIMEOUT",
            "PCI_OPENAI_API_KEY",
        ):
            monkeypatch.delenv(var, raising=False)
        backend = OpenAICompatBackend.from_env()
        assert backend.base_url == DEFAULT_OPENAI_BASE_URL
        assert backend.model == MODEL_TIERS["default"]
        assert backend.api_key is None

    def test_from_env_reads_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PCI_OPENAI_API_KEY", "sk-secret")
        backend = OpenAICompatBackend.from_env()
        assert backend.api_key == "sk-secret"

    def test_from_env_rejects_bad_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PCI_OPENAI_TIMEOUT", "not-a-float")
        with pytest.raises(ValueError, match="PCI_OPENAI_TIMEOUT"):
            OpenAICompatBackend.from_env()

    def test_from_env_rejects_infinite_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`inf` parses as a valid float but disables the httpx deadline."""
        monkeypatch.setenv("PCI_OPENAI_TIMEOUT", "inf")
        with pytest.raises(ValueError, match="finite"):
            OpenAICompatBackend.from_env()

    def test_from_env_rejects_non_positive_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PCI_OPENAI_TIMEOUT", "0")
        with pytest.raises(ValueError, match="positive"):
            OpenAICompatBackend.from_env()


class TestTransportSecurity:
    """An API key over plaintext to a remote host leaks the bearer token."""

    def test_remote_http_with_api_key_warns(self) -> None:
        with pytest.warns(UserWarning, match="cleartext"):
            OpenAICompatBackend(
                base_url="http://example.internal:8000/v1",
                model="m",
                api_key="sk-abc",
            )

    def test_loopback_http_with_api_key_does_not_warn(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            OpenAICompatBackend(
                base_url="http://127.0.0.1:8000/v1",
                model="m",
                api_key="sk-abc",
            )

    def test_remote_https_with_api_key_does_not_warn(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            OpenAICompatBackend(
                base_url="https://api.example.com/v1",
                model="m",
                api_key="sk-abc",
            )

    def test_remote_http_without_api_key_does_not_warn(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            OpenAICompatBackend(base_url="http://example.internal:8000/v1", model="m")


# --- generate() ---


class TestGenerate:
    async def test_happy_path(self) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/chat/completions"
            assert request.method == "POST"
            seen.update(json.loads(request.content))
            return _ok(_chat("hello world", completion_tokens=9))

        async with OpenAICompatBackend(
            model="test-model",
            transport=_mock_transport(handler),
        ) as backend:
            result = await backend.generate("Say hi")

        assert isinstance(result, LLMResponse)
        assert result.text == "hello world"
        assert result.tokens_used == 9
        assert result.finish_reason == "stop"
        assert seen["model"] == "test-model"
        assert seen["messages"] == [{"role": "user", "content": "Say hi"}]
        assert seen["stream"] is False
        assert "response_format" not in seen  # unstructured call must NOT constrain output.

    async def test_max_tokens_forwarded(self) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return _ok(_chat("ok"))

        async with OpenAICompatBackend(model="m", transport=_mock_transport(handler)) as backend:
            await backend.generate("Hi", max_tokens=42, temperature=0.9)

        assert seen["max_tokens"] == 42
        assert seen["temperature"] == 0.9

    async def test_api_key_sets_authorization_header(self) -> None:
        seen: dict[str, str | None] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers.get("Authorization")
            return _ok(_chat("ok"))

        async with OpenAICompatBackend(
            model="m", api_key="sk-abc", transport=_mock_transport(handler)
        ) as backend:
            await backend.generate("Hi")

        assert seen["auth"] == "Bearer sk-abc"

    async def test_no_api_key_omits_authorization_header(self) -> None:
        seen: dict[str, str | None] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers.get("Authorization")
            return _ok(_chat("ok"))

        async with OpenAICompatBackend(model="m", transport=_mock_transport(handler)) as backend:
            await backend.generate("Hi")

        assert seen["auth"] is None

    async def test_transport_error(self) -> None:
        def handler(_req: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        backend = OpenAICompatBackend(transport=_mock_transport(handler), max_retries=1)
        with pytest.raises(OpenAITransportError, match="transport error"):
            await backend.generate("Hi")
        await backend.aclose()

    async def test_timeout(self) -> None:
        def handler(_req: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timeout")

        backend = OpenAICompatBackend(transport=_mock_transport(handler), max_retries=1)
        with pytest.raises(OpenAITimeoutError):
            await backend.generate("Hi")
        await backend.aclose()

    async def test_non_200(self) -> None:
        def handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(400, text="bad request")

        backend = OpenAICompatBackend(transport=_mock_transport(handler))
        with pytest.raises(OpenAITransportError, match="HTTP 400"):
            await backend.generate("Hi")
        await backend.aclose()

    async def test_retries_5xx_then_succeeds(self) -> None:
        calls = {"n": 0}

        def handler(_req: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(503, text="unavailable")
            return _ok(_chat("recovered", completion_tokens=3))

        backend = OpenAICompatBackend(transport=_mock_transport(handler), max_retries=2)
        result = await backend.generate("Hi")
        assert result.text == "recovered"
        assert calls["n"] == 2
        await backend.aclose()

    async def test_missing_choices_raises(self) -> None:
        def handler(_req: httpx.Request) -> httpx.Response:
            return _ok({"choices": [], "usage": {}})

        backend = OpenAICompatBackend(transport=_mock_transport(handler))
        with pytest.raises(OpenAITransportError, match="no choices"):
            await backend.generate("Hi")
        await backend.aclose()


# --- generate_structured() ---


_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["allow", "deny"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
}


class TestGenerateStructured:
    async def test_happy_path(self) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            payload = json.dumps({"verdict": "allow", "reason": "ok"})
            return _ok(_chat(payload, completion_tokens=20))

        async with OpenAICompatBackend(
            model="test-model", transport=_mock_transport(handler)
        ) as backend:
            result = await backend.generate_structured("classify", _SCHEMA)

        assert isinstance(result, StructuredResponse)
        assert result.data == {"verdict": "allow", "reason": "ok"}
        assert result.tokens_used == 20
        assert result.model == "test-model"
        assert result.metadata["prompt_tokens"] == 5
        # Structured output travels through response_format.json_schema.
        response_format = captured["response_format"]
        assert isinstance(response_format, dict)
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["schema"] == _SCHEMA
        assert response_format["json_schema"]["strict"] is True

    async def test_schema_mismatch_raises(self) -> None:
        """Model returned text that isn't JSON."""

        def handler(_req: httpx.Request) -> httpx.Response:
            return _ok(_chat("not a JSON object"))

        backend = OpenAICompatBackend(transport=_mock_transport(handler))
        with pytest.raises(OpenAISchemaError, match="not valid JSON"):
            await backend.generate_structured("go", _SCHEMA)
        await backend.aclose()

    async def test_non_object_json_raises(self) -> None:
        def handler(_req: httpx.Request) -> httpx.Response:
            return _ok(_chat('["a", "b"]'))

        backend = OpenAICompatBackend(transport=_mock_transport(handler))
        with pytest.raises(OpenAISchemaError, match="JSON object"):
            await backend.generate_structured("go", _SCHEMA)
        await backend.aclose()

    async def test_empty_content_raises_refusal(self) -> None:
        def handler(_req: httpx.Request) -> httpx.Response:
            return _ok(_chat("   "))

        backend = OpenAICompatBackend(transport=_mock_transport(handler))
        with pytest.raises(OpenAIModelRefusalError):
            await backend.generate_structured("go", _SCHEMA)
        await backend.aclose()

    async def test_explicit_refusal_field_raises(self) -> None:
        """OpenAI structured outputs surface refusals via a dedicated field."""

        def handler(_req: httpx.Request) -> httpx.Response:
            return _ok(
                {
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "refusal": "I cannot",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"completion_tokens": 4},
                }
            )

        backend = OpenAICompatBackend(transport=_mock_transport(handler))
        with pytest.raises(OpenAIModelRefusalError, match="refused"):
            await backend.generate_structured("go", _SCHEMA)
        await backend.aclose()
