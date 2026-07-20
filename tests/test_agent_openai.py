"""Tests for the Agent's OpenAI-compatible wiring (config-driven dispatch)."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from pci_agent import Agent, AgentConfig, LLMConfig
from pci_agent.models.openai import OpenAICompatBackend


def _transport(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def _chat(content: str) -> dict:
    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
    }


class TestAgentBackendDispatch:
    async def test_openai_backend_selected_when_configured(self) -> None:
        config = AgentConfig(llm=LLMConfig(backend="openai"))
        agent = Agent(config)
        await agent.initialize()
        try:
            assert agent._openai_backend is not None
            assert agent._ollama_backend is None
            assert agent._llm is None
        finally:
            await agent.close()

    async def test_openai_generate_used_when_no_context(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert request.url.path == "/v1/chat/completions"
            assert body["messages"][0]["content"].endswith("Answer:")
            assert "Question: hello" in body["messages"][0]["content"]
            return httpx.Response(200, json=_chat("hi from openai"))

        config = AgentConfig(llm=LLMConfig(backend="openai"))
        agent = Agent(config)
        await agent.initialize()
        # Swap the backend with one bound to our mock transport.
        await agent._openai_backend.aclose()  # type: ignore[union-attr]
        agent._openai_backend = OpenAICompatBackend(
            model="test-model", transport=_transport(handler)
        )
        try:
            response = await agent.process("hello")
            assert response.content == "hi from openai"
        finally:
            await agent.close()

    async def test_close_releases_openai_client(self) -> None:
        config = AgentConfig(llm=LLMConfig(backend="openai"))
        agent = Agent(config)
        await agent.initialize()
        assert agent._openai_backend is not None
        await agent.close()
        assert agent._openai_backend is None


class TestAgentConfigFromEnv:
    def test_env_selects_openai_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PCI_LLM_BACKEND", "openai")
        monkeypatch.setenv("PCI_LLM_TIER", "small")
        monkeypatch.setenv("PCI_OPENAI_BASE_URL", "http://example.internal:8000/v1")
        monkeypatch.setenv("PCI_OPENAI_API_KEY", "sk-xyz")
        monkeypatch.delenv("PCI_OPENAI_MODEL", raising=False)
        monkeypatch.delenv("PCI_OPENAI_TIMEOUT", raising=False)

        cfg = AgentConfig.from_env()
        assert cfg.llm.backend == "openai"
        assert cfg.llm.openai_tier == "small"
        assert cfg.llm.openai_base_url == "http://example.internal:8000/v1"
        assert cfg.llm.openai_api_key == "sk-xyz"

    def test_env_openai_timeout_rejects_infinity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Infinite timeouts disable the httpx deadline; reject at parse time."""
        monkeypatch.setenv("PCI_OPENAI_TIMEOUT", "inf")
        with pytest.raises(ValueError, match="finite"):
            AgentConfig.from_env()

    def test_llm_config_rejects_infinite_openai_timeout(self) -> None:
        with pytest.raises(ValueError):
            LLMConfig(openai_timeout_seconds=float("inf"))


class TestAgentLifecycleSafety:
    async def test_initialize_releases_backend_on_context_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If context connect() fails, the OpenAI backend must not leak."""
        agent = Agent(AgentConfig(llm=LLMConfig(backend="openai")))

        async def boom() -> None:
            raise RuntimeError("context store unavailable")

        monkeypatch.setattr(agent._context_client, "connect", boom)

        with pytest.raises(RuntimeError, match="context store unavailable"):
            await agent.initialize()

        assert agent._openai_backend is None
        assert not agent._initialized
