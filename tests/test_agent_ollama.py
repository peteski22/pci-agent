"""Tests for the Agent's Ollama wiring (config-driven dispatch + S-PAL hook)."""

from __future__ import annotations

import json

import httpx2
import pytest

from pci_agent import Agent, AgentConfig, LLMConfig
from pci_agent.models.ollama import OllamaBackend
from pci_agent.spal import IdentityType, ProofType


def _transport(handler):  # noqa: ANN001, ANN201
    return httpx2.MockTransport(handler)


class TestAgentBackendDispatch:
    async def test_default_config_uses_llamacpp_backend(self) -> None:
        """Default config = llamacpp; no LLM loaded until ``model_path`` is set."""
        agent = Agent(AgentConfig())
        await agent.initialize()
        try:
            assert agent._llm is None
            assert agent._ollama_backend is None
        finally:
            await agent.close()

    async def test_ollama_backend_selected_when_configured(self) -> None:
        config = AgentConfig(llm=LLMConfig(backend="ollama"))
        agent = Agent(config)
        await agent.initialize()
        try:
            assert agent._ollama_backend is not None
            assert agent._llm is None
        finally:
            await agent.close()

    async def test_ollama_generate_used_when_no_context(self) -> None:
        def handler(request: httpx2.Request) -> httpx2.Response:
            body = json.loads(request.content)
            assert "Question: hello" in body["prompt"]
            return httpx2.Response(
                200,
                json={"response": "hi from ollama", "done": True, "eval_count": 5},
            )

        config = AgentConfig(llm=LLMConfig(backend="ollama"))
        agent = Agent(config)
        await agent.initialize()
        # Swap the backend with one bound to our mock transport.
        await agent._ollama_backend.aclose()  # type: ignore[union-attr]
        agent._ollama_backend = OllamaBackend(model="test-model", transport=_transport(handler))
        try:
            response = await agent.process("hello")
            assert response.content == "hi from ollama"
        finally:
            await agent.close()

    async def test_close_releases_ollama_client(self) -> None:
        config = AgentConfig(llm=LLMConfig(backend="ollama"))
        agent = Agent(config)
        await agent.initialize()
        assert agent._ollama_backend is not None
        await agent.close()
        assert agent._ollama_backend is None


class TestAgentPolicyHook:
    """The ``propose_request_context`` hook is the S-PAL wiring point."""

    async def test_propose_request_context_requires_ollama(self) -> None:
        agent = Agent(AgentConfig())  # defaults to llamacpp with no model_path
        await agent.initialize()
        try:
            with pytest.raises(RuntimeError, match="requires the Ollama backend"):
                await agent.propose_request_context("some prompt")
        finally:
            await agent.close()

    async def test_propose_request_context_returns_validated_model(self) -> None:
        proposal = {
            "identity": {"type": "ephemeral_required", "did": "did:key:zabc"},
            "proofs": [{"type": "zkp", "claim": "age_over_18"}],
            "intended_use": {"training": False, "aggregation": False, "resale": False},
            "offered_retention_seconds": 0,
            "payment_offered": False,
        }

        def handler(request: httpx2.Request) -> httpx2.Response:
            body = json.loads(request.content)
            # The bridge must pass a JSON schema via Ollama's `format` field.
            assert isinstance(body.get("format"), dict)
            return httpx2.Response(
                200,
                json={
                    "response": json.dumps(proposal),
                    "done": True,
                    "eval_count": 25,
                },
            )

        agent = Agent(AgentConfig(llm=LLMConfig(backend="ollama")))
        await agent.initialize()
        await agent._ollama_backend.aclose()  # type: ignore[union-attr]
        agent._ollama_backend = OllamaBackend(model="test-model", transport=_transport(handler))
        try:
            ctx = await agent.propose_request_context(
                "Business needs age >= 18 verification for alcohol purchase"
            )
            assert ctx.identity is not None
            assert ctx.identity.type == IdentityType.EPHEMERAL_REQUIRED
            assert ctx.proofs[0].type == ProofType.ZKP
            assert ctx.proofs[0].claim == "age_over_18"
        finally:
            await agent.close()


class TestAgentConfigFromEnv:
    def test_env_selects_ollama_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PCI_LLM_BACKEND", "ollama")
        monkeypatch.setenv("PCI_LLM_TIER", "small")
        monkeypatch.setenv("PCI_OLLAMA_URL", "http://example.internal:11434")
        monkeypatch.delenv("PCI_OLLAMA_MODEL", raising=False)
        monkeypatch.delenv("PCI_OLLAMA_TIMEOUT", raising=False)

        cfg = AgentConfig.from_env()
        assert cfg.llm.backend == "ollama"
        assert cfg.llm.ollama_tier == "small"
        assert cfg.llm.ollama_base_url == "http://example.internal:11434"

    def test_env_defaults_to_llamacpp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in (
            "PCI_LLM_BACKEND",
            "PCI_LLM_TIER",
            "PCI_OLLAMA_MODEL",
            "PCI_OLLAMA_URL",
            "PCI_OLLAMA_TIMEOUT",
        ):
            monkeypatch.delenv(var, raising=False)
        cfg = AgentConfig.from_env()
        assert cfg.llm.backend == "llamacpp"

    def test_env_unknown_backend_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PCI_LLM_BACKEND", "vllm")
        cfg = AgentConfig.from_env()
        assert cfg.llm.backend == "llamacpp"

    def test_env_timeout_rejects_infinity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Infinite timeouts disable the httpx2 deadline; reject at parse time."""
        monkeypatch.setenv("PCI_OLLAMA_TIMEOUT", "inf")
        with pytest.raises(ValueError, match="finite"):
            AgentConfig.from_env()

    def test_llm_config_rejects_infinite_timeout(self) -> None:
        with pytest.raises(ValueError):
            LLMConfig(ollama_timeout_seconds=float("inf"))


class TestAgentLifecycleSafety:
    async def test_initialize_releases_backend_on_context_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If context connect() fails, the Ollama backend must not leak."""
        agent = Agent(AgentConfig(llm=LLMConfig(backend="ollama")))

        async def boom() -> None:
            raise RuntimeError("context store unavailable")

        monkeypatch.setattr(agent._context_client, "connect", boom)

        with pytest.raises(RuntimeError, match="context store unavailable"):
            await agent.initialize()

        assert agent._ollama_backend is None
        assert not agent._initialized

    async def test_close_runs_backend_release_even_if_disconnect_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A failing disconnect() must not skip backend cleanup."""
        agent = Agent(AgentConfig(llm=LLMConfig(backend="ollama")))
        await agent.initialize()
        assert agent._ollama_backend is not None

        async def boom() -> None:
            raise RuntimeError("disconnect blew up")

        monkeypatch.setattr(agent._context_client, "disconnect", boom)

        with pytest.raises(RuntimeError, match="disconnect blew up"):
            await agent.close()

        assert agent._ollama_backend is None
        assert not agent._initialized
