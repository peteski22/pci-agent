"""Tests for the PCI Agent"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from pci_agent import Agent, AgentConfig
from pci_agent.config import LLMConfig
from pci_agent.context import ContextItem
from pci_agent.models.local_llm import LLMResponse, LocalLLM
from pci_agent.policy import PolicyChecker


class TestAgent:
    """Tests for the Agent class"""

    @pytest.fixture
    def agent(self) -> Agent:
        return Agent(AgentConfig())

    async def test_agent_initialization(self, agent: Agent) -> None:
        """Test that agent initializes correctly"""
        await agent.initialize()
        assert agent._initialized is True

    async def test_agent_process_without_policy(self, agent: Agent) -> None:
        """Test processing a query without policy"""
        response = await agent.process("What is the weather?")
        assert response.content is not None
        assert response.policy_applied is None

    async def test_agent_close(self, agent: Agent) -> None:
        """Test that agent closes cleanly"""
        await agent.initialize()
        await agent.close()
        assert agent._initialized is False

    async def test_no_llm_without_model_path(self, agent: Agent) -> None:
        """Test that no LLM is loaded when model_path is not set"""
        await agent.initialize()
        assert agent._llm is None

    async def test_fallback_response_without_llm(self, agent: Agent) -> None:
        """Test that agent returns fallback response when no LLM loaded"""
        response = await agent.process("Hello")
        assert "[No LLM loaded]" in response.content
        assert "Hello" in response.content


class TestAgentWithLLM:
    """Tests for the Agent class with LLM integration"""

    @pytest.fixture
    def llm_config(self, tmp_path: Path) -> AgentConfig:
        return AgentConfig(
            llm=LLMConfig(
                model_path=str(tmp_path / "test-model.gguf"),
                context_length=2048,
                n_gpu_layers=0,
            )
        )

    async def test_llm_loaded_when_model_path_set(self, llm_config: AgentConfig) -> None:
        """Test that LocalLLM is instantiated and loaded when model_path is configured"""
        agent = Agent(llm_config)

        with patch.object(LocalLLM, "load", new_callable=AsyncMock) as mock_load:
            await agent.initialize()

            mock_load.assert_called_once()
            assert agent._llm is not None
            assert isinstance(agent._llm, LocalLLM)
            assert agent._llm.config.model_path is not None
            assert agent._llm.config.model_path.endswith("test-model.gguf")

    async def test_generate_response_calls_llm(self, llm_config: AgentConfig) -> None:
        """Test that _generate_response uses the LLM when loaded"""
        agent = Agent(llm_config)

        mock_response = LLMResponse(text="Generated answer", tokens_used=42, finish_reason="stop")

        with patch.object(LocalLLM, "load", new_callable=AsyncMock):
            await agent.initialize()

        with patch.object(agent._llm, "generate", new_callable=AsyncMock) as mock_gen:  # type: ignore[union-attr]
            mock_gen.return_value = mock_response
            response = await agent.process("What is 2+2?")

            mock_gen.assert_called_once()
            prompt_arg = mock_gen.call_args.args[0]
            assert "Question: What is 2+2?" in prompt_arg
            assert response.content == "Generated answer"

    async def test_prompt_includes_context(self) -> None:
        """Test that _build_prompt incorporates context items"""
        context_items = [
            ContextItem(id="1", content="The sky is blue", score=0.9),
            ContextItem(id="2", content="Water is wet", score=0.8),
        ]
        prompt = Agent._build_prompt("Why is the sky blue?", context_items)

        assert "Context:" in prompt
        assert "- The sky is blue" in prompt
        assert "- Water is wet" in prompt
        assert "Question: Why is the sky blue?" in prompt
        assert "Answer:" in prompt

    async def test_prompt_without_context(self) -> None:
        """Test that _build_prompt works without context items"""
        prompt = Agent._build_prompt("Hello", [])

        assert "Context:" not in prompt
        assert "Question: Hello" in prompt
        assert "Answer:" in prompt

    async def test_close_unloads_llm(self, llm_config: AgentConfig) -> None:
        """Test that close() calls unload on the LLM"""
        agent = Agent(llm_config)

        with patch.object(LocalLLM, "load", new_callable=AsyncMock):
            await agent.initialize()

        with patch.object(agent._llm, "unload", new_callable=AsyncMock) as mock_unload:  # type: ignore[union-attr]
            await agent.close()

            mock_unload.assert_called_once()
            assert agent._llm is None
            assert agent._initialized is False

    async def test_llm_import_error_propagates(self, llm_config: AgentConfig) -> None:
        """Test that ImportError from missing llama-cpp-python propagates"""
        agent = Agent(llm_config)

        with patch.object(
            LocalLLM,
            "load",
            new_callable=AsyncMock,
            side_effect=ImportError("llama-cpp-python not installed"),
        ), pytest.raises(ImportError, match="llama-cpp-python"):
            await agent.initialize()


class TestPolicyChecker:
    """Tests for the PolicyChecker class"""

    @pytest.fixture
    def checker(self) -> PolicyChecker:
        return PolicyChecker()

    async def test_check_missing_policy(self, checker: PolicyChecker) -> None:
        """Test checking against a non-existent policy"""
        result = await checker.check("non-existent", "test query")
        assert result.allowed is True
        assert "not found" in (result.reason or "")

    async def test_load_and_check_policy(self, checker: PolicyChecker) -> None:
        """Test loading and checking a policy"""
        policy_data = {
            "version": "1.0",
            "id": "test-policy",
            "name": "Test Policy",
            "rules": [],
        }
        await checker.load_policy("test-policy", policy_data)

        result = await checker.check("test-policy", "test query")
        assert result.policy_id == "test-policy"

    async def test_list_policies(self, checker: PolicyChecker) -> None:
        """Test listing loaded policies"""
        await checker.load_policy("policy-1", {})
        await checker.load_policy("policy-2", {})

        policies = await checker.list_policies()
        assert "policy-1" in policies
        assert "policy-2" in policies
