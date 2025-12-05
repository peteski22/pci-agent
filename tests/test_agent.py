"""Tests for the PCI Agent"""

import pytest

from pci_agent import Agent, AgentConfig
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
