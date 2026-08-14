import pytest

from pci_agent.config import AgentConfig
from pci_agent.coordination import ApprovalMode


def test_default_mode_is_manual():
    assert AgentConfig().approval.mode is ApprovalMode.MANUAL


def test_mode_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PCI_APPROVAL_MODE", "fully_autonomous")
    assert AgentConfig.from_env().approval.mode is ApprovalMode.FULLY_AUTONOMOUS


def test_unknown_mode_falls_back_to_manual(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PCI_APPROVAL_MODE", "bogus")
    assert AgentConfig.from_env().approval.mode is ApprovalMode.MANUAL


def test_context_defaults():
    context = AgentConfig().context
    assert context.endpoint == "http://localhost:8081"
    assert context.user_id == "default"
    assert context.password is None


def test_context_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CONTEXT_STORE_URL", "http://context-store:8081")
    monkeypatch.setenv("PCI_CONTEXT_USER_ID", "demo-user")
    monkeypatch.setenv("PCI_CONTEXT_PASSWORD", "vault-pass")

    context = AgentConfig.from_env().context
    assert context.endpoint == "http://context-store:8081"
    assert context.user_id == "demo-user"
    assert context.password == "vault-pass"
