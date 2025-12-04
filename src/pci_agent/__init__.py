"""
PCI Agent - Layer 2: Local AI agent for Personal Context Infrastructure
"""

from pci_agent.agent import Agent
from pci_agent.config import AgentConfig
from pci_agent.context import ContextClient
from pci_agent.policy import PolicyChecker

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentConfig",
    "ContextClient",
    "PolicyChecker",
]
