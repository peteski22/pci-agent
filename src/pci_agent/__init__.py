"""
PCI Agent - Layer 2: Local AI agent for Personal Context Infrastructure
"""

from pci_agent.agent import Agent, AgentResponse
from pci_agent.config import AgentConfig, LLMConfig
from pci_agent.context import ContextClient
from pci_agent.models import (
    MODEL_TIERS,
    LLMBackend,
    LLMResponse,
    LocalLLM,
    OllamaBackend,
    OllamaError,
    OllamaModelRefusalError,
    OllamaSchemaError,
    OllamaTimeoutError,
    OllamaTransportError,
    StructuredResponse,
    propose_request_context,
)
from pci_agent.policy import PolicyChecker, PolicyCheckResult
from pci_agent.spal import RequestContext, SPALPolicy

__version__ = "0.1.0"

__all__ = [
    "MODEL_TIERS",
    "Agent",
    "AgentConfig",
    "AgentResponse",
    "ContextClient",
    "LLMBackend",
    "LLMConfig",
    "LLMResponse",
    "LocalLLM",
    "OllamaBackend",
    "OllamaError",
    "OllamaModelRefusalError",
    "OllamaSchemaError",
    "OllamaTimeoutError",
    "OllamaTransportError",
    "PolicyCheckResult",
    "PolicyChecker",
    "RequestContext",
    "SPALPolicy",
    "StructuredResponse",
    "propose_request_context",
]
