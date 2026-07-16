"""
Model backends for the PCI Agent.

The agent supports two local inference backends:

- ``LocalLLM`` — in-process ``llama-cpp-python`` (GGUF file, no daemon).
- ``OllamaBackend`` — HTTP client for a local Ollama daemon (recommended
  default per the runtime research doc; enables JSON-schema constrained
  outputs via Ollama's ``format`` parameter).

Both implementations satisfy the :class:`LLMBackend` protocol so callers can
swap them transparently. See ``ollama.py`` for the structured-output entry
point that the S-PAL flow uses to synthesise ``RequestContext`` candidates
for :class:`pci_agent.policy.PolicyChecker`.
"""

from pci_agent.models.backend import (
    ChatMessage,
    LLMBackend,
    LLMResponse,
    StructuredResponse,
)
from pci_agent.models.local_llm import LocalLLM
from pci_agent.models.ollama import (
    DEFAULT_OLLAMA_URL,
    MODEL_TIERS,
    OllamaBackend,
    OllamaModelRefusalError,
    OllamaSchemaError,
    OllamaTimeoutError,
    OllamaTransportError,
    resolve_model_for_tier,
)
from pci_agent.models.spal_bridge import propose_request_context

__all__ = [
    "DEFAULT_OLLAMA_URL",
    "MODEL_TIERS",
    "ChatMessage",
    "LLMBackend",
    "LLMResponse",
    "LocalLLM",
    "OllamaBackend",
    "OllamaModelRefusalError",
    "OllamaSchemaError",
    "OllamaTimeoutError",
    "OllamaTransportError",
    "StructuredResponse",
    "propose_request_context",
    "resolve_model_for_tier",
]
