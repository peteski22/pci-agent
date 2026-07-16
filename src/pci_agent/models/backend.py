"""
Shared model-backend contract.

Both ``LocalLLM`` (llama-cpp-python) and ``OllamaBackend`` satisfy the
:class:`LLMBackend` protocol, letting the :class:`~pci_agent.agent.Agent`
dispatch on config without knowing which runtime is in use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class LLMResponse:
    """Result of an unstructured ``generate`` call."""

    text: str
    tokens_used: int
    finish_reason: str


@dataclass
class ChatMessage:
    """A single chat-format message."""

    role: Role
    content: str


@dataclass
class StructuredResponse:
    """Result of a JSON-schema constrained generation call.

    ``data`` is the parsed JSON object the model produced. ``raw_text`` is
    kept for diagnostic and audit purposes (S-PAL flows need an audit trail
    of what the model actually emitted).
    """

    data: dict[str, Any]
    raw_text: str
    model: str
    tokens_used: int
    finish_reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LLMBackend(Protocol):
    """Minimum interface every backend must implement."""

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Free-form text generation from a plain prompt."""
        ...

    async def aclose(self) -> None:
        """Release any transport / model resources held by the backend."""
        ...
