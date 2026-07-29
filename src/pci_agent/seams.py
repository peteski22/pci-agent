"""Seams between the approval service and external systems.

Each Protocol is a narrow interface with a Phase 1 implementation. Phase 2
swaps in the real envelope verifier and richer context handling without
changing the service.
"""

from __future__ import annotations

from typing import Protocol

from pci_agent.context import ContextClient
from pci_agent.coordination import VerificationRequest
from pci_agent.errors import ContextUnavailable
from pci_agent.spal import RequestContext


class PrivateDataProvider(Protocol):
    """Supplies the user's private data for a context scope."""

    async def fetch(self, scope: str | None) -> dict[str, object]:
        """Return the private data needed to prove a claim in this scope."""
        ...


class RequestContextBuilder(Protocol):
    """Builds a policy-evaluation RequestContext from a verification request."""

    def build(self, request: VerificationRequest) -> RequestContext:
        """Map a verification request to a RequestContext for the policy checker."""
        ...


class EnvelopeVerifier(Protocol):
    """Authenticates an inbound agent message."""

    async def verify(self, headers: dict[str, str], body: bytes) -> None:
        """Raise if the request envelope is not authentic. No-op when disabled."""
        ...


class ContextStoreDataProvider:
    """Phase 1 PrivateDataProvider backed by the (mocked) context store."""

    def __init__(self, client: ContextClient) -> None:
        self._client = client

    async def fetch(self, scope: str | None) -> dict[str, object]:
        """Fetch private data for a scope from the context store.

        Raises:
            ContextUnavailable: if the store cannot be queried.
        """
        try:
            items = await self._client.search(query="", scope=scope, limit=50)
        except RuntimeError as exc:
            raise ContextUnavailable("private data unavailable") from exc
        return {item.id: item.content for item in items}


class DeterministicContextBuilder:
    """Phase 1 RequestContextBuilder: maps request fields with no LLM.

    The natural-language LLM path (Agent.propose_request_context) is reserved
    for free-text requests and is intentionally off the autonomous-approval
    critical path so the flow needs no live model.
    """

    def build(self, request: VerificationRequest) -> RequestContext:
        """Produce a minimal RequestContext from a structured request."""
        return RequestContext()


class PassThroughEnvelopeVerifier:
    """Phase 1 EnvelopeVerifier: accepts every request. Enforced in Phase 2."""

    async def verify(self, headers: dict[str, str], body: bytes) -> None:
        """Accept unconditionally (Phase 1)."""
        return None
