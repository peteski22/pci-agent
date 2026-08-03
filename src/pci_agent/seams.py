"""Seams between the approval service and external systems.

Each Protocol is a narrow interface with a Phase 1 implementation. Phase 2
swaps in the real envelope verifier and richer context handling without
changing the service.
"""

from __future__ import annotations

from typing import Protocol

from pci_agent.context import ContextItem
from pci_agent.coordination import VerificationRequest
from pci_agent.errors import ContextUnavailable
from pci_agent.policy import PolicyCheckResult
from pci_agent.spal import RequestContext
from pci_agent.zkp import ZKPResult


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


class PolicyCheck(Protocol):
    """Evaluates a request against an S-PAL policy."""

    async def check(
        self,
        policy_id: str,
        query: str,
        context_scope: str | None = None,
        request_context: RequestContext | None = None,
    ) -> PolicyCheckResult:
        """Return whether the query is permitted by the named policy."""
        ...


class ProofGenerator(Protocol):
    """Generates a zero-knowledge proof for a claim."""

    async def generate(self, proof_type: str, proof_data: dict[str, object]) -> ZKPResult:
        """Produce a proof of the given type over the provided inputs."""
        ...


class ContextSearcher(Protocol):
    """Searches the context store for a scope's items."""

    async def search(
        self, query: str, scope: str | None = None, limit: int = 10
    ) -> list[ContextItem]:
        """Return items matching the query within an optional scope."""
        ...


class ContextStoreDataProvider:
    """Phase 1 PrivateDataProvider backed by the (mocked) context store."""

    def __init__(self, client: ContextSearcher) -> None:
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
    """Phase 1 RequestContextBuilder: returns an empty RequestContext, no LLM.

    Phase 1 does not populate any RequestContext field from the verification
    request; it always returns an empty context. The natural-language LLM
    path (Agent.propose_request_context) is reserved for free-text requests
    and is intentionally off the autonomous-approval critical path, so the
    flow needs no live model.

    Warning:
        Because the returned context is empty, S-PAL policy rules whose
        conditions depend on request context (e.g. derivative-use or
        retention conditions) short-circuit on the missing values and are
        NOT enforced — they fall through to a conclusive allow. Until
        Phase 2 populates the context, condition-bearing rules must not be
        trusted as a conclusive allow.
    """

    def build(self, request: VerificationRequest) -> RequestContext:
        """Return an empty RequestContext; Phase 1 maps no request fields.

        Warning:
            The empty context causes condition-bearing policy rules
            (derivative-use, retention) to be skipped rather than enforced,
            which falls through to allow. See the class docstring.
        """
        return RequestContext()


class PassThroughEnvelopeVerifier:
    """Phase 1 EnvelopeVerifier: accepts every request. Enforced in Phase 2."""

    async def verify(self, headers: dict[str, str], body: bytes) -> None:
        """Accept unconditionally (Phase 1)."""
        return None
