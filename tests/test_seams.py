from datetime import UTC, datetime

import pytest

from pci_agent.context import ContextItem
from pci_agent.coordination import VerificationClaim, VerificationRequest
from pci_agent.errors import ContextUnavailable
from pci_agent.seams import (
    ContextStoreDataProvider,
    DeterministicContextBuilder,
    PassThroughEnvelopeVerifier,
)
from pci_agent.spal import RequestContext


class _FakeContextClient:
    """Minimal stand-in for ContextClient.search, without a real connection."""

    def __init__(
        self,
        items: list[ContextItem] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._items = items or []
        self._error = error

    async def search(
        self, query: str, scope: str | None = None, limit: int = 10
    ) -> list[ContextItem]:
        if self._error is not None:
            raise self._error
        return self._items


def _req() -> VerificationRequest:
    now = datetime.now(UTC)
    return VerificationRequest(
        id="a",
        business_id="biz",
        business_name="Biz",
        claim=VerificationClaim(type="age", params={"minAge": 18}),
        context_scope="health/age",
        created_at=now,
        expires_at=now,
    )


def test_deterministic_builder_returns_request_context():
    ctx = DeterministicContextBuilder().build(_req())
    assert isinstance(ctx, RequestContext)


async def test_passthrough_verifier_accepts_anything():
    # Must not raise.
    await PassThroughEnvelopeVerifier().verify({}, b"")


async def test_context_store_provider_maps_items_to_dict():
    items = [
        ContextItem(id="1", content="alice", score=0.9),
        ContextItem(id="2", content="bob", score=0.5),
    ]
    fake = _FakeContextClient(items=items)

    result = await ContextStoreDataProvider(fake).fetch("health/age")

    assert isinstance(result, dict)
    assert result == {"1": "alice", "2": "bob"}


async def test_context_store_provider_raises_context_unavailable_on_failure():
    fake = _FakeContextClient(error=RuntimeError("boom"))

    with pytest.raises(ContextUnavailable) as exc_info:
        await ContextStoreDataProvider(fake).fetch("health/age")

    assert isinstance(exc_info.value.__cause__, RuntimeError)
