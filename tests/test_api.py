"""Tests for the FastAPI coordination app."""

from fastapi.testclient import TestClient

from pci_agent.api import create_app
from pci_agent.config import AgentConfig, ApprovalConfig
from pci_agent.coordination import (
    ApprovalDecision,
    ApprovalMode,
    DecisionOutcome,
    RequestStatus,
)
from pci_agent.store import RequestRepository


def _payload() -> dict:
    return {
        "business_id": "biz",
        "business_name": "Biz",
        "claim": {"type": "age", "params": {"minAge": 18}},
        "policy_id": "p1",
        "context_scope": "health/age",
    }


class StubService:
    def __init__(
        self, outcome: DecisionOutcome, approve_outcome: DecisionOutcome | None = None
    ) -> None:
        self._outcome = outcome
        self._approve_outcome = approve_outcome if approve_outcome is not None else outcome

    async def decide(self, request):
        return ApprovalDecision(outcome=self._outcome, reason="stub", proof={"ok": True})

    async def approve(self, request):
        return ApprovalDecision(
            outcome=self._approve_outcome, reason="stub-approve", proof={"ok": True}
        )


def test_health():
    app = create_app(AgentConfig())
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "healthy"


def test_manual_mode_parks_pending():
    app = create_app(AgentConfig(approval=ApprovalConfig(mode=ApprovalMode.MANUAL)))
    client = TestClient(app)
    resp = client.post("/requests", json=_payload())
    assert resp.status_code == 201
    assert resp.json()["status"] == "pending"


def test_fully_autonomous_approves_on_decide():
    config = AgentConfig(approval=ApprovalConfig(mode=ApprovalMode.FULLY_AUTONOMOUS))
    app = create_app(config, service_factory=lambda: StubService(DecisionOutcome.APPROVE))
    client = TestClient(app)
    resp = client.post("/requests", json=_payload())
    assert resp.status_code == 201
    assert resp.json()["status"] == "approved"


def test_fully_autonomous_denies_on_decide():
    config = AgentConfig(approval=ApprovalConfig(mode=ApprovalMode.FULLY_AUTONOMOUS))
    app = create_app(config, service_factory=lambda: StubService(DecisionOutcome.DENY))
    client = TestClient(app)
    resp = client.post("/requests", json=_payload())
    assert resp.json()["status"] == "denied"


def test_get_missing_request_returns_404():
    app = create_app(AgentConfig())
    client = TestClient(app)
    assert client.get("/requests/nope").status_code == 404


def test_manual_approve_verified_sets_approved():
    config = AgentConfig(approval=ApprovalConfig(mode=ApprovalMode.MANUAL))
    app = create_app(
        config,
        service_factory=lambda: StubService(
            DecisionOutcome.APPROVE, approve_outcome=DecisionOutcome.APPROVE
        ),
    )
    client = TestClient(app)
    created = client.post("/requests", json=_payload()).json()
    resp = client.post(f"/requests/{created['id']}/approve")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "approved"
    assert body["response"]["proof"] == {"ok": True}


def test_manual_approve_unverified_sets_rejected():
    config = AgentConfig(approval=ApprovalConfig(mode=ApprovalMode.MANUAL))
    app = create_app(
        config,
        service_factory=lambda: StubService(
            DecisionOutcome.APPROVE, approve_outcome=DecisionOutcome.REJECT
        ),
    )
    client = TestClient(app)
    created = client.post("/requests", json=_payload()).json()
    resp = client.post(f"/requests/{created['id']}/approve")
    assert resp.status_code == 200
    assert resp.json()["status"] == "rejected"


def test_manual_deny_sets_denied():
    config = AgentConfig(approval=ApprovalConfig(mode=ApprovalMode.MANUAL))
    app = create_app(config, service_factory=lambda: StubService(DecisionOutcome.APPROVE))
    client = TestClient(app)
    created = client.post("/requests", json=_payload()).json()
    resp = client.post(f"/requests/{created['id']}/deny")
    assert resp.status_code == 200
    assert resp.json()["status"] == "denied"


def test_approve_missing_request_404():
    app = create_app(AgentConfig())
    client = TestClient(app)
    resp = client.post("/requests/nope/approve")
    assert resp.status_code == 404


def test_approve_non_pending_returns_409():
    config = AgentConfig(approval=ApprovalConfig(mode=ApprovalMode.FULLY_AUTONOMOUS))
    app = create_app(config, service_factory=lambda: StubService(DecisionOutcome.APPROVE))
    client = TestClient(app)
    created = client.post("/requests", json=_payload()).json()
    assert created["status"] == "approved"
    resp = client.post(f"/requests/{created['id']}/approve")
    assert resp.status_code == 409


def test_deny_non_pending_returns_409():
    config = AgentConfig(approval=ApprovalConfig(mode=ApprovalMode.MANUAL))
    app = create_app(config, service_factory=lambda: StubService(DecisionOutcome.APPROVE))
    client = TestClient(app)
    created = client.post("/requests", json=_payload()).json()
    first = client.post(f"/requests/{created['id']}/deny")
    assert first.status_code == 200
    assert first.json()["status"] == "denied"
    resp = client.post(f"/requests/{created['id']}/deny")
    assert resp.status_code == 409


def test_approve_resolves_escalated_request():
    config = AgentConfig(approval=ApprovalConfig(mode=ApprovalMode.AUTO_WITH_NOTIFICATION))
    app = create_app(
        config,
        service_factory=lambda: StubService(
            DecisionOutcome.DENY, approve_outcome=DecisionOutcome.APPROVE
        ),
    )
    client = TestClient(app)
    created = client.post("/requests", json=_payload()).json()
    assert created["status"] == "escalated"

    resp = client.post(f"/requests/{created['id']}/approve")
    assert resp.status_code == 200
    assert resp.json()["status"] == "approved"


def test_deny_resolves_escalated_request():
    config = AgentConfig(approval=ApprovalConfig(mode=ApprovalMode.AUTO_WITH_NOTIFICATION))
    app = create_app(
        config,
        service_factory=lambda: StubService(DecisionOutcome.DENY),
    )
    client = TestClient(app)
    created = client.post("/requests", json=_payload()).json()
    assert created["status"] == "escalated"

    resp = client.post(f"/requests/{created['id']}/deny")
    assert resp.status_code == 200
    assert resp.json()["status"] == "denied"


class RaisingService:
    """Service whose evaluation always raises an unexpected error."""

    async def decide(self, request):
        raise RuntimeError("boom")

    async def approve(self, request):
        raise RuntimeError("boom")


def test_request_id_is_full_uuid():
    app = create_app(AgentConfig(approval=ApprovalConfig(mode=ApprovalMode.MANUAL)))
    client = TestClient(app)
    rid = client.post("/requests", json=_payload()).json()["id"]
    assert len(rid) == 32
    assert all(c in "0123456789abcdef" for c in rid)


def test_manual_approve_denied_outcome_is_terminal_not_escalated():
    # Under auto-with-notification, a human's denial must resolve to DENIED, not re-escalate.
    config = AgentConfig(approval=ApprovalConfig(mode=ApprovalMode.AUTO_WITH_NOTIFICATION))
    app = create_app(
        config,
        service_factory=lambda: StubService(
            DecisionOutcome.DENY, approve_outcome=DecisionOutcome.DENY
        ),
    )
    client = TestClient(app)
    created = client.post("/requests", json=_payload()).json()
    assert created["status"] == "escalated"
    resp = client.post(f"/requests/{created['id']}/approve")
    assert resp.status_code == 200
    assert resp.json()["status"] == "denied"


def test_create_request_records_error_on_unexpected_failure():
    repo = RequestRepository()
    config = AgentConfig(approval=ApprovalConfig(mode=ApprovalMode.FULLY_AUTONOMOUS))
    app = create_app(config, service_factory=lambda: RaisingService(), repository=repo)
    client = TestClient(app)
    resp = client.post("/requests", json=_payload())
    assert resp.status_code == 500
    stored = repo.list()
    assert len(stored) == 1
    assert stored[0].status is RequestStatus.ERROR


def test_approve_records_error_on_unexpected_failure():
    repo = RequestRepository()
    config = AgentConfig(approval=ApprovalConfig(mode=ApprovalMode.MANUAL))
    app = create_app(config, service_factory=lambda: RaisingService(), repository=repo)
    client = TestClient(app)
    created = client.post("/requests", json=_payload()).json()
    assert created["status"] == "pending"
    resp = client.post(f"/requests/{created['id']}/approve")
    assert resp.status_code == 500
    stored = repo.get(created["id"])
    assert stored is not None
    assert stored.status is RequestStatus.ERROR


def test_default_app_lifespan_closes_cleanly():
    # Exercises the default factory + shared ZKPClient lifespan (startup/shutdown).
    app = create_app(AgentConfig(approval=ApprovalConfig(mode=ApprovalMode.MANUAL)))
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "healthy"
