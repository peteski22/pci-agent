"""Tests for the restored /services and /service-requests routes."""

from fastapi.testclient import TestClient

from pci_agent.api import create_app
from pci_agent.config import AgentConfig, ApprovalConfig
from pci_agent.coordination import (
    ApprovalDecision,
    ApprovalMode,
    DecisionOutcome,
    RequestStatus,
    ServiceRequestStatus,
    VerificationRequest,
)
from pci_agent.services.service_requests import service_status_for
from pci_agent.status import CardanoStatus, EndpointStatus, ServicesStatus


class StubService:
    """Approval decider returning fixed outcomes."""

    def __init__(
        self, outcome: DecisionOutcome, approve_outcome: DecisionOutcome | None = None
    ) -> None:
        self._outcome = outcome
        self._approve_outcome = approve_outcome if approve_outcome is not None else outcome

    async def decide(self, request: VerificationRequest) -> ApprovalDecision:
        return ApprovalDecision(outcome=self._outcome, reason="stub", proof={"ok": True})

    async def approve(self, request: VerificationRequest) -> ApprovalDecision:
        return ApprovalDecision(
            outcome=self._approve_outcome, reason="stub-approve", proof={"ok": True}
        )


class RaisingService:
    """Approval decider whose evaluation always raises an unexpected error."""

    async def decide(self, request: VerificationRequest) -> ApprovalDecision:
        raise RuntimeError("boom")

    async def approve(self, request: VerificationRequest) -> ApprovalDecision:
        raise RuntimeError("boom")


class StubStatusSource:
    """Status source returning a fixed healthy aggregate."""

    async def check(self) -> ServicesStatus:
        return ServicesStatus(
            agent=EndpointStatus(status="healthy", url="http://localhost:8082"),
            zkp=EndpointStatus(status="healthy", url="http://zkp"),
            cardano=CardanoStatus(status="healthy", url="http://cardano", latest_block=42),
        )


def _client(
    mode: ApprovalMode = ApprovalMode.MANUAL,
    service: StubService | RaisingService | None = None,
) -> TestClient:
    config = AgentConfig(approval=ApprovalConfig(mode=mode))
    factory = None if service is None else (lambda: service)
    app = create_app(config, service_factory=factory, status_source=StubStatusSource())
    return TestClient(app)


def _service_payload() -> dict[str, str]:
    return {
        "user_id": "demo-user",
        "user_name": "Alice Demo",
        "business_id": "demo-business",
        "service_type": "purchase",
        "service_name": "Purchase Alcohol",
    }


def _verification_payload(service_request_id: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "business_id": "biz",
        "business_name": "Biz",
        "claim": {"type": "age", "params": {"minAge": 18}},
    }
    if service_request_id is not None:
        payload["service_request_id"] = service_request_id
    return payload


def test_services_returns_status_aggregate() -> None:
    client = _client()
    body = client.get("/services").json()
    assert body["agent"]["status"] == "healthy"
    assert body["zkp"]["status"] == "healthy"
    assert body["cardano"]["latest_block"] == 42


def test_create_service_request_is_pending_with_full_uuid() -> None:
    client = _client()
    resp = client.post("/service-requests", json=_service_payload())
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "pending"
    assert body["verification_request_id"] is None
    assert len(body["id"]) == 32
    assert all(c in "0123456789abcdef" for c in body["id"])


def test_list_service_requests_filters_by_status() -> None:
    client = _client()
    client.post("/service-requests", json=_service_payload())
    client.post("/service-requests", json=_service_payload())

    listed = client.get("/service-requests").json()["requests"]
    assert len(listed) == 2

    pending = client.get("/service-requests", params={"status": "pending"}).json()["requests"]
    assert len(pending) == 2
    verified = client.get("/service-requests", params={"status": "verified"}).json()["requests"]
    assert verified == []


def test_get_service_request_by_id() -> None:
    client = _client()
    created = client.post("/service-requests", json=_service_payload()).json()
    resp = client.get(f"/service-requests/{created['id']}")
    assert resp.status_code == 200
    assert resp.json()["id"] == created["id"]


def test_get_missing_service_request_returns_404() -> None:
    client = _client()
    assert client.get("/service-requests/nope").status_code == 404


def test_linked_verification_marks_verification_required() -> None:
    client = _client()
    service_req = client.post("/service-requests", json=_service_payload()).json()
    verification = client.post("/requests", json=_verification_payload(service_req["id"])).json()

    updated = client.get(f"/service-requests/{service_req['id']}").json()
    assert updated["status"] == "verification_required"
    assert updated["verification_request_id"] == verification["id"]


def test_linked_verification_unknown_service_request_returns_404() -> None:
    client = _client()
    resp = client.post("/requests", json=_verification_payload("nope"))
    assert resp.status_code == 404


def test_approve_marks_service_request_verified() -> None:
    client = _client(service=StubService(DecisionOutcome.APPROVE))
    service_req = client.post("/service-requests", json=_service_payload()).json()
    verification = client.post("/requests", json=_verification_payload(service_req["id"])).json()

    resp = client.post(f"/requests/{verification['id']}/approve")
    assert resp.status_code == 200
    assert resp.json()["status"] == "approved"

    updated = client.get(f"/service-requests/{service_req['id']}").json()
    assert updated["status"] == "verified"


def test_approve_unverified_proof_marks_service_request_rejected() -> None:
    client = _client(
        service=StubService(DecisionOutcome.APPROVE, approve_outcome=DecisionOutcome.REJECT)
    )
    service_req = client.post("/service-requests", json=_service_payload()).json()
    verification = client.post("/requests", json=_verification_payload(service_req["id"])).json()

    client.post(f"/requests/{verification['id']}/approve")
    updated = client.get(f"/service-requests/{service_req['id']}").json()
    assert updated["status"] == "rejected"


def test_deny_marks_service_request_denied() -> None:
    client = _client(service=StubService(DecisionOutcome.APPROVE))
    service_req = client.post("/service-requests", json=_service_payload()).json()
    verification = client.post("/requests", json=_verification_payload(service_req["id"])).json()

    client.post(f"/requests/{verification['id']}/deny")
    updated = client.get(f"/service-requests/{service_req['id']}").json()
    assert updated["status"] == "denied"


def test_autonomous_approval_marks_service_request_verified_on_create() -> None:
    client = _client(
        mode=ApprovalMode.FULLY_AUTONOMOUS, service=StubService(DecisionOutcome.APPROVE)
    )
    service_req = client.post("/service-requests", json=_service_payload()).json()
    resp = client.post("/requests", json=_verification_payload(service_req["id"]))
    assert resp.json()["status"] == "approved"

    updated = client.get(f"/service-requests/{service_req['id']}").json()
    assert updated["status"] == "verified"


def test_evaluation_error_leaves_service_request_awaiting_verification() -> None:
    client = _client(mode=ApprovalMode.FULLY_AUTONOMOUS, service=RaisingService())
    service_req = client.post("/service-requests", json=_service_payload()).json()
    resp = client.post("/requests", json=_verification_payload(service_req["id"]))
    assert resp.status_code == 500

    updated = client.get(f"/service-requests/{service_req['id']}").json()
    assert updated["status"] == "verification_required"


def test_complete_verified_service_request() -> None:
    client = _client(service=StubService(DecisionOutcome.APPROVE))
    service_req = client.post("/service-requests", json=_service_payload()).json()
    verification = client.post("/requests", json=_verification_payload(service_req["id"])).json()
    client.post(f"/requests/{verification['id']}/approve")

    resp = client.post(f"/service-requests/{service_req['id']}/complete")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["completed_at"] is not None


def test_repeat_complete_is_idempotent() -> None:
    client = _client(service=StubService(DecisionOutcome.APPROVE))
    service_req = client.post("/service-requests", json=_service_payload()).json()
    verification = client.post("/requests", json=_verification_payload(service_req["id"])).json()
    client.post(f"/requests/{verification['id']}/approve")

    first = client.post(f"/service-requests/{service_req['id']}/complete")
    second = client.post(f"/service-requests/{service_req['id']}/complete")
    assert second.status_code == 200
    assert second.json() == first.json()


def test_complete_unverified_service_request_returns_409() -> None:
    client = _client()
    service_req = client.post("/service-requests", json=_service_payload()).json()
    resp = client.post(f"/service-requests/{service_req['id']}/complete")
    assert resp.status_code == 409
    assert resp.json()["detail"] == "service request not verified"


def test_complete_missing_service_request_returns_404() -> None:
    client = _client()
    assert client.post("/service-requests/nope/complete").status_code == 404


def test_linked_verification_on_completed_service_request_returns_409() -> None:
    client = _client(service=StubService(DecisionOutcome.APPROVE))
    service_req = client.post("/service-requests", json=_service_payload()).json()
    verification = client.post("/requests", json=_verification_payload(service_req["id"])).json()
    client.post(f"/requests/{verification['id']}/approve")
    client.post(f"/service-requests/{service_req['id']}/complete")

    resp = client.post("/requests", json=_verification_payload(service_req["id"]))
    assert resp.status_code == 409
    assert resp.json()["detail"] == "service request already resolved"

    updated = client.get(f"/service-requests/{service_req['id']}").json()
    assert updated["status"] == "completed"


def test_stale_verification_does_not_drive_relinked_service_request() -> None:
    client = _client(service=StubService(DecisionOutcome.APPROVE))
    service_req = client.post("/service-requests", json=_service_payload()).json()
    first = client.post("/requests", json=_verification_payload(service_req["id"])).json()
    second = client.post("/requests", json=_verification_payload(service_req["id"])).json()

    client.post(f"/requests/{first['id']}/deny")
    updated = client.get(f"/service-requests/{service_req['id']}").json()
    assert updated["status"] == "verification_required"
    assert updated["verification_request_id"] == second["id"]

    client.post(f"/requests/{second['id']}/approve")
    updated = client.get(f"/service-requests/{service_req['id']}").json()
    assert updated["status"] == "verified"


def test_service_status_for_leaves_unresolved_outcomes_unmapped() -> None:
    assert service_status_for(RequestStatus.APPROVED) is ServiceRequestStatus.VERIFIED
    assert service_status_for(RequestStatus.REJECTED) is ServiceRequestStatus.REJECTED
    assert service_status_for(RequestStatus.DENIED) is ServiceRequestStatus.DENIED
    assert service_status_for(RequestStatus.PENDING) is None
    assert service_status_for(RequestStatus.ESCALATED) is None
    assert service_status_for(RequestStatus.ERROR) is None
