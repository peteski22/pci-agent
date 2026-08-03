from datetime import UTC, datetime

from pci_agent.coordination import (
    RequestStatus,
    ServiceRequest,
    ServiceRequestStatus,
    VerificationClaim,
    VerificationRequest,
)
from pci_agent.store import RequestRepository, ServiceRequestRepository


def _req(request_id: str) -> VerificationRequest:
    now = datetime.now(UTC)
    return VerificationRequest(
        id=request_id,
        business_id="biz",
        business_name="Biz",
        claim=VerificationClaim(type="age"),
        created_at=now,
        expires_at=now,
    )


def test_add_and_get():
    repo = RequestRepository()
    repo.add(_req("a"))
    assert repo.get("a").id == "a"
    assert repo.get("missing") is None


def test_replace_updates_in_place():
    repo = RequestRepository()
    repo.add(_req("a"))
    updated = repo.get("a").model_copy(update={"status": RequestStatus.APPROVED})
    repo.replace(updated)
    assert repo.get("a").status is RequestStatus.APPROVED
    assert len(repo.list()) == 1


def _service_req(request_id: str) -> ServiceRequest:
    now = datetime.now(UTC)
    return ServiceRequest(
        id=request_id,
        user_id="user",
        user_name="User",
        business_id="biz",
        service_type="purchase",
        service_name="Purchase Alcohol",
        created_at=now,
        expires_at=now,
    )


def test_service_request_add_and_get():
    repo = ServiceRequestRepository()
    repo.add(_service_req("a"))
    assert repo.get("a").id == "a"
    assert repo.get("missing") is None


def test_service_request_replace_updates_in_place():
    repo = ServiceRequestRepository()
    repo.add(_service_req("a"))
    updated = repo.get("a").model_copy(update={"status": ServiceRequestStatus.VERIFIED})
    repo.replace(updated)
    assert repo.get("a").status is ServiceRequestStatus.VERIFIED
    assert len(repo.list()) == 1
