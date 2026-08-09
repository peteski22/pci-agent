"""Service-request lifecycle orchestration.

A service request's status is driven by its linked verification request:
creating one moves the service request to verification_required, and the
verification outcome maps onto the service request via service_status_for.

Deliberate deviations from the legacy agent, which updated the linked
service request unconditionally:

- An ERROR verification outcome leaves the service request unchanged (the
  legacy server recorded it as rejected, conflating a service outage with
  the user failing the claim's criteria). The business can re-request
  verification instead.
- Only a service request that is pending or awaiting verification accepts a
  new verification request; resolved ones no longer regress to
  verification_required.
- Only the verification request the service request currently links to may
  drive its status; a stale, re-requested verification is ignored.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from pci_agent.coordination import (
    CreateServiceRequest,
    RequestStatus,
    ServiceRequest,
    ServiceRequestStatus,
    VerificationRequest,
)
from pci_agent.errors import ServiceRequestConflict, ServiceRequestNotFound
from pci_agent.store import ServiceRequestRepository

SERVICE_REQUEST_TTL = timedelta(minutes=10)

LINKABLE_STATUSES = (
    ServiceRequestStatus.PENDING,
    ServiceRequestStatus.VERIFICATION_REQUIRED,
)

_RESOLUTION_MAP = {
    RequestStatus.APPROVED: ServiceRequestStatus.VERIFIED,
    RequestStatus.REJECTED: ServiceRequestStatus.REJECTED,
    RequestStatus.DENIED: ServiceRequestStatus.DENIED,
}


def service_status_for(resolution: RequestStatus) -> ServiceRequestStatus | None:
    """Map a verification outcome onto the linked service request's status.

    Args:
        resolution: The verification request's status after evaluation.

    Returns:
        The status the linked service request should move to, or None when
        the outcome leaves it unchanged (pending, escalated, or error).
    """
    return _RESOLUTION_MAP.get(resolution)


def create(repo: ServiceRequestRepository, payload: CreateServiceRequest) -> ServiceRequest:
    """Create and store a pending service request.

    Args:
        repo: The service-request store.
        payload: The requesting user's and target business's details.

    Returns:
        The stored request, pending with an advisory expiry.
    """
    now = datetime.now(UTC)
    request = ServiceRequest(
        id=uuid.uuid4().hex,
        user_id=payload.user_id,
        user_name=payload.user_name,
        business_id=payload.business_id,
        service_type=payload.service_type,
        service_name=payload.service_name,
        created_at=now,
        expires_at=now + SERVICE_REQUEST_TTL,
    )
    repo.add(request)
    return request


def link_verification(repo: ServiceRequestRepository, verification: VerificationRequest) -> None:
    """Attach a new verification request to the service request it references.

    Args:
        repo: The service-request store.
        verification: The verification request being created; a None
            service_request_id makes this a no-op.

    Raises:
        ServiceRequestNotFound: If the referenced service request is not
            tracked.
        ServiceRequestConflict: If the referenced service request has
            already been resolved.
    """
    if verification.service_request_id is None:
        return
    request = repo.get(verification.service_request_id)
    if request is None:
        raise ServiceRequestNotFound("service request not found")
    if request.status not in LINKABLE_STATUSES:
        raise ServiceRequestConflict("service request already resolved")
    repo.replace(
        request.model_copy(
            update={
                "status": ServiceRequestStatus.VERIFICATION_REQUIRED,
                "verification_request_id": verification.id,
            }
        )
    )


def apply_verification_outcome(
    repo: ServiceRequestRepository, verification: VerificationRequest
) -> None:
    """Propagate a verification outcome to the linked service request.

    Args:
        repo: The service-request store.
        verification: The verification request after resolution.
    """
    if verification.service_request_id is None:
        return
    request = repo.get(verification.service_request_id)
    if request is None or request.verification_request_id != verification.id:
        return
    new_status = service_status_for(verification.status)
    if new_status is None or request.status is new_status:
        return
    repo.replace(request.model_copy(update={"status": new_status}))


def complete(repo: ServiceRequestRepository, request_id: str) -> ServiceRequest:
    """Complete a verified service request.

    Completing an already-completed request is idempotent and returns it
    unchanged, because the business app polls this operation.

    Args:
        repo: The service-request store.
        request_id: The service request to complete.

    Returns:
        The completed request.

    Raises:
        ServiceRequestNotFound: If the service request is not tracked.
        ServiceRequestConflict: If the service request is not verified.
    """
    request = repo.get(request_id)
    if request is None:
        raise ServiceRequestNotFound("service request not found")
    if request.status is ServiceRequestStatus.COMPLETED:
        return request
    if request.status is not ServiceRequestStatus.VERIFIED:
        raise ServiceRequestConflict("service request not verified")
    request = request.model_copy(
        update={
            "status": ServiceRequestStatus.COMPLETED,
            "completed_at": datetime.now(UTC),
        }
    )
    repo.replace(request)
    return request
