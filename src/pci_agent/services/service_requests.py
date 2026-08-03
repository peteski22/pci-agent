"""Service-request lifecycle rules.

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
"""

from __future__ import annotations

from pci_agent.coordination import RequestStatus, ServiceRequestStatus

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
