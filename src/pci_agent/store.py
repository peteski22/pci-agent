"""In-memory repository for coordination requests.

Encapsulates storage so a persistent backend can replace it without touching
the service layer.
"""

from __future__ import annotations

from pci_agent.coordination import ServiceRequest, VerificationRequest


class RequestRepository:
    """Stores verification requests keyed by id."""

    def __init__(self) -> None:
        self._items: dict[str, VerificationRequest] = {}

    def add(self, req: VerificationRequest) -> None:
        """Store a new request."""
        self._items[req.id] = req

    def get(self, request_id: str) -> VerificationRequest | None:
        """Return the request with this id, or None if absent."""
        return self._items.get(request_id)

    def list(self) -> list[VerificationRequest]:
        """Return all stored requests."""
        return list(self._items.values())

    def replace(self, req: VerificationRequest) -> None:
        """Overwrite an existing request with the same id."""
        self._items[req.id] = req


class ServiceRequestRepository:
    """Stores service requests keyed by id."""

    def __init__(self) -> None:
        self._items: dict[str, ServiceRequest] = {}

    def add(self, req: ServiceRequest) -> None:
        """Store a new service request."""
        self._items[req.id] = req

    def get(self, request_id: str) -> ServiceRequest | None:
        """Return the service request with this id, or None if absent."""
        return self._items.get(request_id)

    def list(self) -> list[ServiceRequest]:
        """Return all stored service requests."""
        return list(self._items.values())

    def replace(self, req: ServiceRequest) -> None:
        """Overwrite an existing service request with the same id."""
        self._items[req.id] = req
