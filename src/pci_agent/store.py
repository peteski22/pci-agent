"""In-memory repository for coordination requests.

Encapsulates storage so a persistent backend can replace it without touching
the service layer.
"""

from __future__ import annotations

from pci_agent.coordination import VerificationRequest


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
