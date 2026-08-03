"""Coordination-layer error hierarchy.

All service-raised errors inherit :class:`ServiceError` so the API layer can
map them to HTTP status codes in one place.
"""

from __future__ import annotations


class ServiceError(Exception):
    """Base for pci-agent coordination errors."""


class ContextUnavailable(ServiceError):  # noqa: N818
    """The user's private data could not be retrieved for a scope."""


class ZKPUnavailable(ServiceError):  # noqa: N818
    """The ZKP service could not be reached or returned an error."""


class RequestExpired(ServiceError):  # noqa: N818
    """An action was attempted on a request past its expiry."""


class ServiceRequestNotFound(ServiceError):  # noqa: N818
    """A referenced service request is not tracked."""


class ServiceRequestConflict(ServiceError):  # noqa: N818
    """The service request's status does not allow the attempted action."""
