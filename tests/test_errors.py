from pci_agent.errors import (
    ContextUnavailable,
    RequestExpired,
    ServiceError,
    ZKPUnavailable,
)


def test_service_errors_share_a_base():
    for exc in (ContextUnavailable, ZKPUnavailable, RequestExpired):
        assert issubclass(exc, ServiceError)


def test_service_error_is_an_exception():
    assert issubclass(ServiceError, Exception)
