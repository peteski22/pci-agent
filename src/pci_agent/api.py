"""FastAPI coordination service.

Thin controllers only: build the request model, run the mode-appropriate
path, and serialize. All decision logic lives in ApprovalService.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Protocol

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pci_agent.config import AgentConfig
from pci_agent.context import ContextClient
from pci_agent.coordination import (
    ApprovalDecision,
    ApprovalMode,
    RequestStatus,
    ServiceRequest,
    ServiceRequestStatus,
    VerificationClaim,
    VerificationRequest,
)
from pci_agent.policy import PolicyChecker
from pci_agent.seams import (
    ContextStoreDataProvider,
    DeterministicContextBuilder,
)
from pci_agent.services.approval import ApprovalService, resolved_status_for, status_for
from pci_agent.services.service_requests import LINKABLE_STATUSES, service_status_for
from pci_agent.status import ServicesStatus, StatusClient
from pci_agent.store import RequestRepository, ServiceRequestRepository
from pci_agent.zkp import ZKPClient

ZKP_SERVICE_URL = os.environ.get("ZKP_SERVICE_URL", "http://localhost:8084")
CARDANO_API_URL = os.environ.get("CARDANO_API_URL", "http://localhost:8080")


class _Decider(Protocol):
    async def decide(self, request: VerificationRequest) -> ApprovalDecision: ...

    async def approve(self, request: VerificationRequest) -> ApprovalDecision: ...


class _StatusSource(Protocol):
    async def check(self) -> ServicesStatus: ...


class CreateVerificationRequest(BaseModel):
    """Inbound payload for POST /requests."""

    business_id: str
    business_name: str
    claim: VerificationClaim
    policy_id: str | None = None
    context_scope: str | None = None
    service_request_id: str | None = None


class CreateServiceRequest(BaseModel):
    """Inbound payload for POST /service-requests."""

    user_id: str
    user_name: str
    business_id: str
    service_type: str
    service_name: str


def _default_service_factory(config: AgentConfig, zkp_client: ZKPClient) -> Callable[[], _Decider]:
    def factory() -> _Decider:
        return ApprovalService(
            PolicyChecker(),
            DeterministicContextBuilder(),
            ContextStoreDataProvider(ContextClient(config.context)),
            zkp_client,
        )

    return factory


def create_app(
    config: AgentConfig | None = None,
    *,
    service_factory: Callable[[], _Decider] | None = None,
    repository: RequestRepository | None = None,
    service_request_repository: ServiceRequestRepository | None = None,
    status_source: _StatusSource | None = None,
) -> FastAPI:
    """Build the coordination app.

    Args:
        config: Agent configuration; defaults to AgentConfig().
        service_factory: Builds the approval decider (injected in tests). When
            omitted, one shared ZKPClient is created for the app's lifetime and
            closed on shutdown.
        repository: Request store (injected in tests).
        service_request_repository: Service-request store (injected in tests).
        status_source: Provides the /services aggregate (injected in tests).
            When omitted, one shared StatusClient is created for the app's
            lifetime and closed on shutdown.

    Returns:
        The configured FastAPI application.
    """
    config = config or AgentConfig()
    repo = repository or RequestRepository()
    service_repo = service_request_repository or ServiceRequestRepository()

    # A single ZKPClient owns one connection pool for the app's lifetime; the
    # per-request service is otherwise cheap to rebuild. Tests inject their own
    # factory and so never create the shared client.
    zkp_client: ZKPClient | None = None
    if service_factory is None:
        zkp_client = ZKPClient(ZKP_SERVICE_URL)
        make_service: Callable[[], _Decider] = _default_service_factory(config, zkp_client)
    else:
        make_service = service_factory

    status_client: StatusClient | None = None
    if status_source is None:
        agent_url = f"http://localhost:{os.environ.get('PORT', '8082')}"
        status_client = StatusClient(
            agent_url=agent_url, zkp_url=ZKP_SERVICE_URL, cardano_url=CARDANO_API_URL
        )
        status_checker: _StatusSource = status_client
    else:
        status_checker = status_source

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        """Close the shared HTTP clients' connection pools on shutdown."""
        try:
            yield
        finally:
            if zkp_client is not None:
                await zkp_client.aclose()
            if status_client is not None:
                await status_client.aclose()

    app = FastAPI(title="pci-agent", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Report liveness for container health checks."""
        return {"status": "healthy", "service": "pci-agent"}

    @app.get("/requests")
    async def list_requests() -> dict[str, list[VerificationRequest]]:
        """Return every tracked verification request."""
        return {"requests": repo.list()}

    @app.get("/requests/{request_id}")
    async def get_request(request_id: str) -> VerificationRequest:
        """Return a single request, or 404 if it is not tracked."""
        req = repo.get(request_id)
        if req is None:
            raise HTTPException(status_code=404, detail="request not found")
        return req

    def _link_service_request(req: VerificationRequest) -> None:
        """Attach a new verification request to its service request, if any.

        Raises:
            HTTPException: 404 if the referenced service request is not
                tracked, 409 if it has already been resolved.
        """
        if req.service_request_id is None:
            return
        service_req = service_repo.get(req.service_request_id)
        if service_req is None:
            raise HTTPException(status_code=404, detail="service request not found")
        if service_req.status not in LINKABLE_STATUSES:
            raise HTTPException(status_code=409, detail="service request already resolved")
        service_repo.replace(
            service_req.model_copy(
                update={
                    "status": ServiceRequestStatus.VERIFICATION_REQUIRED,
                    "verification_request_id": req.id,
                }
            )
        )

    def _sync_service_request(req: VerificationRequest) -> None:
        """Propagate a verification outcome to the linked service request.

        Only the verification request the service request currently links to
        may drive it; a stale, re-requested verification is ignored.
        """
        if req.service_request_id is None:
            return
        service_req = service_repo.get(req.service_request_id)
        if service_req is None or service_req.verification_request_id != req.id:
            return
        new_status = service_status_for(req.status)
        if new_status is None or service_req.status is new_status:
            return
        service_repo.replace(service_req.model_copy(update={"status": new_status}))

    @app.post("/requests", status_code=201)
    async def create_request(payload: CreateVerificationRequest) -> VerificationRequest:
        """Create a request; autonomously resolve it unless in manual mode."""
        now = datetime.now(UTC)
        req = VerificationRequest(
            id=uuid.uuid4().hex,
            business_id=payload.business_id,
            business_name=payload.business_name,
            claim=payload.claim,
            policy_id=payload.policy_id,
            context_scope=payload.context_scope,
            service_request_id=payload.service_request_id,
            created_at=now,
            expires_at=now + timedelta(minutes=5),
        )
        _link_service_request(req)
        if config.approval.mode is not ApprovalMode.MANUAL:
            try:
                decision = await make_service().decide(req)
            except Exception as exc:
                repo.add(_as_error(req))
                raise HTTPException(status_code=500, detail="approval evaluation failed") from exc
            req = req.model_copy(
                update={
                    "status": status_for(decision.outcome, config.approval.mode),
                    "response": {"reason": decision.reason, "proof": decision.proof},
                }
            )
        repo.add(req)
        _sync_service_request(req)
        return req

    @app.post("/requests/{request_id}/approve")
    async def approve_request(request_id: str) -> VerificationRequest:
        """Resolve a pending or escalated request by human approval."""
        req = repo.get(request_id)
        if req is None:
            raise HTTPException(status_code=404, detail="request not found")
        if req.status not in (RequestStatus.PENDING, RequestStatus.ESCALATED):
            raise HTTPException(status_code=409, detail="request already resolved")

        try:
            decision = await make_service().approve(req)
        except Exception as exc:
            repo.replace(_as_error(req))
            raise HTTPException(status_code=500, detail="approval evaluation failed") from exc
        req = req.model_copy(
            update={
                "status": resolved_status_for(decision.outcome),
                "response": {"reason": decision.reason, "proof": decision.proof},
            }
        )
        repo.replace(req)
        _sync_service_request(req)
        return req

    @app.post("/requests/{request_id}/deny")
    async def deny_request(request_id: str) -> VerificationRequest:
        """Resolve a pending or escalated request as denied by the user."""
        req = repo.get(request_id)
        if req is None:
            raise HTTPException(status_code=404, detail="request not found")
        if req.status not in (RequestStatus.PENDING, RequestStatus.ESCALATED):
            raise HTTPException(status_code=409, detail="request already resolved")

        req = req.model_copy(
            update={
                "status": RequestStatus.DENIED,
                "response": {"reason": "denied by user"},
            }
        )
        repo.replace(req)
        _sync_service_request(req)
        return req

    @app.get("/services")
    async def services_status() -> ServicesStatus:
        """Report reachability of the agent and its coordinated services."""
        return await status_checker.check()

    @app.get("/service-requests")
    async def list_service_requests(
        status: ServiceRequestStatus | None = None,
    ) -> dict[str, list[ServiceRequest]]:
        """Return tracked service requests, optionally filtered by status."""
        requests = service_repo.list()
        if status is not None:
            requests = [r for r in requests if r.status is status]
        return {"requests": requests}

    @app.get("/service-requests/{request_id}")
    async def get_service_request(request_id: str) -> ServiceRequest:
        """Return a single service request, or 404 if it is not tracked."""
        service_req = service_repo.get(request_id)
        if service_req is None:
            raise HTTPException(status_code=404, detail="service request not found")
        return service_req

    @app.post("/service-requests", status_code=201)
    async def create_service_request(payload: CreateServiceRequest) -> ServiceRequest:
        """Create a pending service request for the business to pick up."""
        now = datetime.now(UTC)
        service_req = ServiceRequest(
            id=uuid.uuid4().hex,
            user_id=payload.user_id,
            user_name=payload.user_name,
            business_id=payload.business_id,
            service_type=payload.service_type,
            service_name=payload.service_name,
            created_at=now,
            expires_at=now + timedelta(minutes=10),
        )
        service_repo.add(service_req)
        return service_req

    @app.post("/service-requests/{request_id}/complete")
    async def complete_service_request(request_id: str) -> ServiceRequest:
        """Complete a verified service request.

        Completing an already-completed request is idempotent and returns the
        request unchanged with 200; any other unverified state is a 409.
        """
        service_req = service_repo.get(request_id)
        if service_req is None:
            raise HTTPException(status_code=404, detail="service request not found")
        if service_req.status is ServiceRequestStatus.COMPLETED:
            return service_req
        if service_req.status is not ServiceRequestStatus.VERIFIED:
            raise HTTPException(status_code=409, detail="service request not verified")
        service_req = service_req.model_copy(
            update={
                "status": ServiceRequestStatus.COMPLETED,
                "completed_at": datetime.now(UTC),
            }
        )
        service_repo.replace(service_req)
        return service_req

    return app


def _as_error(req: VerificationRequest) -> VerificationRequest:
    """Return a copy of the request marked as failed, for durable auditing."""
    return req.model_copy(
        update={
            "status": RequestStatus.ERROR,
            "response": {"reason": "approval evaluation failed"},
        }
    )
