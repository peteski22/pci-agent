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
    VerificationClaim,
    VerificationRequest,
)
from pci_agent.policy import PolicyChecker
from pci_agent.seams import (
    ContextStoreDataProvider,
    DeterministicContextBuilder,
)
from pci_agent.services.approval import ApprovalService, resolved_status_for, status_for
from pci_agent.store import RequestRepository
from pci_agent.zkp import ZKPClient

ZKP_SERVICE_URL = os.environ.get("ZKP_SERVICE_URL", "http://localhost:8084")


class _Decider(Protocol):
    async def decide(self, request: VerificationRequest) -> ApprovalDecision: ...

    async def approve(self, request: VerificationRequest) -> ApprovalDecision: ...


class CreateVerificationRequest(BaseModel):
    """Inbound payload for POST /requests."""

    business_id: str
    business_name: str
    claim: VerificationClaim
    policy_id: str | None = None
    context_scope: str | None = None
    service_request_id: str | None = None


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
) -> FastAPI:
    """Build the coordination app.

    Args:
        config: Agent configuration; defaults to AgentConfig().
        service_factory: Builds the approval decider (injected in tests). When
            omitted, one shared ZKPClient is created for the app's lifetime and
            closed on shutdown.
        repository: Request store (injected in tests).

    Returns:
        The configured FastAPI application.
    """
    config = config or AgentConfig()
    repo = repository or RequestRepository()

    # A single ZKPClient owns one connection pool for the app's lifetime; the
    # per-request service is otherwise cheap to rebuild. Tests inject their own
    # factory and so never create the shared client.
    zkp_client: ZKPClient | None = None
    if service_factory is None:
        zkp_client = ZKPClient(ZKP_SERVICE_URL)
        make_service: Callable[[], _Decider] = _default_service_factory(config, zkp_client)
    else:
        make_service = service_factory

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        """Close the shared ZKP client's connection pool on shutdown."""
        try:
            yield
        finally:
            if zkp_client is not None:
                await zkp_client.aclose()

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
        return req

    return app


def _as_error(req: VerificationRequest) -> VerificationRequest:
    """Return a copy of the request marked as failed, for durable auditing."""
    return req.model_copy(
        update={
            "status": RequestStatus.ERROR,
            "response": {"reason": "approval evaluation failed"},
        }
    )
