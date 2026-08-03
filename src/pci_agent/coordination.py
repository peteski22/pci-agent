"""Domain models for the agent coordination flow.

These replace the raw dicts the HTTP layer previously used. Leaf module:
imports only stdlib and pydantic.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class Action(StrEnum):
    """The a2a message vocabulary; enumerated by the DID envelope's action field."""

    SERVICE_REQUEST = "service.request"
    VERIFICATION_REQUEST = "verification.request"
    NEGOTIATE = "negotiate"
    PROOF_SUBMIT = "proof.submit"
    PAYMENT_OFFER = "payment.offer"
    DECISION_APPROVE = "decision.approve"
    DECISION_DENY = "decision.deny"
    SERVICE_COMPLETE = "service.complete"


class ApprovalMode(StrEnum):
    """How the agent resolves an incoming verification request."""

    MANUAL = "manual"
    AUTO_WITH_NOTIFICATION = "auto_with_notification"
    FULLY_AUTONOMOUS = "fully_autonomous"


class RequestStatus(StrEnum):
    """Lifecycle state of a verification request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    DENIED = "denied"
    ESCALATED = "escalated"
    ERROR = "error"


class DecisionOutcome(StrEnum):
    """Result of a pure approval evaluation, before mode is applied."""

    APPROVE = "approve"
    REJECT = "reject"
    DENY = "deny"
    ERROR = "error"


class ServiceRequestStatus(StrEnum):
    """Lifecycle state of a user-to-business service request."""

    PENDING = "pending"
    VERIFICATION_REQUIRED = "verification_required"
    VERIFIED = "verified"
    COMPLETED = "completed"
    DENIED = "denied"
    REJECTED = "rejected"


class ServiceRequest(BaseModel):
    """A user-to-business service request tracked by the coordinator."""

    id: str
    user_id: str
    user_name: str
    business_id: str
    service_type: str
    service_name: str
    status: ServiceRequestStatus = ServiceRequestStatus.PENDING
    created_at: datetime
    expires_at: datetime
    verification_request_id: str | None = None
    completed_at: datetime | None = None


class VerificationClaim(BaseModel):
    """The claim a business asks the user to prove (e.g. age >= 18)."""

    type: str
    params: dict[str, object] = Field(default_factory=dict)


class VerificationRequest(BaseModel):
    """A business-to-user verification request tracked by the coordinator."""

    id: str
    business_id: str
    business_name: str
    claim: VerificationClaim
    policy_id: str | None = None
    context_scope: str | None = None
    service_request_id: str | None = None
    status: RequestStatus = RequestStatus.PENDING
    created_at: datetime
    expires_at: datetime
    response: dict[str, object] | None = None


class ApprovalDecision(BaseModel):
    """The outcome of evaluating a verification request against policy."""

    outcome: DecisionOutcome
    reason: str
    matched_rule_id: str | None = None
    proof: dict[str, object] | None = None
