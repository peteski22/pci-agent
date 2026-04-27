"""
S-PAL (Sovereign Privacy & Access Language) policy models.

Pydantic models mirroring the S-PAL v1.0 JSON schema, plus request-side
models used for policy evaluation.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

# --- Enums ---


class IdentityType(StrEnum):
    EPHEMERAL_REQUIRED = "ephemeral_required"
    PERSISTENT_ALLOWED = "persistent_allowed"
    ANY = "any"


class ProofType(StrEnum):
    ZKP = "zkp"
    ATTESTATION = "attestation"
    SIGNATURE = "signature"


class DerivativePermission(StrEnum):
    FORBIDDEN = "forbidden"
    ALLOWED = "allowed"
    REQUIRES_PAYMENT = "requires_payment"
    REQUIRES_CONSENT = "requires_consent"
    ANONYMIZED_ONLY = "anonymized_only"


class PaymentProtocol(StrEnum):
    X402 = "x402"
    LIGHTNING = "lightning"
    CARDANO = "cardano"


class PaymentCurrency(StrEnum):
    SATS = "sats"
    LOVELACE = "lovelace"
    USD_CENTS = "usd_cents"


# --- Policy-side models ---


class IdentityLinkage(BaseModel):
    ephemeral_required: bool
    proof_of_root_allowed: bool = True
    zk_continuity_allowed: bool = False


class IdentityRequirement(BaseModel):
    type: IdentityType = IdentityType.ANY
    linkage: IdentityLinkage | None = None

    @field_validator("linkage", mode="before")
    @classmethod
    def coerce_linkage_string(cls, v: Any) -> Any:
        """Handle example policies that use 'forbidden' string instead of object."""
        if v == "forbidden":
            return IdentityLinkage(
                ephemeral_required=True,
                proof_of_root_allowed=False,
                zk_continuity_allowed=False,
            )
        if v == "allowed":
            return IdentityLinkage(
                ephemeral_required=False,
                proof_of_root_allowed=True,
                zk_continuity_allowed=True,
            )
        return v


class ProofRequirement(BaseModel):
    type: ProofType
    claim: str
    params: dict[str, Any] | None = None


class RetentionPolicy(BaseModel):
    max_seconds: int = Field(ge=0)
    audit_log: bool = False


class DerivativePolicy(BaseModel):
    training: DerivativePermission = DerivativePermission.FORBIDDEN
    aggregation: DerivativePermission = DerivativePermission.FORBIDDEN
    resale: DerivativePermission = DerivativePermission.FORBIDDEN


class PaymentRequirement(BaseModel):
    protocol: PaymentProtocol
    amount: int = Field(ge=0)
    currency: PaymentCurrency


class Conditions(BaseModel):
    identity: IdentityRequirement | None = None
    proofs: list[ProofRequirement] = Field(default_factory=list)
    retention: RetentionPolicy | None = None
    derivatives: DerivativePolicy | None = None
    payment: PaymentRequirement | None = None


class AccessRule(BaseModel):
    id: str
    context_scope: str
    conditions: Conditions


class Enforcement(BaseModel):
    smart_contract: str | None = None
    validators: list[str] = Field(default_factory=list)
    dispute_resolution: str | None = None


class SPALPolicy(BaseModel):
    version: str
    id: str
    name: str
    description: str | None = None
    created: str | None = None
    updated: str | None = None
    owner: str
    rules: list[AccessRule] = Field(min_length=1)
    enforcement: Enforcement | None = None
    signature: str | None = None


# --- Request-side models (for evaluation) ---


class RequestIdentity(BaseModel):
    """Identity info provided by the requester."""

    type: IdentityType
    did: str | None = None


class AvailableProof(BaseModel):
    """A proof the requester can present."""

    type: ProofType
    claim: str


class IntendedUse(BaseModel):
    """What the requester intends to do with the data."""

    training: bool = False
    aggregation: bool = False
    resale: bool = False


class RequestContext(BaseModel):
    """Context about the incoming request, used for policy evaluation."""

    identity: RequestIdentity | None = None
    proofs: list[AvailableProof] = Field(default_factory=list)
    intended_use: IntendedUse | None = None
    offered_retention_seconds: int | None = None
    payment_offered: bool = False
