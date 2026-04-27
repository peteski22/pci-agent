"""Comprehensive tests for S-PAL policy evaluation."""

import pytest

from pci_agent.policy import PolicyChecker
from pci_agent.spal import (
    AvailableProof,
    IdentityLinkage,
    IdentityRequirement,
    IdentityType,
    IntendedUse,
    ProofType,
    RequestContext,
    RequestIdentity,
    SPALPolicy,
)

# --- Helpers ---


def _policy_dict(
    rules: list[dict],
    policy_id: str = "test-policy",
    name: str = "Test Policy",
) -> dict:
    return {
        "version": "1.0",
        "id": policy_id,
        "name": name,
        "owner": "did:pci:cardano:addr1test",
        "rules": rules,
    }


def _rule_dict(
    rule_id: str = "rule-1",
    scope: str = "test/data",
    conditions: dict | None = None,
) -> dict:
    return {
        "id": rule_id,
        "context_scope": scope,
        "conditions": conditions or {},
    }


def _derivs(
    training: str = "forbidden",
    aggregation: str = "forbidden",
    resale: str = "forbidden",
) -> dict:
    return {
        "training": training,
        "aggregation": aggregation,
        "resale": resale,
    }


# --- Realistic policy fixture (based on pci-spec health-records example) ---

HEALTH_POLICY_FIXTURE: dict = {
    "version": "1.0",
    "id": "spal:did:pci:cardano:addr1abc123:health-records",
    "name": "Health Records Access Policy",
    "owner": "did:pci:cardano:addr1abc123",
    "rules": [
        {
            "id": "rule-diagnosis",
            "context_scope": "medical/diagnosis_codes",
            "conditions": {
                "identity": {"type": "ephemeral_required", "linkage": "forbidden"},
                "proofs": [{"type": "zkp", "claim": "is_licensed_provider"}],
                "retention": {"max_seconds": 0, "audit_log": True},
                "derivatives": _derivs(),
            },
        },
        {
            "id": "rule-vaccination-status",
            "context_scope": "medical/immunization/status",
            "conditions": {
                "identity": {"type": "ephemeral_required", "linkage": "forbidden"},
                "proofs": [{"type": "zkp", "claim": "has_vaccination"}],
                "retention": {"max_seconds": 3600, "audit_log": True},
                "derivatives": _derivs(aggregation="anonymized_only"),
                "payment": {"protocol": "x402", "amount": 100, "currency": "sats"},
            },
        },
        {
            "id": "rule-allergies",
            "context_scope": "medical/allergies",
            "conditions": {
                "identity": {"type": "ephemeral_required", "linkage": "forbidden"},
                "proofs": [{"type": "zkp", "claim": "has_allergy"}],
                "retention": {"max_seconds": 0, "audit_log": True},
                "derivatives": _derivs(),
            },
        },
    ],
}


# --- Model parsing tests ---


class TestSPALModelParsing:
    def test_parse_minimal_policy(self) -> None:
        policy = SPALPolicy.model_validate(_policy_dict([_rule_dict()]))
        assert policy.version == "1.0"
        assert len(policy.rules) == 1
        assert policy.rules[0].context_scope == "test/data"

    def test_reject_empty_rules(self) -> None:
        with pytest.raises(ValueError):
            SPALPolicy.model_validate(_policy_dict([]))

    def test_reject_missing_required_fields(self) -> None:
        with pytest.raises(ValueError):
            SPALPolicy.model_validate({"version": "1.0"})

    def test_linkage_string_coercion_forbidden(self) -> None:
        req = IdentityRequirement.model_validate(
            {
                "type": "ephemeral_required",
                "linkage": "forbidden",
            }
        )
        assert isinstance(req.linkage, IdentityLinkage)
        assert req.linkage.ephemeral_required is True
        assert req.linkage.proof_of_root_allowed is False

    def test_linkage_string_coercion_allowed(self) -> None:
        req = IdentityRequirement.model_validate(
            {
                "type": "any",
                "linkage": "allowed",
            }
        )
        assert isinstance(req.linkage, IdentityLinkage)
        assert req.linkage.ephemeral_required is False
        assert req.linkage.proof_of_root_allowed is True

    def test_linkage_object_passthrough(self) -> None:
        req = IdentityRequirement.model_validate(
            {
                "type": "ephemeral_required",
                "linkage": {
                    "ephemeral_required": True,
                    "proof_of_root_allowed": False,
                    "zk_continuity_allowed": True,
                },
            }
        )
        assert isinstance(req.linkage, IdentityLinkage)
        assert req.linkage.zk_continuity_allowed is True

    def test_parse_multi_rule_policy_with_all_conditions(self) -> None:
        """Parse a realistic policy with identity, proofs, retention, derivatives, payment."""
        policy = SPALPolicy.model_validate(HEALTH_POLICY_FIXTURE)
        assert policy.name == "Health Records Access Policy"
        assert len(policy.rules) == 3
        assert policy.rules[0].context_scope == "medical/diagnosis_codes"
        assert policy.rules[1].conditions.payment is not None
        assert policy.rules[1].conditions.payment.amount == 100


# --- Scope matching tests ---


class TestScopeMatching:
    @pytest.fixture
    def checker(self) -> PolicyChecker:
        return PolicyChecker()

    async def test_exact_scope_match(self, checker: PolicyChecker) -> None:
        await checker.load_policy(
            "p",
            _policy_dict(
                [
                    _rule_dict(scope="medical/diagnosis_codes"),
                ]
            ),
        )
        result = await checker.check("p", "query", context_scope="medical/diagnosis_codes")
        assert result.matched_rule_id == "rule-1"

    async def test_sub_scope_match(self, checker: PolicyChecker) -> None:
        await checker.load_policy(
            "p",
            _policy_dict(
                [
                    _rule_dict(scope="medical/diagnosis_codes"),
                ]
            ),
        )
        result = await checker.check("p", "query", context_scope="medical/diagnosis_codes/icd10")
        assert result.matched_rule_id == "rule-1"

    async def test_no_match_for_parent_scope(self, checker: PolicyChecker) -> None:
        """A broader request should not match a more specific rule."""
        await checker.load_policy(
            "p",
            _policy_dict(
                [
                    _rule_dict(scope="medical/diagnosis_codes"),
                ]
            ),
        )
        result = await checker.check("p", "query", context_scope="medical")
        assert result.allowed is True
        assert "No rules govern" in (result.reason or "")

    async def test_no_match_for_unrelated_scope(self, checker: PolicyChecker) -> None:
        await checker.load_policy(
            "p",
            _policy_dict(
                [
                    _rule_dict(scope="medical/diagnosis_codes"),
                ]
            ),
        )
        result = await checker.check("p", "query", context_scope="financial/transactions")
        assert result.allowed is True

    async def test_most_specific_rule_wins(self, checker: PolicyChecker) -> None:
        await checker.load_policy(
            "p",
            _policy_dict(
                [
                    _rule_dict(rule_id="broad", scope="medical"),
                    _rule_dict(rule_id="specific", scope="medical/diagnosis_codes"),
                ]
            ),
        )
        result = await checker.check("p", "query", context_scope="medical/diagnosis_codes")
        assert result.matched_rule_id == "specific"

    async def test_no_partial_prefix_match(self, checker: PolicyChecker) -> None:
        """'medical/diag' should not match 'medical/diagnosis_codes'."""
        await checker.load_policy(
            "p",
            _policy_dict(
                [
                    _rule_dict(scope="medical/diagnosis_codes"),
                ]
            ),
        )
        result = await checker.check("p", "query", context_scope="medical/diag")
        assert result.allowed is True
        assert "No rules govern" in (result.reason or "")


# --- Condition evaluation tests ---


class TestConditionEvaluation:
    @pytest.fixture
    def checker(self) -> PolicyChecker:
        return PolicyChecker()

    async def test_no_request_context_returns_required_conditions(
        self, checker: PolicyChecker
    ) -> None:
        await checker.load_policy(
            "p",
            _policy_dict(
                [
                    _rule_dict(
                        conditions={
                            "identity": {"type": "ephemeral_required"},
                            "proofs": [{"type": "zkp", "claim": "age_over_18"}],
                        }
                    ),
                ]
            ),
        )
        result = await checker.check("p", "query", context_scope="test/data")
        assert result.allowed is False
        assert "requires request context" in (result.reason or "")
        assert result.required_conditions is not None
        assert len(result.required_conditions.proofs) == 1

    async def test_empty_conditions_allows_without_request_context(
        self, checker: PolicyChecker
    ) -> None:
        """A rule with no effective conditions should allow without request_context."""
        await checker.load_policy("p", _policy_dict([_rule_dict(conditions={})]))
        result = await checker.check("p", "query", context_scope="test/data")
        assert result.allowed is True
        assert result.matched_rule_id == "rule-1"

    # --- Identity ---

    async def test_ephemeral_identity_required_and_provided(self, checker: PolicyChecker) -> None:
        await checker.load_policy(
            "p",
            _policy_dict(
                [
                    _rule_dict(conditions={"identity": {"type": "ephemeral_required"}}),
                ]
            ),
        )
        ctx = RequestContext(
            identity=RequestIdentity(type=IdentityType.EPHEMERAL_REQUIRED, did="did:key:z123")
        )
        result = await checker.check("p", "query", context_scope="test/data", request_context=ctx)
        assert result.allowed is True

    async def test_ephemeral_identity_required_but_persistent_given(
        self, checker: PolicyChecker
    ) -> None:
        await checker.load_policy(
            "p",
            _policy_dict(
                [
                    _rule_dict(conditions={"identity": {"type": "ephemeral_required"}}),
                ]
            ),
        )
        ctx = RequestContext(identity=RequestIdentity(type=IdentityType.PERSISTENT_ALLOWED))
        result = await checker.check("p", "query", context_scope="test/data", request_context=ctx)
        assert result.allowed is False
        assert "Ephemeral identity required" in (result.reason or "")

    async def test_ephemeral_identity_required_but_none_given(self, checker: PolicyChecker) -> None:
        await checker.load_policy(
            "p",
            _policy_dict(
                [
                    _rule_dict(conditions={"identity": {"type": "ephemeral_required"}}),
                ]
            ),
        )
        ctx = RequestContext()
        result = await checker.check("p", "query", context_scope="test/data", request_context=ctx)
        assert result.allowed is False
        assert "none provided" in (result.reason or "")

    async def test_any_identity_type_allows_all(self, checker: PolicyChecker) -> None:
        await checker.load_policy(
            "p",
            _policy_dict(
                [
                    _rule_dict(conditions={"identity": {"type": "any"}}),
                ]
            ),
        )
        ctx = RequestContext(identity=RequestIdentity(type=IdentityType.PERSISTENT_ALLOWED))
        result = await checker.check("p", "query", context_scope="test/data", request_context=ctx)
        assert result.allowed is True

    # --- Proofs ---

    async def test_required_proof_provided(self, checker: PolicyChecker) -> None:
        await checker.load_policy(
            "p",
            _policy_dict(
                [
                    _rule_dict(
                        conditions={
                            "proofs": [{"type": "zkp", "claim": "age_over_18"}],
                        }
                    ),
                ]
            ),
        )
        ctx = RequestContext(proofs=[AvailableProof(type=ProofType.ZKP, claim="age_over_18")])
        result = await checker.check("p", "query", context_scope="test/data", request_context=ctx)
        assert result.allowed is True

    async def test_required_proof_missing(self, checker: PolicyChecker) -> None:
        await checker.load_policy(
            "p",
            _policy_dict(
                [
                    _rule_dict(
                        conditions={
                            "proofs": [{"type": "zkp", "claim": "age_over_18"}],
                        }
                    ),
                ]
            ),
        )
        ctx = RequestContext(proofs=[])
        result = await checker.check("p", "query", context_scope="test/data", request_context=ctx)
        assert result.allowed is False
        assert "Missing required proof" in (result.reason or "")

    async def test_wrong_proof_type_denied(self, checker: PolicyChecker) -> None:
        await checker.load_policy(
            "p",
            _policy_dict(
                [
                    _rule_dict(
                        conditions={
                            "proofs": [{"type": "zkp", "claim": "age_over_18"}],
                        }
                    ),
                ]
            ),
        )
        ctx = RequestContext(
            proofs=[AvailableProof(type=ProofType.ATTESTATION, claim="age_over_18")]
        )
        result = await checker.check("p", "query", context_scope="test/data", request_context=ctx)
        assert result.allowed is False

    # --- Derivatives ---

    async def test_training_forbidden_and_requested(self, checker: PolicyChecker) -> None:
        await checker.load_policy(
            "p",
            _policy_dict(
                [
                    _rule_dict(
                        conditions={
                            "derivatives": _derivs(aggregation="allowed"),
                        }
                    ),
                ]
            ),
        )
        ctx = RequestContext(intended_use=IntendedUse(training=True))
        result = await checker.check("p", "query", context_scope="test/data", request_context=ctx)
        assert result.allowed is False
        assert "Training" in (result.reason or "")

    async def test_aggregation_forbidden_and_requested(self, checker: PolicyChecker) -> None:
        await checker.load_policy(
            "p",
            _policy_dict(
                [
                    _rule_dict(
                        conditions={
                            "derivatives": _derivs(
                                training="allowed",
                                aggregation="forbidden",
                                resale="allowed",
                            ),
                        }
                    ),
                ]
            ),
        )
        ctx = RequestContext(intended_use=IntendedUse(aggregation=True))
        result = await checker.check("p", "query", context_scope="test/data", request_context=ctx)
        assert result.allowed is False
        assert "Aggregation" in (result.reason or "")

    async def test_resale_forbidden_and_requested(self, checker: PolicyChecker) -> None:
        await checker.load_policy(
            "p",
            _policy_dict(
                [
                    _rule_dict(
                        conditions={
                            "derivatives": _derivs(training="allowed", aggregation="allowed"),
                        }
                    ),
                ]
            ),
        )
        ctx = RequestContext(intended_use=IntendedUse(resale=True))
        result = await checker.check("p", "query", context_scope="test/data", request_context=ctx)
        assert result.allowed is False
        assert "Resale" in (result.reason or "")

    async def test_derivatives_allowed(self, checker: PolicyChecker) -> None:
        await checker.load_policy(
            "p",
            _policy_dict(
                [
                    _rule_dict(
                        conditions={
                            "derivatives": _derivs(
                                training="allowed", aggregation="allowed", resale="allowed"
                            ),
                        }
                    ),
                ]
            ),
        )
        ctx = RequestContext(intended_use=IntendedUse(training=True, aggregation=True, resale=True))
        result = await checker.check("p", "query", context_scope="test/data", request_context=ctx)
        assert result.allowed is True

    # --- Retention ---

    async def test_retention_within_limit(self, checker: PolicyChecker) -> None:
        await checker.load_policy(
            "p",
            _policy_dict(
                [
                    _rule_dict(conditions={"retention": {"max_seconds": 3600, "audit_log": True}}),
                ]
            ),
        )
        ctx = RequestContext(offered_retention_seconds=1800)
        result = await checker.check("p", "query", context_scope="test/data", request_context=ctx)
        assert result.allowed is True

    async def test_retention_exceeds_limit(self, checker: PolicyChecker) -> None:
        await checker.load_policy(
            "p",
            _policy_dict(
                [
                    _rule_dict(conditions={"retention": {"max_seconds": 3600, "audit_log": True}}),
                ]
            ),
        )
        ctx = RequestContext(offered_retention_seconds=7200)
        result = await checker.check("p", "query", context_scope="test/data", request_context=ctx)
        assert result.allowed is False
        assert "retention" in (result.reason or "").lower()

    async def test_immediate_deletion_required(self, checker: PolicyChecker) -> None:
        await checker.load_policy(
            "p",
            _policy_dict(
                [
                    _rule_dict(conditions={"retention": {"max_seconds": 0}}),
                ]
            ),
        )
        ctx = RequestContext(offered_retention_seconds=1)
        result = await checker.check("p", "query", context_scope="test/data", request_context=ctx)
        assert result.allowed is False

    # --- Payment ---

    async def test_payment_required_and_offered(self, checker: PolicyChecker) -> None:
        await checker.load_policy(
            "p",
            _policy_dict(
                [
                    _rule_dict(
                        conditions={
                            "payment": {"protocol": "x402", "amount": 100, "currency": "sats"},
                        }
                    ),
                ]
            ),
        )
        ctx = RequestContext(payment_offered=True)
        result = await checker.check("p", "query", context_scope="test/data", request_context=ctx)
        assert result.allowed is True

    async def test_payment_required_but_not_offered(self, checker: PolicyChecker) -> None:
        await checker.load_policy(
            "p",
            _policy_dict(
                [
                    _rule_dict(
                        conditions={
                            "payment": {"protocol": "x402", "amount": 100, "currency": "sats"},
                        }
                    ),
                ]
            ),
        )
        ctx = RequestContext(payment_offered=False)
        result = await checker.check("p", "query", context_scope="test/data", request_context=ctx)
        assert result.allowed is False
        assert "Payment required" in (result.reason or "")

    # --- All conditions combined ---

    async def test_all_conditions_met(self, checker: PolicyChecker) -> None:
        await checker.load_policy(
            "p",
            _policy_dict(
                [
                    _rule_dict(
                        conditions={
                            "identity": {"type": "ephemeral_required"},
                            "proofs": [{"type": "zkp", "claim": "age_over_18"}],
                            "retention": {"max_seconds": 3600},
                            "derivatives": _derivs(aggregation="allowed"),
                            "payment": {"protocol": "x402", "amount": 100, "currency": "sats"},
                        }
                    ),
                ]
            ),
        )
        ctx = RequestContext(
            identity=RequestIdentity(type=IdentityType.EPHEMERAL_REQUIRED, did="did:key:z123"),
            proofs=[AvailableProof(type=ProofType.ZKP, claim="age_over_18")],
            intended_use=IntendedUse(aggregation=True),
            offered_retention_seconds=1800,
            payment_offered=True,
        )
        result = await checker.check("p", "query", context_scope="test/data", request_context=ctx)
        assert result.allowed is True
        assert result.matched_rule_id == "rule-1"


# --- Integration with realistic policy ---


class TestRealisticPolicyIntegration:
    @pytest.fixture
    def checker(self) -> PolicyChecker:
        return PolicyChecker()

    async def test_health_policy_valid_request(self, checker: PolicyChecker) -> None:
        await checker.load_policy("health", HEALTH_POLICY_FIXTURE)
        ctx = RequestContext(
            identity=RequestIdentity(type=IdentityType.EPHEMERAL_REQUIRED, did="did:key:z123"),
            proofs=[AvailableProof(type=ProofType.ZKP, claim="is_licensed_provider")],
            offered_retention_seconds=0,
        )
        result = await checker.check(
            "health", "query", context_scope="medical/diagnosis_codes", request_context=ctx
        )
        assert result.allowed is True

    async def test_health_policy_wrong_identity(self, checker: PolicyChecker) -> None:
        await checker.load_policy("health", HEALTH_POLICY_FIXTURE)
        ctx = RequestContext(
            identity=RequestIdentity(type=IdentityType.PERSISTENT_ALLOWED),
            proofs=[AvailableProof(type=ProofType.ZKP, claim="is_licensed_provider")],
        )
        result = await checker.check(
            "health", "query", context_scope="medical/diagnosis_codes", request_context=ctx
        )
        assert result.allowed is False
        assert "Ephemeral" in (result.reason or "")

    async def test_health_policy_missing_proof(self, checker: PolicyChecker) -> None:
        await checker.load_policy("health", HEALTH_POLICY_FIXTURE)
        ctx = RequestContext(
            identity=RequestIdentity(type=IdentityType.EPHEMERAL_REQUIRED, did="did:key:z123"),
            proofs=[],
        )
        result = await checker.check(
            "health", "query", context_scope="medical/diagnosis_codes", request_context=ctx
        )
        assert result.allowed is False
        assert "Missing required proof" in (result.reason or "")

    async def test_health_policy_vaccination_needs_payment(self, checker: PolicyChecker) -> None:
        await checker.load_policy("health", HEALTH_POLICY_FIXTURE)
        ctx = RequestContext(
            identity=RequestIdentity(type=IdentityType.EPHEMERAL_REQUIRED, did="did:key:z123"),
            proofs=[AvailableProof(type=ProofType.ZKP, claim="has_vaccination")],
            offered_retention_seconds=3600,
            payment_offered=False,
        )
        result = await checker.check(
            "health",
            "query",
            context_scope="medical/immunization/status",
            request_context=ctx,
        )
        assert result.allowed is False
        assert "Payment required" in (result.reason or "")


# --- Backward compatibility ---


class TestBackwardCompatibility:
    @pytest.fixture
    def checker(self) -> PolicyChecker:
        return PolicyChecker()

    async def test_missing_policy_still_allows(self, checker: PolicyChecker) -> None:
        result = await checker.check("nonexistent", "query")
        assert result.allowed is True

    async def test_no_context_scope_still_allows(self, checker: PolicyChecker) -> None:
        await checker.load_policy("p", _policy_dict([_rule_dict()]))
        result = await checker.check("p", "query")
        assert result.allowed is True

    async def test_invalid_policy_raises_valueerror(self, checker: PolicyChecker) -> None:
        with pytest.raises(ValueError, match="Invalid S-PAL policy"):
            await checker.load_policy("bad", {})

    async def test_empty_rules_raises_valueerror(self, checker: PolicyChecker) -> None:
        with pytest.raises(ValueError, match="Invalid S-PAL policy"):
            await checker.load_policy("bad", _policy_dict([]))
