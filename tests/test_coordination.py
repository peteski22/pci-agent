from datetime import UTC, datetime

from pci_agent.coordination import (
    Action,
    ApprovalMode,
    DecisionOutcome,
    RequestStatus,
    VerificationClaim,
    VerificationRequest,
)


def test_action_vocabulary_values():
    assert Action.VERIFICATION_REQUEST == "verification.request"
    assert Action.DECISION_APPROVE == "decision.approve"


def test_approval_mode_parses_from_string():
    assert ApprovalMode("fully_autonomous") is ApprovalMode.FULLY_AUTONOMOUS


def test_verification_request_defaults_to_pending():
    req = VerificationRequest(
        id="abc123",
        business_id="biz",
        business_name="Biz",
        claim=VerificationClaim(type="age", params={"minAge": 18}),
        created_at=datetime.now(UTC),
        expires_at=datetime.now(UTC),
    )
    assert req.status is RequestStatus.PENDING
    assert req.policy_id is None


def test_decision_outcomes_are_distinct():
    assert len({o.value for o in DecisionOutcome}) == 4
