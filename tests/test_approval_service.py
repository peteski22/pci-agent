from datetime import UTC, datetime, timedelta

import pytest

from pci_agent.coordination import (
    ApprovalMode,
    DecisionOutcome,
    RequestStatus,
    VerificationClaim,
    VerificationRequest,
)
from pci_agent.policy import PolicyCheckResult
from pci_agent.services.approval import ApprovalService, status_for
from pci_agent.spal import RequestContext


class FakePolicy:
    def __init__(self, result: PolicyCheckResult) -> None:
        self._result = result

    async def check(self, policy_id, query, context_scope=None, request_context=None):
        return self._result


class FakeBuilder:
    def build(self, request):
        return RequestContext()


class FakeData:
    def __init__(self, data=None, fail=False):
        self._data = data or {"birthDate": "2000-01-01"}
        self._fail = fail

    async def fetch(self, scope):
        if self._fail:
            from pci_agent.errors import ContextUnavailable

            raise ContextUnavailable("no data")
        return self._data


class FakeZKP:
    def __init__(self, verified=True, fail=False):
        self._verified = verified
        self._fail = fail

    async def generate(self, proof_type, proof_data):
        if self._fail:
            from pci_agent.errors import ZKPUnavailable

            raise ZKPUnavailable("down")
        from pci_agent.zkp import ZKPResult

        return ZKPResult(verified=self._verified, proof={"ok": True})


def _req(minutes_valid: int = 5) -> VerificationRequest:
    now = datetime.now(UTC)
    return VerificationRequest(
        id="a",
        business_id="biz",
        business_name="Biz",
        claim=VerificationClaim(type="age", params={"minAge": 18}),
        policy_id="p1",
        context_scope="health/age",
        created_at=now,
        expires_at=now + timedelta(minutes=minutes_valid),
    )


def _svc(policy, data=None, zkp=None) -> ApprovalService:
    return ApprovalService(policy, FakeBuilder(), data or FakeData(), zkp or FakeZKP())


async def test_conclusive_allow_and_verified_proof_approves():
    policy = FakePolicy(PolicyCheckResult(allowed=True, matched_rule_id="r1"))
    decision = await _svc(policy).decide(_req())
    assert decision.outcome is DecisionOutcome.APPROVE
    assert decision.proof == {"ok": True}


class BoomData:
    """Fetch fake that fails loudly if reached: a deny must fetch no private data."""

    async def fetch(self, scope):
        raise AssertionError("must not fetch private data on a deny")


class BoomZKP:
    """Generate fake that fails loudly if reached: a deny must generate no proof."""

    async def generate(self, proof_type, proof_data):
        raise AssertionError("must not generate a proof on a deny")


async def test_default_allow_without_matched_rule_does_not_approve():
    # Fail-closed: allowed=True but no rule matched must NOT auto-approve.
    policy = FakePolicy(PolicyCheckResult(allowed=True, matched_rule_id=None))
    decision = await _svc(policy, data=BoomData(), zkp=BoomZKP()).decide(_req())
    assert decision.outcome is DecisionOutcome.DENY


async def test_policy_denied_denies():
    policy = FakePolicy(PolicyCheckResult(allowed=False, matched_rule_id="r1", reason="no"))
    decision = await _svc(policy, data=BoomData(), zkp=BoomZKP()).decide(_req())
    assert decision.outcome is DecisionOutcome.DENY


async def test_allowed_but_proof_fails_rejects():
    policy = FakePolicy(PolicyCheckResult(allowed=True, matched_rule_id="r1"))
    decision = await _svc(policy, zkp=FakeZKP(verified=False)).decide(_req())
    assert decision.outcome is DecisionOutcome.REJECT


async def test_context_unavailable_errors():
    policy = FakePolicy(PolicyCheckResult(allowed=True, matched_rule_id="r1"))
    decision = await _svc(policy, data=FakeData(fail=True)).decide(_req())
    assert decision.outcome is DecisionOutcome.ERROR


async def test_zkp_unavailable_errors():
    policy = FakePolicy(PolicyCheckResult(allowed=True, matched_rule_id="r1"))
    decision = await _svc(policy, zkp=FakeZKP(fail=True)).decide(_req())
    assert decision.outcome is DecisionOutcome.ERROR


async def test_expired_request_denies_without_calling_policy():
    class Boom:
        async def check(self, *a, **k):
            raise AssertionError("policy must not be consulted for expired request")

    decision = await _svc(Boom()).decide(_req(minutes_valid=-1))
    assert decision.outcome is DecisionOutcome.DENY


async def test_missing_policy_id_denies_without_calling_policy():
    class Boom:
        async def check(self, *a, **k):
            raise AssertionError("policy must not be consulted with no policy_id")

    request = _req()
    request.policy_id = None
    decision = await _svc(Boom()).decide(request)
    assert decision.outcome is DecisionOutcome.DENY


@pytest.mark.parametrize(
    "outcome,mode,expected",
    [
        (DecisionOutcome.APPROVE, ApprovalMode.FULLY_AUTONOMOUS, RequestStatus.APPROVED),
        (DecisionOutcome.REJECT, ApprovalMode.FULLY_AUTONOMOUS, RequestStatus.REJECTED),
        (DecisionOutcome.ERROR, ApprovalMode.FULLY_AUTONOMOUS, RequestStatus.ERROR),
        (DecisionOutcome.DENY, ApprovalMode.FULLY_AUTONOMOUS, RequestStatus.DENIED),
        (DecisionOutcome.DENY, ApprovalMode.AUTO_WITH_NOTIFICATION, RequestStatus.ESCALATED),
    ],
)
def test_status_for(outcome, mode, expected):
    assert status_for(outcome, mode) is expected
