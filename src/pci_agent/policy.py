"""
S-PAL Policy enforcement
"""

from pydantic import BaseModel, ValidationError

from pci_agent.spal import (
    AccessRule,
    Conditions,
    DerivativePermission,
    IdentityType,
    RequestContext,
    SPALPolicy,
)


class PolicyCheckResult(BaseModel):
    """Result of a policy check"""

    allowed: bool
    reason: str | None = None
    policy_id: str | None = None
    matched_rule_id: str | None = None
    required_conditions: Conditions | None = None


class PolicyChecker:
    """
    Enforces S-PAL policies on agent operations

    Validates that requests comply with user-defined
    privacy and access policies.
    """

    def __init__(self) -> None:
        self._policies: dict[str, SPALPolicy] = {}

    async def load_policy(self, policy_id: str, policy_data: dict[str, object]) -> None:
        """Load and validate an S-PAL policy.

        Raises ValueError if the policy data does not conform to the S-PAL schema.
        """
        try:
            policy = SPALPolicy.model_validate(policy_data)
        except ValidationError as e:
            raise ValueError(f"Invalid S-PAL policy '{policy_id}': {e}") from e
        self._policies[policy_id] = policy

    async def check(
        self,
        policy_id: str,
        query: str,
        context_scope: str | None = None,
        request_context: RequestContext | None = None,
    ) -> PolicyCheckResult:
        """
        Check if a query is allowed by a policy.

        Args:
            policy_id: The S-PAL policy ID to check against
            query: The query being made
            context_scope: The data scope being accessed
            request_context: Request metadata for condition evaluation

        Returns:
            PolicyCheckResult indicating if allowed and why
        """
        policy = self._policies.get(policy_id)
        if policy is None:
            return PolicyCheckResult(
                allowed=True,
                reason="Policy not found, defaulting to allow",
                policy_id=policy_id,
            )

        if context_scope is None:
            return PolicyCheckResult(
                allowed=True,
                reason="No context scope specified",
                policy_id=policy_id,
            )

        matching_rules = self._find_matching_rules(policy, context_scope)
        if not matching_rules:
            return PolicyCheckResult(
                allowed=True,
                reason=f"No rules govern scope '{context_scope}'",
                policy_id=policy_id,
            )

        # Use most specific (longest scope) matching rule
        rule = max(matching_rules, key=lambda r: len(r.context_scope))

        if request_context is None:
            return PolicyCheckResult(
                allowed=False,
                reason=f"Rule '{rule.id}' requires request context for evaluation",
                policy_id=policy_id,
                matched_rule_id=rule.id,
                required_conditions=rule.conditions,
            )

        return self._evaluate_conditions(rule, policy_id, request_context)

    async def list_policies(self) -> list[str]:
        """List all loaded policy IDs"""
        return list(self._policies.keys())

    @staticmethod
    def _find_matching_rules(policy: SPALPolicy, context_scope: str) -> list[AccessRule]:
        """Find rules whose context_scope covers the requested scope."""
        matches: list[AccessRule] = []
        for rule in policy.rules:
            if (
                context_scope == rule.context_scope
                or context_scope.startswith(rule.context_scope + "/")
            ):
                matches.append(rule)
        return matches

    @staticmethod
    def _evaluate_conditions(
        rule: AccessRule,
        policy_id: str,
        ctx: RequestContext,
    ) -> PolicyCheckResult:
        """Evaluate a rule's conditions against a request context."""
        conditions = rule.conditions

        # Identity check
        if conditions.identity is not None:
            id_req = conditions.identity
            if id_req.type == IdentityType.EPHEMERAL_REQUIRED:
                if ctx.identity is None:
                    return PolicyCheckResult(
                        allowed=False,
                        reason="Ephemeral identity required but none provided",
                        policy_id=policy_id,
                        matched_rule_id=rule.id,
                    )
                if ctx.identity.type != IdentityType.EPHEMERAL_REQUIRED:
                    return PolicyCheckResult(
                        allowed=False,
                        reason="Ephemeral identity required but got "
                        f"'{ctx.identity.type}'",
                        policy_id=policy_id,
                        matched_rule_id=rule.id,
                    )

        # Proof requirements
        available_claims = {(p.type, p.claim) for p in ctx.proofs}
        for proof_req in conditions.proofs:
            if (proof_req.type, proof_req.claim) not in available_claims:
                return PolicyCheckResult(
                    allowed=False,
                    reason=f"Missing required proof: {proof_req.type} for "
                    f"'{proof_req.claim}'",
                    policy_id=policy_id,
                    matched_rule_id=rule.id,
                )

        # Derivative use checks
        if conditions.derivatives is not None and ctx.intended_use is not None:
            derivs = conditions.derivatives
            use = ctx.intended_use

            if use.training and derivs.training == DerivativePermission.FORBIDDEN:
                return PolicyCheckResult(
                    allowed=False,
                    reason="Training use is forbidden by policy",
                    policy_id=policy_id,
                    matched_rule_id=rule.id,
                )
            if use.aggregation and derivs.aggregation == DerivativePermission.FORBIDDEN:
                return PolicyCheckResult(
                    allowed=False,
                    reason="Aggregation is forbidden by policy",
                    policy_id=policy_id,
                    matched_rule_id=rule.id,
                )
            if use.resale and derivs.resale == DerivativePermission.FORBIDDEN:
                return PolicyCheckResult(
                    allowed=False,
                    reason="Resale is forbidden by policy",
                    policy_id=policy_id,
                    matched_rule_id=rule.id,
                )

        # Retention check
        if (
            conditions.retention is not None
            and ctx.offered_retention_seconds is not None
            and ctx.offered_retention_seconds > conditions.retention.max_seconds
        ):
                return PolicyCheckResult(
                    allowed=False,
                    reason=f"Offered retention ({ctx.offered_retention_seconds}s) exceeds "
                    f"maximum ({conditions.retention.max_seconds}s)",
                    policy_id=policy_id,
                    matched_rule_id=rule.id,
                )

        # Payment check
        if conditions.payment is not None and not ctx.payment_offered:
            return PolicyCheckResult(
                allowed=False,
                reason=f"Payment required: {conditions.payment.amount} "
                f"{conditions.payment.currency}",
                policy_id=policy_id,
                matched_rule_id=rule.id,
            )

        return PolicyCheckResult(
            allowed=True,
            reason="All conditions met",
            policy_id=policy_id,
            matched_rule_id=rule.id,
        )
