"""
S-PAL <-> LLM wiring.

The deterministic :class:`~pci_agent.policy.PolicyChecker` is the S-PAL trust
root — no LLM output ever adjudicates policy. The model's job here is
*composition*: turn a natural-language request from a business into a
:class:`~pci_agent.spal.RequestContext` shape the checker can then decide on.

This module is the single hook where the S-PAL flow calls into the LLM. It
wraps Ollama's JSON-schema constrained generation and validates the result
against the pydantic ``RequestContext`` model before returning.
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from pci_agent.models.ollama import OllamaBackend, OllamaSchemaError
from pci_agent.spal import RequestContext

_DEFAULT_SYSTEM_PROMPT = (
    "You are a privacy-policy assistant for the PCI (Personal Context "
    "Infrastructure) agent. Given a business's verification request, propose "
    "a JSON object describing the identity, proofs, retention and payment "
    "the user is willing to offer. Emit ONLY the JSON object — no prose, no "
    "code fences. The object MUST conform to the provided schema."
)


async def propose_request_context(
    backend: OllamaBackend,
    prompt: str,
    *,
    system_prompt: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> RequestContext:
    """Ask the LLM to synthesise a :class:`RequestContext` for policy evaluation.

    Args:
        backend: Configured :class:`OllamaBackend` (usually
            ``Agent._ollama_backend``).
        prompt: Natural-language description of the incoming business
            request — e.g. "Foo Bar wants to verify age >= 18 for a
            purchase; user is happy to disclose an age ZKP but not their DOB".
        system_prompt: Optional override for the framing prompt.
        max_tokens / temperature: Forwarded to the backend.

    Returns:
        A validated :class:`RequestContext` ready to hand to
        :meth:`pci_agent.policy.PolicyChecker.check`.

    Raises:
        OllamaSchemaError: If the model returns JSON that doesn't validate
            against ``RequestContext`` (schema shape enforced by Ollama's
            ``format`` field means the fields exist; this catches value-level
            violations).
        Any :class:`~pci_agent.models.ollama.OllamaError` subclass on
            transport / timeout / refusal failures.
    """
    schema = RequestContext.model_json_schema()
    framing = system_prompt if system_prompt is not None else _DEFAULT_SYSTEM_PROMPT
    full_prompt = (
        f"{framing}\n\n"
        f"Schema (JSON):\n{json.dumps(schema)}\n\n"
        f"Business request:\n{prompt}\n\n"
        "Respond with the JSON object only."
    )
    response = await backend.generate_structured(
        full_prompt,
        schema,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    try:
        return RequestContext.model_validate(response.data)
    except ValidationError as exc:
        raise OllamaSchemaError(f"Model produced an invalid RequestContext: {exc}") from exc
