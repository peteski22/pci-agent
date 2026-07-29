"""HTTP client for the PCI ZKP service.

Replaces the urllib call in the old server with an injectable httpx client so
the approval flow can be tested against a fake transport.
"""

from __future__ import annotations

import httpx
from pydantic import BaseModel, Field

from pci_agent.errors import ZKPUnavailable


class ZKPResult(BaseModel):
    """Outcome of a proof-generation request."""

    verified: bool
    proof: dict[str, object] = Field(default_factory=dict)


class ZKPClient:
    """Generates zero-knowledge proofs via the PCI ZKP service."""

    def __init__(self, base_url: str, *, client: httpx.AsyncClient | None = None) -> None:
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(base_url=base_url, timeout=30.0)

    async def generate(self, proof_type: str, proof_data: dict[str, object]) -> ZKPResult:
        """Request a proof of the given type.

        Args:
            proof_type: The kind of proof to request (e.g. ``"age"``).
            proof_data: Claim-specific parameters sent as the request body.

        Returns:
            The verification outcome and raw proof payload.

        Raises:
            ZKPUnavailable: If the service cannot be reached, returns an error, or
                returns a response whose shape does not match the expected schema.
        """
        try:
            response = await self._client.post(f"/proofs/{proof_type}", json=proof_data)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ZKPUnavailable("verification service unavailable") from exc

        if not isinstance(payload, dict):
            raise ZKPUnavailable("verification service unavailable")

        proof = payload.get("proof", {})
        if not isinstance(proof, dict):
            raise ZKPUnavailable("verification service unavailable")

        signals = proof.get("publicSignals", {})
        verified = bool(signals.get("verified", False)) if isinstance(signals, dict) else False
        return ZKPResult(verified=verified, proof=proof)

    async def aclose(self) -> None:
        """Close the underlying client if this instance owns it."""
        if self._owns_client:
            await self._client.aclose()
