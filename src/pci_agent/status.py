"""Reachability checks for the services the agent coordinates.

Backs the /services aggregate: the agent itself plus the ZKP service and
the Cardano devnet API, probed concurrently with a short timeout.
"""

from __future__ import annotations

import asyncio

import httpx2
from pydantic import BaseModel


class EndpointStatus(BaseModel):
    """Reachability of one coordinated service."""

    status: str
    url: str
    error: str | None = None


class CardanoStatus(EndpointStatus):
    """Cardano devnet reachability, with the latest block height when available."""

    latest_block: int | None = None


class ServicesStatus(BaseModel):
    """Aggregate status of the agent and the services it coordinates."""

    agent: EndpointStatus
    zkp: EndpointStatus
    cardano: CardanoStatus


class StatusClient:
    """Probes the health of the services the agent depends on."""

    def __init__(
        self,
        *,
        agent_url: str,
        zkp_url: str,
        cardano_url: str,
        client: httpx2.AsyncClient | None = None,
    ) -> None:
        """Configure the probe targets.

        Args:
            agent_url: The agent's own base URL, reported as-is.
            zkp_url: Base URL of the ZKP service; its /health route is probed.
            cardano_url: Base URL of the Cardano devnet API.
            client: HTTP client override (injected in tests). When omitted, an
                owned client with a 5-second timeout is created and closed by
                aclose.
        """
        self._agent_url = agent_url
        self._zkp_url = zkp_url
        self._cardano_url = cardano_url
        self._owns_client = client is None
        self._client = client or httpx2.AsyncClient(timeout=5.0)

    async def check(self) -> ServicesStatus:
        """Probe the coordinated services concurrently and aggregate the results."""
        zkp, cardano = await asyncio.gather(self._check_zkp(), self._check_cardano())
        return ServicesStatus(
            agent=EndpointStatus(status="healthy", url=self._agent_url),
            zkp=zkp,
            cardano=cardano,
        )

    async def _check_zkp(self) -> EndpointStatus:
        """Probe the ZKP service's health route."""
        try:
            payload = await self._get_json(f"{self._zkp_url}/health")
        except (httpx2.HTTPError, ValueError) as exc:
            return EndpointStatus(status="unavailable", url=self._zkp_url, error=_reason(exc))
        status = payload.get("status", "healthy") if isinstance(payload, dict) else "healthy"
        return EndpointStatus(status=str(status), url=self._zkp_url)

    async def _check_cardano(self) -> CardanoStatus:
        """Probe the Cardano devnet's latest block.

        The block number field differs across Yaci Store versions (number vs
        height), so both are tried before giving up on the block.
        """
        try:
            payload = await self._get_json(f"{self._cardano_url}/api/v1/blocks/latest")
        except (httpx2.HTTPError, ValueError) as exc:
            return CardanoStatus(status="unavailable", url=self._cardano_url, error=_reason(exc))
        block = None
        if isinstance(payload, dict):
            raw = payload.get("number") or payload.get("height")
            block = raw if isinstance(raw, int) else None
        return CardanoStatus(status="healthy", url=self._cardano_url, latest_block=block)

    async def _get_json(self, url: str) -> object:
        """Fetch a URL and decode its JSON body, raising on any failure."""
        response = await self._client.get(url)
        response.raise_for_status()
        return response.json()

    async def aclose(self) -> None:
        """Close the underlying client if this instance owns it."""
        if self._owns_client:
            await self._client.aclose()


def _reason(exc: Exception) -> str:
    """Render a probe failure for the status payload."""
    return str(exc) or "unreachable"
