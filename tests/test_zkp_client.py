import httpx
import pytest

from pci_agent.errors import ZKPUnavailable
from pci_agent.zkp import ZKPClient, ZKPResult


def _client(handler) -> ZKPClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url="http://zkp")
    return ZKPClient("http://zkp", client=http)


async def test_generate_returns_verified_result():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"proof": {"publicSignals": {"verified": True}}})

    client = _client(handler)
    result = await client.generate("age", {"minAge": 18, "birthDate": "2000-01-01"})
    assert isinstance(result, ZKPResult)
    assert result.verified is True
    await client.aclose()


async def test_generate_raises_on_transport_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = _client(handler)
    with pytest.raises(ZKPUnavailable):
        await client.generate("age", {})
    await client.aclose()


async def test_generate_raises_on_non_dict_proof_shape():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"proof": "oops"})

    client = _client(handler)
    with pytest.raises(ZKPUnavailable):
        await client.generate("age", {})
    await client.aclose()


async def test_generate_raises_on_non_dict_body_shape():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    client = _client(handler)
    with pytest.raises(ZKPUnavailable):
        await client.generate("age", {})
    await client.aclose()
