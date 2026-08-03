import httpx2

from pci_agent.status import StatusClient


def _client(handler) -> StatusClient:
    transport = httpx2.MockTransport(handler)
    http = httpx2.AsyncClient(transport=transport)
    return StatusClient(
        agent_url="http://agent",
        zkp_url="http://zkp",
        cardano_url="http://cardano",
        client=http,
    )


async def test_check_reports_healthy_services_and_latest_block():
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.host == "zkp":
            return httpx2.Response(200, json={"status": "healthy"})
        return httpx2.Response(200, json={"number": 123})

    client = _client(handler)
    result = await client.check()
    assert result.agent.status == "healthy"
    assert result.zkp.status == "healthy"
    assert result.cardano.status == "healthy"
    assert result.cardano.latest_block == 123
    await client.aclose()


async def test_check_reports_unavailable_on_transport_error():
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused")

    client = _client(handler)
    result = await client.check()
    assert result.zkp.status == "unavailable"
    assert result.zkp.error
    assert result.cardano.status == "unavailable"
    assert result.cardano.latest_block is None
    await client.aclose()


async def test_check_reports_unavailable_on_http_error_status():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(500)

    client = _client(handler)
    result = await client.check()
    assert result.zkp.status == "unavailable"
    assert result.cardano.status == "unavailable"
    await client.aclose()


async def test_check_uses_cardano_height_fallback():
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.host == "zkp":
            return httpx2.Response(200, json={})
        return httpx2.Response(200, json={"height": 7})

    client = _client(handler)
    result = await client.check()
    assert result.zkp.status == "healthy"
    assert result.cardano.latest_block == 7
    await client.aclose()


async def test_check_tolerates_non_numeric_block():
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.host == "zkp":
            return httpx2.Response(200, json={"status": "healthy"})
        return httpx2.Response(200, json={"number": "not-a-block"})

    client = _client(handler)
    result = await client.check()
    assert result.cardano.status == "healthy"
    assert result.cardano.latest_block is None
    await client.aclose()
