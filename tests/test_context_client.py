"""Tests for the context-store HTTP client.

Uses ``httpx2.MockTransport`` so no store is required. Entry fixtures are
encrypted with the same PBKDF2/AES-GCM format the demo user app produces,
so decryption is exercised against realistic blobs.
"""

import base64
import json
import os

import httpx2
import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from pci_agent.config import ContextConfig
from pci_agent.context import ContextClient, ContextItem
from pci_agent.errors import ContextUnavailable

PASSWORD = "vault-pass"


def _encrypt(plaintext: str, password: str) -> str:
    """Produce a blob in the WebCrypto vault format used by the demo apps."""
    salt = os.urandom(32)
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100_000)
    key = kdf.derive(password.encode())
    iv = os.urandom(12)
    sealed = AESGCM(key).encrypt(iv, plaintext.encode(), None)
    return json.dumps(
        {
            "ciphertext": base64.b64encode(sealed[:-16]).decode(),
            "iv": base64.b64encode(iv).decode(),
            "authTag": base64.b64encode(sealed[-16:]).decode(),
            "salt": base64.b64encode(salt).decode(),
        }
    )


def _entry(key: str, plaintext: str, *, version: int = 1) -> dict[str, object]:
    return {
        "key": key,
        "encrypted_data": _encrypt(plaintext, PASSWORD),
        "created_at": "2026-08-14T00:00:00.000Z",
        "updated_at": "2026-08-14T00:00:00.000Z",
        "version": version,
    }


def _client(handler, *, password: str | None = PASSWORD, user_id: str = "demo-user") -> ContextClient:
    config = ContextConfig(endpoint="http://store", user_id=user_id, password=password)
    transport = httpx2.MockTransport(handler)
    http = httpx2.AsyncClient(transport=transport, base_url="http://store")
    return ContextClient(config, client=http)


def _store_handler(entries: dict[str, dict[str, object]]):
    def handler(request: httpx2.Request) -> httpx2.Response:
        path = request.url.path
        if path == "/health":
            return httpx2.Response(200, json={"status": "healthy", "users": 1})
        if path == "/entries":
            return httpx2.Response(200, json={"keys": list(entries)})
        if path.startswith("/entries/"):
            key = path.removeprefix("/entries/")
            if key in entries:
                return httpx2.Response(200, json=entries[key])
            return httpx2.Response(404, json={"error": "Not found"})
        return httpx2.Response(404, json={"error": "Not found"})

    return handler


async def test_connect_succeeds_against_healthy_store():
    client = _client(_store_handler({}))
    await client.connect()
    assert client.is_connected is True
    await client.disconnect()
    assert client.is_connected is False


async def test_connect_raises_when_store_unreachable():
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused")

    client = _client(handler)
    with pytest.raises(ContextUnavailable):
        await client.connect()
    assert client.is_connected is False
    await client.disconnect()


async def test_connect_sends_user_id_header():
    seen: list[str | None] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request.headers.get("X-User-ID"))
        return httpx2.Response(200, json={"status": "healthy"})

    client = _client(handler, user_id="alice")
    await client.connect()
    assert seen == ["alice"]
    await client.disconnect()


async def test_search_returns_decrypted_items():
    plaintext = json.dumps({"birthDate": "1990-05-15", "fullName": "Alice Demo"})
    client = _client(_store_handler({"personal-context": _entry("personal-context", plaintext, version=3)}))

    items = await client.search("")

    assert len(items) == 1
    item = items[0]
    assert isinstance(item, ContextItem)
    assert item.id == "personal-context"
    assert json.loads(item.content) == {"birthDate": "1990-05-15", "fullName": "Alice Demo"}
    assert item.metadata is not None
    assert item.metadata["version"] == 3
    await client.disconnect()


async def test_search_scope_filters_by_key_prefix():
    entries = {
        "personal-context": _entry("personal-context", '{"birthDate": "1990-05-15"}'),
        "finance-history": _entry("finance-history", '{"balance": 12}'),
    }
    client = _client(_store_handler(entries))

    items = await client.search("", scope="personal")

    assert [item.id for item in items] == ["personal-context"]
    await client.disconnect()


async def test_search_query_filters_content():
    entries = {
        "a": _entry("a", '{"fullName": "Alice Demo"}'),
        "b": _entry("b", '{"fullName": "Bob Demo"}'),
    }
    client = _client(_store_handler(entries))

    items = await client.search("alice")

    assert [item.id for item in items] == ["a"]
    await client.disconnect()


async def test_search_respects_limit():
    entries = {f"key-{i}": _entry(f"key-{i}", f'{{"n": {i}}}') for i in range(5)}
    client = _client(_store_handler(entries))

    items = await client.search("", limit=2)

    assert len(items) == 2
    await client.disconnect()


async def test_search_skips_entry_deleted_mid_scan():
    plaintext = '{"birthDate": "1990-05-15"}'
    entries = {"kept": _entry("kept", plaintext)}

    def handler(request: httpx2.Request) -> httpx2.Response:
        path = request.url.path
        if path == "/entries":
            return httpx2.Response(200, json={"keys": ["gone", "kept"]})
        if path == "/entries/gone":
            return httpx2.Response(404, json={"error": "Not found"})
        if path == "/entries/kept":
            return httpx2.Response(200, json=entries["kept"])
        return httpx2.Response(404, json={"error": "Not found"})

    client = _client(handler)
    items = await client.search("")
    assert [item.id for item in items] == ["kept"]
    await client.disconnect()


async def test_search_raises_unavailable_on_transport_error():
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused")

    client = _client(handler)
    with pytest.raises(ContextUnavailable):
        await client.search("")
    await client.disconnect()


async def test_search_raises_without_password():
    client = _client(_store_handler({"k": _entry("k", "{}")}), password=None)
    with pytest.raises(ContextUnavailable):
        await client.search("")
    await client.disconnect()


async def test_search_raises_on_wrong_password():
    client = _client(_store_handler({"k": _entry("k", "{}")}), password="not-the-vault-pass")
    with pytest.raises(ContextUnavailable):
        await client.search("")
    await client.disconnect()


async def test_get_returns_decrypted_item():
    plaintext = '{"birthDate": "1990-05-15"}'
    client = _client(_store_handler({"personal-context": _entry("personal-context", plaintext)}))

    item = await client.get("personal-context")

    assert item is not None
    assert item.id == "personal-context"
    assert json.loads(item.content) == {"birthDate": "1990-05-15"}
    await client.disconnect()


async def test_get_returns_none_when_missing():
    client = _client(_store_handler({}))
    assert await client.get("nope") is None
    await client.disconnect()
