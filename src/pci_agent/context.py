"""HTTP client for the PCI context store.

The store is a zero-knowledge blob service: every entry is encrypted
client-side before upload, so this client decrypts each blob locally with
the vault password. The wire format matches the WebCrypto vault used by the
PCI apps: PBKDF2-SHA256 key derivation and AES-256-GCM, serialized as a
JSON object of base64 ``ciphertext``/``iv``/``authTag``/``salt`` fields.
"""

from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass
from urllib.parse import quote

import httpx2
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from pci_agent.config import ContextConfig
from pci_agent.errors import ContextUnavailable

_PBKDF2_ITERATIONS = 100_000
_KEY_LENGTH = 32


@dataclass
class ContextItem:
    """An item retrieved from the context store"""

    id: str
    content: str
    score: float
    metadata: dict[str, object] | None = None


def _decrypt_blob(serialized: str, password: str) -> str:
    """Decrypt a vault blob back to its plaintext.

    Args:
        serialized: The stored blob: a JSON object with base64 ``ciphertext``,
            ``iv``, ``authTag``, and ``salt`` fields.
        password: The vault password the blob was encrypted with.

    Returns:
        The decrypted plaintext.

    Raises:
        ContextUnavailable: If the blob is malformed or the password does not
            match.
    """
    try:
        blob = json.loads(serialized)
        salt = base64.b64decode(blob["salt"])
        iv = base64.b64decode(blob["iv"])
        sealed = base64.b64decode(blob["ciphertext"]) + base64.b64decode(blob["authTag"])
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=_KEY_LENGTH,
            salt=salt,
            iterations=_PBKDF2_ITERATIONS,
        )
        key = kdf.derive(password.encode())
        return AESGCM(key).decrypt(iv, sealed, None).decode()
    except (ValueError, KeyError, TypeError, binascii.Error, InvalidTag) as exc:
        raise ContextUnavailable("private data cannot be decrypted") from exc


class ContextClient:
    """Reads a user's entries from the PCI context store.

    Entries come back decrypted: the store only ever holds ciphertext, so
    the vault password from the configuration is required to read content.
    """

    def __init__(self, config: ContextConfig, *, client: httpx2.AsyncClient | None = None) -> None:
        self.config = config
        self._connected = False
        self._owns_client = client is None
        self._client = client or httpx2.AsyncClient(
            base_url=config.endpoint, timeout=config.timeout_seconds
        )
        # Every request must identify the user whose entries are being read,
        # including on injected clients.
        self._client.headers["X-User-ID"] = config.user_id

    async def connect(self) -> None:
        """Verify the store is reachable and mark the client connected.

        Raises:
            ContextUnavailable: If the store's health endpoint cannot be
                reached or reports an error.
        """
        try:
            response = await self._client.get("/health")
            response.raise_for_status()
        except httpx2.HTTPError as exc:
            self._connected = False
            raise ContextUnavailable("context store unreachable") from exc
        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from the context store.

        For a client this instance owns, the connection pool is closed and
        the instance cannot be reused; injected clients stay open and remain
        their owner's responsibility.
        """
        self._connected = False
        if self._owns_client:
            await self._client.aclose()

    async def search(
        self,
        query: str,
        scope: str | None = None,
        limit: int = 10,
    ) -> list[ContextItem]:
        """Return decrypted entries, filtered by scope and query.

        The store offers no server-side search, so filtering happens here:
        ``scope`` keeps entries whose key starts with it, and a non-empty
        ``query`` keeps entries whose decrypted content contains it
        (case-insensitive). There is no relevance ranking; every returned
        item has a score of 1.0.

        Args:
            query: Substring to match against decrypted content; empty
                matches everything.
            scope: Optional key prefix to limit the search.
            limit: Maximum number of results.

        Returns:
            List of matching context items.

        Raises:
            ContextUnavailable: If the store cannot be reached or entries
                cannot be decrypted.
        """
        keys = [key for key in await self._list_keys() if scope is None or key.startswith(scope)]

        items: list[ContextItem] = []
        for key in keys:
            if len(items) >= limit:
                break
            item = await self.get(key)
            if item is None:
                continue
            if query and query.lower() not in item.content.lower():
                continue
            items.append(item)
        return items

    async def get(self, item_id: str) -> ContextItem | None:
        """Return a single decrypted entry, or None if it does not exist.

        Raises:
            ContextUnavailable: If the store cannot be reached, no vault
                password is configured, or the entry cannot be decrypted.
        """
        try:
            response = await self._client.get(f"/entries/{quote(item_id, safe='')}")
            if response.status_code == 404:
                return None
            response.raise_for_status()
            entry = response.json()
        except (httpx2.HTTPError, ValueError) as exc:
            raise ContextUnavailable("context store unreachable") from exc

        if self.config.password is None:
            raise ContextUnavailable("no vault password configured")

        content = _decrypt_blob(str(entry.get("encrypted_data", "")), self.config.password)
        return ContextItem(
            id=str(entry.get("key", item_id)),
            content=content,
            score=1.0,
            metadata={
                "created_at": entry.get("created_at"),
                "updated_at": entry.get("updated_at"),
                "version": entry.get("version"),
            },
        )

    async def _list_keys(self) -> list[str]:
        """Return every entry key in the user's store.

        Raises:
            ContextUnavailable: If the store cannot be reached or responds
                with an unexpected shape.
        """
        try:
            response = await self._client.get("/entries")
            response.raise_for_status()
            payload = response.json()
        except (httpx2.HTTPError, ValueError) as exc:
            raise ContextUnavailable("context store unreachable") from exc

        keys = payload.get("keys") if isinstance(payload, dict) else None
        if not isinstance(keys, list):
            raise ContextUnavailable("context store unreachable")
        return [str(key) for key in keys]

    @property
    def is_connected(self) -> bool:
        """Report the outcome of the most recent connect()/disconnect().

        This is not live reachability: reads proceed regardless, so a store
        that comes up after a failed connect() still serves them.
        """
        return self._connected
