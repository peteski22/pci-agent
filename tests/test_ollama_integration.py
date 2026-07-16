"""Live-daemon smoke tests for the Ollama backend.

Skipped by default. Enable with ``uv run pytest -m ollama``. Requires a
local Ollama daemon and the ``PCI_OLLAMA_MODEL`` env var pointing at a
model tag that has already been pulled (``ollama pull qwen3.6:27b``).
"""

from __future__ import annotations

import os

import pytest

from pci_agent.models.ollama import DEFAULT_OLLAMA_URL, OllamaBackend

pytestmark = pytest.mark.ollama


@pytest.fixture
def _live_model() -> str:
    model = os.environ.get("PCI_OLLAMA_MODEL")
    if not model:
        pytest.skip(
            "PCI_OLLAMA_MODEL not set — export a pulled tag to run the "
            "integration smoke tests, e.g. PCI_OLLAMA_MODEL=phi4-mini:3.8b"
        )
    return model


async def test_smoke_generate(_live_model: str) -> None:
    base_url = os.environ.get("PCI_OLLAMA_URL", DEFAULT_OLLAMA_URL)
    async with OllamaBackend(base_url=base_url, model=_live_model) as backend:
        response = await backend.generate("Reply with the single word: pong", max_tokens=8)
    assert response.text.strip() != ""


async def test_smoke_structured(_live_model: str) -> None:
    base_url = os.environ.get("PCI_OLLAMA_URL", DEFAULT_OLLAMA_URL)
    schema = {
        "type": "object",
        "properties": {"greeting": {"type": "string"}},
        "required": ["greeting"],
    }
    async with OllamaBackend(base_url=base_url, model=_live_model) as backend:
        result = await backend.generate_structured(
            "Return a friendly greeting as JSON matching the schema.",
            schema,
            max_tokens=64,
        )
    assert "greeting" in result.data
    assert isinstance(result.data["greeting"], str)
