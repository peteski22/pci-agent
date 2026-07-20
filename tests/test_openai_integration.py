"""Live-daemon smoke tests for the OpenAI-compatible backend.

Skipped by default. Enable with ``uv run pytest -m openai``. Requires a
running OpenAI-compatible server (vLLM, SGLang, LM Studio, llama-cpp-python's
``server.py``, …) and the ``PCI_OPENAI_MODEL`` env var pointing at a model
the server exposes. Set ``PCI_OPENAI_BASE_URL`` / ``PCI_OPENAI_API_KEY`` if
the endpoint differs from the default or requires auth.
"""

from __future__ import annotations

import os

import pytest

from pci_agent.models.openai import DEFAULT_OPENAI_BASE_URL, OpenAICompatBackend

pytestmark = pytest.mark.openai


@pytest.fixture
def _live_model() -> str:
    model = os.environ.get("PCI_OPENAI_MODEL")
    if not model:
        pytest.skip(
            "PCI_OPENAI_MODEL not set — export a model the server exposes to "
            "run the integration smoke tests, e.g. PCI_OPENAI_MODEL=Qwen/Qwen3.6-27B"
        )
    return model


def _backend(model: str) -> OpenAICompatBackend:
    return OpenAICompatBackend(
        base_url=os.environ.get("PCI_OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
        model=model,
        api_key=os.environ.get("PCI_OPENAI_API_KEY"),
    )


async def test_smoke_generate(_live_model: str) -> None:
    async with _backend(_live_model) as backend:
        response = await backend.generate("Reply with the single word: pong", max_tokens=8)
    assert response.text.strip() != ""


async def test_smoke_structured(_live_model: str) -> None:
    schema = {
        "type": "object",
        "properties": {"greeting": {"type": "string"}},
        "required": ["greeting"],
    }
    async with _backend(_live_model) as backend:
        result = await backend.generate_structured(
            "Return a friendly greeting as JSON matching the schema.",
            schema,
            max_tokens=64,
        )
    assert "greeting" in result.data
    assert isinstance(result.data["greeting"], str)
