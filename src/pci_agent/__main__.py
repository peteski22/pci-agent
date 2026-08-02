"""PCI Agent entry point: serves the FastAPI coordination app via uvicorn."""

from __future__ import annotations

import os

import uvicorn

from pci_agent.api import create_app
from pci_agent.config import AgentConfig


def main() -> None:
    """Start the PCI Agent HTTP server."""
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8082"))
    app = create_app(AgentConfig.from_env())
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
