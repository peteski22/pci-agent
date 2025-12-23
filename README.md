# PCI Agent

Layer 2: Local AI agent for Personal Context Infrastructure.

## Overview

The PCI Agent provides:

- **Local AI Processing** - Run language models on your device
- **Context Retrieval** - Query encrypted context store
- **Policy Enforcement** - Validate requests against S-PAL policies
- **Decision Support** - AI-powered assistance without data leakage

## Installation

```bash
# Clone and install
git clone https://github.com/peteski22/pci-agent.git
cd pci-agent
uv sync

# With LLM support (llama-cpp-python)
uv sync --extra llm
```

## Quick Start

```python
from pci_agent import Agent, AgentConfig

# Initialize agent
agent = Agent(AgentConfig(
    model_path="models/phi-3-mini-q4.gguf",
))

# Process a request with policy enforcement
response = await agent.process(
    query="What are my upcoming appointments?",
    policy_id="calendar-access",
)

print(response.content)
```

## Architecture

```mermaid
flowchart TB
    subgraph Agent["PCI Agent"]
        QH["Query Handler"]
        QH --> PC["Policy Checker"]
        QH --> CR["Context Retriever"]
        QH --> LLM["LLM Inference"]
    end

    PC --> SPAL["S-PAL Policy"]
    CR --> CS["Context Store"]
```

## Supported Models

The agent works with GGUF-format models:

| Model | Parameters | RAM Required | Quality |
|-------|------------|--------------|---------|
| Phi-3 Mini | 3.8B | 4GB | Good |
| Llama 3.2 | 3B | 3GB | Good |
| Mistral | 7B | 8GB | Better |
| Llama 3.1 | 8B | 10GB | Best |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8082` | HTTP server port |
| `ZKP_SERVICE_URL` | `http://localhost:8084` | ZKP service endpoint |
| `CARDANO_API_URL` | `http://localhost:8080` | Cardano devnet API endpoint |

## Development

```bash
# Install with dev dependencies
uv sync --group dev

# Run tests
uv run pytest

# Type check
uv run mypy src/

# Lint
uv run ruff check src/

# Run the agent
uv run python -m pci_agent
```

## Related Packages

- [pci-spec](https://github.com/peteski22/pci-spec) - S-PAL schema and protocols
- [pci-context-store](https://github.com/peteski22/pci-context-store) - Layer 1: Context Store
- [pci-contracts](https://github.com/peteski22/pci-contracts) - Layer 3: Smart Contracts
- [pci-zkp](https://github.com/peteski22/pci-zkp) - Layer 4: Zero-Knowledge Proofs
- [pci-identity](https://github.com/peteski22/pci-identity) - Layer 5: DID Management

## License

Apache 2.0
