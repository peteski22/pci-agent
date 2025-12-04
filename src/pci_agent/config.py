"""
Agent configuration
"""

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for the local language model"""

    model_path: str | None = None
    context_length: int = Field(default=4096, ge=512, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    n_gpu_layers: int = Field(default=0, ge=0)
    n_threads: int | None = None


class ContextConfig(BaseModel):
    """Configuration for context store connection"""

    endpoint: str = "http://localhost:8080"
    timeout_seconds: float = 30.0


class AgentConfig(BaseModel):
    """Main agent configuration"""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)

    # Agent behavior
    max_context_items: int = Field(default=10, ge=1, le=100)
    audit_logging: bool = True
