"""
Core Agent implementation.

The agent has two orthogonal responsibilities:

1. Answer user queries with retrieved context (``process``) — routed to
   whichever LLM backend is configured.
2. Synthesise :class:`~pci_agent.spal.RequestContext` candidates for the
   S-PAL flow (``propose_request_context``) — the "wiring hook" between the
   deterministic policy checker and the model. Requires the Ollama backend
   (structured-output support).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from pci_agent.config import AgentConfig
from pci_agent.context import ContextClient, ContextItem
from pci_agent.models.local_llm import LocalLLM
from pci_agent.models.ollama import OllamaBackend
from pci_agent.models.openai import OpenAICompatBackend
from pci_agent.models.spal_bridge import (
    propose_request_context as _propose_request_context,
)
from pci_agent.policy import PolicyChecker
from pci_agent.spal import RequestContext


@dataclass
class AgentResponse:
    """Response from agent processing"""

    content: str
    context_used: list[str]
    policy_applied: str | None
    timestamp: datetime


class Agent:
    """
    PCI Personal Agent

    Processes queries using local AI while enforcing S-PAL policies
    and retrieving context from the encrypted store.
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()
        self._llm: LocalLLM | None = None
        self._ollama_backend: OllamaBackend | None = None
        self._openai_backend: OpenAICompatBackend | None = None
        self._context_client = ContextClient(self.config.context)
        self._policy_checker = PolicyChecker()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the agent (load model, connect to context store)"""
        if self._initialized:
            return

        # Load whichever backend the config selects. The llama-cpp path is
        # gated on ``model_path`` for backward compatibility; the Ollama
        # backend has no local file to load — the daemon is expected to be
        # running separately.
        if self.config.llm.backend == "llamacpp":
            if self.config.llm.model_path:
                self._llm = await self._load_llm()
        elif self.config.llm.backend == "ollama":
            self._ollama_backend = self._build_ollama_backend()
        elif self.config.llm.backend == "openai":
            self._openai_backend = self._build_openai_backend()

        try:
            await self._context_client.connect()
        except BaseException:
            # Release any backend resources we just acquired so a retry can
            # start from a clean slate; ``_initialized`` stays False.
            await self._release_backend()
            raise

        self._initialized = True

    async def process(
        self,
        query: str,
        policy_id: str | None = None,
        context_scope: str | None = None,
    ) -> AgentResponse:
        """
        Process a query with policy enforcement

        Args:
            query: The user's query
            policy_id: Optional S-PAL policy to apply
            context_scope: Optional scope to limit context retrieval

        Returns:
            AgentResponse with the result
        """
        if not self._initialized:
            await self.initialize()

        # Check policy if specified
        if policy_id:
            policy_result = await self._policy_checker.check(
                policy_id, query, context_scope=context_scope
            )
            if not policy_result.allowed:
                return AgentResponse(
                    content=f"Request blocked by policy: {policy_result.reason}",
                    context_used=[],
                    policy_applied=policy_id,
                    timestamp=datetime.now(),
                )

        # Retrieve relevant context
        context_items = await self._context_client.search(
            query,
            scope=context_scope,
            limit=self.config.max_context_items,
        )

        # Generate response
        response_content = await self._generate_response(query, context_items)

        return AgentResponse(
            content=response_content,
            context_used=[item.id for item in context_items],
            policy_applied=policy_id,
            timestamp=datetime.now(),
        )

    async def propose_request_context(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> RequestContext:
        """Ask the LLM to synthesise a ``RequestContext`` for policy evaluation.

        This is the S-PAL wiring point: the caller supplies a natural-language
        description of an incoming business request and gets back a validated
        ``RequestContext`` ready for
        :meth:`pci_agent.policy.PolicyChecker.check`.

        The LLM is a composer only — ``PolicyChecker`` remains the sole
        adjudicator.
        """
        if not self._initialized:
            await self.initialize()
        if self._ollama_backend is None:
            raise RuntimeError(
                "propose_request_context requires the Ollama backend "
                "(set LLMConfig.backend='ollama' or PCI_LLM_BACKEND=ollama)"
            )
        return await _propose_request_context(
            self._ollama_backend,
            prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def _load_llm(self) -> LocalLLM:
        """Load the local language model"""
        llm = LocalLLM(self.config.llm)
        await llm.load()
        return llm

    def _build_ollama_backend(self) -> OllamaBackend:
        """Construct an Ollama backend from the current LLM config."""
        cfg = self.config.llm
        return OllamaBackend(
            base_url=cfg.ollama_base_url,
            model=cfg.ollama_model,
            tier=cfg.ollama_tier if cfg.ollama_model is None else None,
            request_timeout=cfg.ollama_timeout_seconds,
            temperature=cfg.temperature,
        )

    def _build_openai_backend(self) -> OpenAICompatBackend:
        """Construct an OpenAI-compatible backend from the current LLM config."""
        cfg = self.config.llm
        return OpenAICompatBackend(
            base_url=cfg.openai_base_url,
            model=cfg.openai_model,
            tier=cfg.openai_tier if cfg.openai_model is None else None,
            api_key=cfg.openai_api_key,
            request_timeout=cfg.openai_timeout_seconds,
            temperature=cfg.temperature,
        )

    async def _generate_response(
        self,
        query: str,
        context_items: list[ContextItem],
    ) -> str:
        """Generate a response using the configured LLM backend."""
        prompt = self._build_prompt(query, context_items)

        if self._llm is not None:
            response = await self._llm.generate(prompt)
            return response.text
        if self._ollama_backend is not None:
            response = await self._ollama_backend.generate(prompt)
            return response.text
        if self._openai_backend is not None:
            response = await self._openai_backend.generate(prompt)
            return response.text

        context_summary = ", ".join(item.id for item in context_items)
        return f"[No LLM loaded] Query: {query}, Context: {context_summary}"

    @staticmethod
    def _build_prompt(query: str, context_items: list[ContextItem]) -> str:
        """Build a prompt from the query and retrieved context"""
        parts: list[str] = []
        if context_items:
            parts.append("Context:")
            for item in context_items:
                parts.append(f"- {item.content}")
            parts.append("")
        parts.append(f"Question: {query}")
        parts.append("Answer:")
        return "\n".join(parts)

    async def close(self) -> None:
        """Cleanup resources.

        Runs context-store disconnect and backend release regardless of
        intermediate failures — a raising :meth:`ContextClient.disconnect`
        must not leak the LLM backend or leave stale lifecycle state.
        """
        try:
            await self._context_client.disconnect()
        finally:
            try:
                await self._release_backend()
            finally:
                self._initialized = False

    async def _release_backend(self) -> None:
        """Release whichever backend is currently loaded, if any."""
        if self._llm is not None:
            try:
                await self._llm.unload()
            finally:
                self._llm = None
        if self._ollama_backend is not None:
            try:
                await self._ollama_backend.aclose()
            finally:
                self._ollama_backend = None
        if self._openai_backend is not None:
            try:
                await self._openai_backend.aclose()
            finally:
                self._openai_backend = None
