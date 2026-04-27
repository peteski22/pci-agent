"""
Core Agent implementation
"""

from dataclasses import dataclass
from datetime import datetime

from pci_agent.config import AgentConfig
from pci_agent.context import ContextClient, ContextItem
from pci_agent.models.local_llm import LocalLLM
from pci_agent.policy import PolicyChecker


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
        self._context_client = ContextClient(self.config.context)
        self._policy_checker = PolicyChecker()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the agent (load model, connect to context store)"""
        if self._initialized:
            return

        # Load LLM if model path provided
        if self.config.llm.model_path:
            self._llm = await self._load_llm()

        # Connect to context store
        await self._context_client.connect()

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
            policy_result = await self._policy_checker.check(policy_id, query)
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

    async def _load_llm(self) -> LocalLLM:
        """Load the local language model"""
        llm = LocalLLM(self.config.llm)
        await llm.load()
        return llm

    async def _generate_response(
        self,
        query: str,
        context_items: list[ContextItem],
    ) -> str:
        """Generate a response using the LLM"""
        if self._llm is None:
            context_summary = ", ".join(item.id for item in context_items)
            return f"[No LLM loaded] Query: {query}, Context: {context_summary}"

        prompt = self._build_prompt(query, context_items)
        response = await self._llm.generate(prompt)
        return response.text

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
        """Cleanup resources"""
        await self._context_client.disconnect()
        if self._llm is not None:
            await self._llm.unload()
            self._llm = None
        self._initialized = False
