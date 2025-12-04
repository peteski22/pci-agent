"""
Core Agent implementation
"""

from dataclasses import dataclass
from datetime import datetime

from pci_agent.config import AgentConfig
from pci_agent.context import ContextClient
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
        self._llm: object | None = None
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

    async def _load_llm(self) -> object:
        """Load the local language model"""
        # TODO: Integrate with llama-cpp-python
        # This is a placeholder for the actual implementation
        return None

    async def _generate_response(
        self,
        query: str,
        context_items: list,
    ) -> str:
        """Generate a response using the LLM"""
        if self._llm is None:
            # Fallback when no LLM is loaded
            context_summary = ", ".join(item.id for item in context_items)
            return f"[No LLM loaded] Query: {query}, Context: {context_summary}"

        # TODO: Implement actual LLM inference
        return f"Response to: {query}"

    async def close(self) -> None:
        """Cleanup resources"""
        await self._context_client.disconnect()
        self._llm = None
        self._initialized = False
