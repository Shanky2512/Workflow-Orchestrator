"""
Runtime guards.
Enforces cost, timeout, and step limits during execution.
"""
from typing import Dict, Any


class RuntimeGuards:
    """
    Runtime safety guards.
    Prevents runaway costs, infinite loops, and timeout violations.
    """

    def __init__(
        self,
        max_total_tokens: int = 100000,
        max_execution_seconds: int = 300,
        max_total_steps: int = 50
    ):
        """
        Initialize guards.

        Args:
            max_total_tokens: Maximum tokens across all agents
            max_execution_seconds: Maximum execution time
            max_total_steps: Maximum total steps
        """
        self.max_total_tokens = max_total_tokens
        self.max_execution_seconds = max_execution_seconds
        self.max_total_steps = max_total_steps

    def check_before_execution(self, workflow: Dict[str, Any]) -> None:
        """
        Check guards before starting execution.

        Args:
            workflow: Workflow definition

        Raises:
            RuntimeError: If guards are violated
        """
        # Check agent constraints
        total_budget = 0
        # Handle both embedded agent objects and ID-referenced agents
        for agent_entry in workflow.get("agents", []):
            if isinstance(agent_entry, str):
                agent_id = agent_entry
            elif isinstance(agent_entry, dict):
                agent_id = agent_entry.get("agent_id")
            else:
                continue
            # TODO: Load actual agent and check constraints
            pass

        # Additional pre-flight checks can be added here

    def check_during_execution(
        self,
        elapsed_seconds: float,
        tokens_used: int,
        steps_taken: int
    ) -> None:
        """
        Check guards during execution.

        Args:
            elapsed_seconds: Elapsed execution time
            tokens_used: Tokens consumed so far
            steps_taken: Steps executed so far

        Raises:
            RuntimeError: If any guard is violated
        """
        if elapsed_seconds > self.max_execution_seconds:
            raise RuntimeError(
                f"Execution timeout: {elapsed_seconds}s exceeds {self.max_execution_seconds}s"
            )

        if tokens_used > self.max_total_tokens:
            raise RuntimeError(
                f"Token budget exceeded: {tokens_used} exceeds {self.max_total_tokens}"
            )

        if steps_taken > self.max_total_steps:
            raise RuntimeError(
                f"Step limit exceeded: {steps_taken} exceeds {self.max_total_steps}"
            )
