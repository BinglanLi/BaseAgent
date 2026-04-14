"""Structured error types for BaseAgent.

All agent errors inherit from BaseAgentError, allowing callers to catch the
entire family with a single ``except BaseAgentError`` clause, or to handle
specific failure modes individually.
"""


class BaseAgentError(Exception):
    """Base class for all BaseAgent errors."""


class ExecutionError(BaseAgentError):
    """Code execution failed (runtime error, syntax error, import error)."""

    def __init__(self, message: str, code: str = "", language: str = "python"):
        super().__init__(message)
        self.code = code
        self.language = language


class ParseError(BaseAgentError):
    """LLM response missing required XML tags after retries."""


class AgentTimeoutError(BaseAgentError):
    """Code execution exceeded the configured timeout."""

    def __init__(self, message: str, timeout_seconds: float = 0.0):
        super().__init__(message)
        self.timeout_seconds = timeout_seconds


class LLMError(BaseAgentError):
    """LLM invocation failed (API error, rate limit, authentication)."""


class BudgetExceededError(BaseAgentError):
    """Execution exceeded the configured cost budget."""

    def __init__(self, message: str, cost: float = 0.0, budget: float = 0.0):
        super().__init__(message)
        self.cost = cost
        self.budget = budget
