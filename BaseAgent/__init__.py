"""
BaseAgent: A flexible and extensible agent framework built on LangChain and LangGraph.
"""

from .base_agent import BaseAgent
from .errors import (
    AgentTimeoutError,
    BaseAgentError,
    BudgetExceededError,
    ExecutionError,
    LLMError,
    ParseError,
)
from .events import AgentEvent, EventType
from .resources import Skill
from .agent_spec import AgentSpec

__version__ = "0.1.0"
__author__ = "BaseAgent Contributors"
__all__ = [
    "BaseAgent",
    "AgentEvent",
    "EventType",
    "Skill",
    "AgentSpec",
    "BaseAgentError",
    "ExecutionError",
    "ParseError",
    "AgentTimeoutError",
    "LLMError",
    "BudgetExceededError",
]