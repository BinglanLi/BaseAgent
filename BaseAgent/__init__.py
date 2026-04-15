"""
BaseAgent: A flexible and extensible agent framework built on LangChain and LangGraph.
"""

from .base_agent import BaseAgent
from .errors import AgentTimeoutError, BaseAgentError, BudgetExceededError, LLMError, MaxRoundsExceededError
from .events import AgentEvent, EventType
from .resources import Skill
from .agent_spec import AgentSpec
from .multi_agent import AgentTeam

__version__ = "0.1.0"
__author__ = "BaseAgent Contributors"
__all__ = [
    "BaseAgent",
    "AgentTeam",
    "AgentEvent",
    "EventType",
    "Skill",
    "AgentSpec",
    "BaseAgentError",
    "AgentTimeoutError",
    "LLMError",
    "BudgetExceededError",
    "MaxRoundsExceededError",
]