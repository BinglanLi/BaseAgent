"""Multi-agent state schema for AgentTeam orchestration."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class MultiAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # conversation log
    next_agent: str     # routing decision; "FINISH" = done
    sub_task: str       # task instruction for the chosen agent (set by supervisor)
    task: str           # original user task (immutable)
    results: dict[str, str]  # agent_name -> output string (last write wins if agent runs twice)
    round: int          # supervisor call count; checked against max_rounds
