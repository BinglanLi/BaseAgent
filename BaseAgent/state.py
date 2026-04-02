"""Agent state definition shared between base_agent and nodes."""

from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    input: list[BaseMessage]
    next_step: str | None
    pending_code: str | None       # code block awaiting human approval
    pending_language: str | None   # detected language ("python", "bash", "r")
