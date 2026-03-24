"""Typed event schema for BaseAgent streaming API.

AgentEvent objects are yielded by BaseAgent.run_stream() as the agent executes.
Each event has a typed EventType so a frontend can render different phases
(thinking, code execution, results, errors, final answers) distinctly.

Typical event sequence for a task that requires code execution:

    RETRIEVAL_START      (if use_tool_retriever=True)
    RETRIEVAL_COMPLETE   (if use_tool_retriever=True)
    THINKING             (agent reasoning block)
    CODE_EXECUTING       (code about to run)
    CODE_RESULT          (execution output)
    THINKING             (next reasoning step, if any)
    ...
    FINAL_ANSWER         (solution tag found, agent terminates)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """Types of events emitted by the agent during execution."""

    THINKING = "thinking"
    """Agent reasoning block (<think> tag content)."""

    CODE_EXECUTING = "code_executing"
    """Code block about to be executed (<execute> tag content)."""

    CODE_RESULT = "code_result"
    """Output returned from code execution (<observation> tag content)."""

    TOOL_SELECTED = "tool_selected"
    """A specific tool was selected for use (emitted by callers that parse tool calls)."""

    ERROR = "error"
    """An error occurred during generation or execution."""

    FINAL_ANSWER = "final_answer"
    """Agent produced a final answer (<solution> tag content)."""

    RETRIEVAL_START = "retrieval_start"
    """Tool retriever started selecting resources."""

    RETRIEVAL_COMPLETE = "retrieval_complete"
    """Tool retriever finished selecting resources."""


@dataclass
class AgentEvent:
    """A typed event emitted during agent execution.

    Attributes:
        event_type: The kind of event (see EventType).
        content: Human-readable payload for this event — code text, execution
            output, reasoning text, answer text, or an error message.
        node_name: The LangGraph node that produced this event
            (``"retrieve"``, ``"generate"``, ``"execute"``, etc.).
        timestamp: ISO 8601 UTC timestamp, auto-populated at creation.
        metadata: Optional key-value pairs for additional context (e.g.
            model name, token counts, language of the code block).
    """

    event_type: EventType
    content: str
    node_name: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict.

        The ``event_type`` field is serialised to its string value so the dict
        can be passed directly to ``json.dumps`` without a custom encoder.
        """
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d

    def to_json(self) -> str:
        """Return a compact JSON string suitable for SSE or WebSocket payloads."""
        return json.dumps(self.to_dict())
