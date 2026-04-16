"""Example 15: Streaming events with run_stream().

run_stream() is an async generator that yields AgentEvent objects as the agent
executes — one event per reasoning block, code execution, or final answer.
Use it to feed a UI or collect fine-grained execution traces in real time.

Run this script from the repo root::

    python examples/15_streaming.py
"""

import asyncio

from BaseAgent import BaseAgent, EventType


async def stream_all_events():
    """Print every event emitted during a run."""
    agent = BaseAgent(llm="claude-sonnet-4-20250514", require_approval="never")

    print("=== All events ===")
    async for event in agent.run_stream("What is the GC content of the sequence ATCGATCG?"):
        print(f"[{event.event_type.value:20s}] {event.content[:120]}")


async def stream_filtered_events():
    """Receive only THINKING and FINAL_ANSWER events."""
    agent = BaseAgent(llm="claude-sonnet-4-20250514", require_approval="never")

    print("\n=== Filtered: THINKING + FINAL_ANSWER only ===")
    async for event in agent.run_stream(
        "Summarise the role of BRCA1 in DNA repair.",
        event_types={EventType.THINKING, EventType.FINAL_ANSWER},
    ):
        label = "Reasoning" if event.event_type == EventType.THINKING else "Answer"
        print(f"[{label}] {event.content[:200]}")


async def stream_to_json():
    """Serialise each event to JSON (useful for WebSocket / SSE payloads)."""
    agent = BaseAgent(llm="claude-sonnet-4-20250514", require_approval="never")

    print("\n=== JSON payloads ===")
    async for event in agent.run_stream(
        "Name three model organisms used in genetics research.",
        event_types={EventType.FINAL_ANSWER},
    ):
        print(event.to_json())


if __name__ == "__main__":
    asyncio.run(stream_all_events())
    asyncio.run(stream_filtered_events())
    asyncio.run(stream_to_json())
