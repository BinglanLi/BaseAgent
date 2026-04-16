"""Example 9: Async API and streaming events.

Which method to use
-------------------
arun()        — async equivalent of run(); returns (log, content) when the
                agent finishes.  Use inside async orchestrators, web servers,
                or when running multiple agents concurrently with asyncio.gather().
run_stream()  — async generator that yields AgentEvent objects *during*
                execution.  Use when you need real-time visibility into
                reasoning, code execution, and errors (e.g. feeding a UI or
                collecting a fine-grained trace).

Both require an async context (async def / asyncio.run()).

Run this script from the repo root::

    python examples/09_async_and_streaming.py
"""

from __future__ import annotations

import asyncio

from BaseAgent import BaseAgent, EventType


# ===========================================================================
# Part A — arun(), aresume(), areject()
# ===========================================================================

async def basic_async_run():
    """arun() returns (log, content) just like run()."""
    agent = BaseAgent(llm="claude-sonnet-4-20250514")
    log, answer = await agent.arun("What is 2 + 2?")
    print("Answer:", answer)
    print(f"Log entries: {len(log)}")


async def async_hitl():
    """HITL workflow using arun() / aresume() / areject()."""
    agent = BaseAgent(
        llm="claude-sonnet-4-20250514",
        require_approval="always",
    )

    _, payload = await agent.arun("Compute the sum of 1 through 10 in Python.")

    if agent.is_interrupted:
        print("Code pending approval:")
        if isinstance(payload, dict):
            print(payload.get("code", ""))

        # --- Option A: approve ---
        _, answer = await agent.aresume()
        print("Approved. Final answer:", answer)

        # --- Option B: reject with feedback (commented out) ---
        # _, payload2 = await agent.areject("Use a list comprehension instead.")
        # if not agent.is_interrupted:
        #     print("After feedback:", payload2)
    else:
        print("Completed without interruption:", payload)


async def concurrent_agents():
    """Run two independent agents concurrently with asyncio.gather()."""
    agent_a = BaseAgent(llm="claude-sonnet-4-20250514")
    agent_b = BaseAgent(llm="claude-sonnet-4-20250514")

    (_, ans_a), (_, ans_b) = await asyncio.gather(
        agent_a.arun("What is the capital of France?"),
        agent_b.arun("What is the capital of Germany?"),
    )
    print("Agent A:", ans_a)
    print("Agent B:", ans_b)


# ===========================================================================
# Part B — run_stream() and AgentEvent
# ===========================================================================

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
    print("=== Basic async run ===")
    asyncio.run(basic_async_run())

    print("\n=== Async HITL ===")
    asyncio.run(async_hitl())

    print("\n=== Concurrent agents ===")
    asyncio.run(concurrent_agents())

    print("\n=== Streaming: all events ===")
    asyncio.run(stream_all_events())

    asyncio.run(stream_filtered_events())
    asyncio.run(stream_to_json())
