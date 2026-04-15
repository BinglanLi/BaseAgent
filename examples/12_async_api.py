"""Example: async API — arun(), aresume(), areject().

Use these methods when you need non-blocking execution, e.g. inside an
orchestrator or an async web server.  They mirror run(), resume(), and
reject() exactly but use LangGraph's async streaming under the hood.

Run this script from the repo root::

    python examples/12_async_api.py
"""

from __future__ import annotations

import asyncio

from BaseAgent import BaseAgent


async def basic_async_run():
    """arun() returns (log, content) just like run()."""
    agent = BaseAgent(llm="azure-claude-sonnet-4-5")
    log, answer = await agent.arun("What is 2 + 2?")
    print("Answer:", answer)
    print(f"Log entries: {len(log)}")


async def async_hitl():
    """HITL workflow using arun() / aresume() / areject()."""
    agent = BaseAgent(
        llm="azure-claude-sonnet-4-5",
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
    agent_a = BaseAgent(llm="azure-claude-sonnet-4-5")
    agent_b = BaseAgent(llm="azure-claude-sonnet-4-5")

    (_, ans_a), (_, ans_b) = await asyncio.gather(
        agent_a.arun("What is the capital of France?"),
        agent_b.arun("What is the capital of Germany?"),
    )
    print("Agent A:", ans_a)
    print("Agent B:", ans_b)


if __name__ == "__main__":
    print("=== Basic async run ===")
    asyncio.run(basic_async_run())

    print("\n=== Async HITL ===")
    asyncio.run(async_hitl())

    print("\n=== Concurrent agents ===")
    asyncio.run(concurrent_agents())
