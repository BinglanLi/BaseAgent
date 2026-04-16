"""Example 12: Persistent conversations with checkpoint_db_path and thread_id.

By default BaseAgent uses an in-memory SQLite checkpoint store that is
discarded when the process exits.  Pass a file path to checkpoint_db_path
to persist the conversation graph to disk, then supply the same thread_id
on a subsequent run to resume where you left off.

This example simulates a two-turn conversation within the same process.
In practice you would restart the process between turns — the on-disk
checkpoint means the agent picks up the exact same conversation history.

Run this script from the repo root::

    python examples/12_persistence.py
"""

import os

from BaseAgent import BaseAgent

DB_PATH = "conversation.db"
THREAD_ID = "alzheimer-research-session"


def main():
    # ── Turn 1 ───────────────────────────────────────────────────────────────
    # Initialise with a file-backed checkpoint store and a fixed thread_id.
    agent = BaseAgent(
        llm="claude-sonnet-4-20250514",
        checkpoint_db_path=DB_PATH,
        require_approval="never",
    )

    print("=== Turn 1 ===")
    _, result1 = agent.run(
        "What are the two main protein aggregates found in Alzheimer's disease brains?",
        thread_id=THREAD_ID,
    )
    print(result1)

    # ── Turn 2 (same agent instance, same thread) ─────────────────────────────
    # In a real cross-process scenario you would re-create the agent here with
    # the same checkpoint_db_path and thread_id.  The conversation history is
    # loaded from disk automatically.
    print("\n=== Turn 2 (continued conversation) ===")
    _, result2 = agent.run(
        "Which of those two is the primary therapeutic target and why?",
        thread_id=THREAD_ID,
    )
    print(result2)

    # Cleanup demo file
    agent.close()
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)


if __name__ == "__main__":
    main()
