"""
Custom Configuration Example
=============================

This example shows how to customize agent configuration,
including the context window sliding window (Feature 5).
"""

from BaseAgent import BaseAgent

# Default: no context window limit
agent = BaseAgent(llm="claude-sonnet-4-20250514")

# Enable sliding window: pass at most 20 messages to the LLM per call.
# The first message (user task) and the most recent 19 messages are kept.
# The full conversation history is still stored in state for checkpointing.
agent_windowed = BaseAgent(
    llm="claude-sonnet-4-20250514",
    max_context_messages=20,
)

# Run a long task — context will not overflow even after many iterations
log, result = agent_windowed.run("Analyze the structure of the human proteome")
print(result)
