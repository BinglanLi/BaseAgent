"""
Error Handling and Termination Conditions
==========================================

This example shows how to configure termination limits and catch
structured errors from BaseAgent.

Run this script from the repo root::

    python examples/10_error_handling.py
"""

import os

from BaseAgent import BaseAgent, BudgetExceededError, LLMError, AgentTimeoutError, BaseAgentError
from BaseAgent.config import default_config

# ── 1. Iteration limit ────────────────────────────────────────────────────────
# Stop after at most 5 think/execute cycles, regardless of task completion.
# Useful when you want a quick answer and are willing to accept partial results.

agent = BaseAgent(llm="claude-sonnet-4-20250514")
agent.max_iterations = 5

log, result = agent.run("Write a Python function to compute Fibonacci numbers")
print(result)

# ── 2. Consecutive-error limit ────────────────────────────────────────────────
# Stop if infrastructure fails (timeout, crash) 3 times in a row.
# Normal code errors (SyntaxError, NameError) do NOT count — the agent is
# expected to fix its own code.

agent2 = BaseAgent(llm="claude-sonnet-4-20250514")
agent2.max_consecutive_errors = 3

log2, result2 = agent2.run("Parse this CSV file: data.csv")
print(result2)

# ── 3. Cost budget ────────────────────────────────────────────────────────────
# Stop if per-run spend exceeds $0.50.
# Note: only applies when the LLM provider returns cost metadata.
# For Anthropic, set this as a safeguard for future cost-returning models.

agent3 = BaseAgent(llm="claude-sonnet-4-20250514")
agent3.max_cost = 0.50

log3, result3 = agent3.run("Summarize the human genome project in 3 sentences")
print(result3)

# ── 4. Catch structured errors ────────────────────────────────────────────────
# All BaseAgent errors inherit from BaseAgentError.

agent4 = BaseAgent(llm="claude-sonnet-4-20250514")
try:
    log4, result4 = agent4.run("Analyze protein folding dynamics")
except LLMError as e:
    print(f"LLM call failed: {e}")
except AgentTimeoutError as e:
    print(f"Code timed out after {e.timeout_seconds}s")
except BaseAgentError as e:
    print(f"Agent error: {e}")

# ── 5. Lifetime vs per-run usage tracking ─────────────────────────────────────
# agent.usage_metrics accumulates across ALL runs on this instance.
# max_cost is checked against only the CURRENT run's metrics.

agent5 = BaseAgent(llm="claude-sonnet-4-20250514")
agent5.max_cost = 0.10  # $0.10 per run

log_a, _ = agent5.run("What is DNA?")
log_b, _ = agent5.run("What is RNA?")  # new run; budget resets

total_cost = sum(u.cost for u in agent5.usage_metrics if u.cost is not None)
print(f"Lifetime cost across both runs: ${total_cost:.4f}")

# ── 6. Configure via environment variables ────────────────────────────────────
# These can also be set before starting your script:
#
#   BASE_AGENT_MAX_ITERATIONS=10
#   BASE_AGENT_MAX_COST=1.00
#   BASE_AGENT_MAX_CONSECUTIVE_ERRORS=3
#
# The default_config singleton picks them up at import time.
