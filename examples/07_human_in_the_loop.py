"""
Human-in-the-Loop Example
==========================

Demonstrates the three approval policies:

  "never"           — default, agent runs without pausing (skip this file's topic)
  "always"          — pause before every code block
  "dangerous_only"  — pause only for bash / R code; Python runs freely

After agent.run() returns, check agent.is_interrupted:
  True  → a code block is waiting for review; call resume() or reject(feedback)
  False → the agent finished normally; the second return value is the final answer
"""

from BaseAgent import BaseAgent

# ---------------------------------------------------------------------------
# 1. "always" policy — approve a code block
# ---------------------------------------------------------------------------

agent = BaseAgent(require_approval="always")

log, payload = agent.run("Compute the sum of 1 through 10 in Python.")

if agent.is_interrupted:
    print("=== Code pending approval ===")
    print(f"Language : {payload['language']}")
    print(f"Code     :\n{payload['code']}")
    print()

    # Approve — agent continues and returns the final answer
    log, answer = agent.resume()
    print("Final answer:", answer)

# ---------------------------------------------------------------------------
# 2. "always" policy — reject a code block and let the agent try again
# ---------------------------------------------------------------------------

agent2 = BaseAgent(require_approval="always")

log, payload = agent2.run("List the files in /tmp using Python.")

if agent2.is_interrupted:
    print("\n=== Code pending approval ===")
    print(payload["code"])

    # Reject with feedback — agent regenerates
    log, payload2 = agent2.reject("Do not use the os module. Use pathlib instead.")

    if agent2.is_interrupted:
        # Agent produced a revised code block — approve it
        print("\n=== Revised code ===")
        print(payload2["code"])
        log, answer = agent2.resume()
        print("Final answer:", answer)
    else:
        print("Final answer:", payload2)

# ---------------------------------------------------------------------------
# 3. "dangerous_only" policy — only bash / R triggers the gate
# ---------------------------------------------------------------------------

agent3 = BaseAgent(require_approval="dangerous_only")

# Python code runs without interruption
log, answer = agent3.run("Calculate 2 ** 10 in Python.")
assert not agent3.is_interrupted
print("\n[dangerous_only] Python ran freely:", answer)

# Bash code pauses for review
log, payload = agent3.run("Print the current working directory using bash.")

if agent3.is_interrupted:
    print("\n[dangerous_only] Bash code paused for review:")
    print(payload["code"])
    log, answer = agent3.resume()
    print("Result:", answer)
