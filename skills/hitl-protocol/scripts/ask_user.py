"""Canonical ask_user tool for the HITL agent.

Register with the HITL agent before running:

    from skills.hitl_protocol.scripts.ask_user import ask_user
    hitl_agent.add_tool(ask_user)

The module must be imported (via add_tool) before any REPL execution so that
_real_stdout captures the real stdout rather than the REPL's StringIO redirect.
"""

import sys

# Capture the real stdout at import time, before the REPL redirects sys.stdout.
_real_stdout = sys.stdout


def ask_user(message: str) -> str:
    """Present a summary to the human user and collect approval or feedback."""
    prompt = f"\n{'─' * 60}\n{message}\n{'─' * 60}\nPress Enter to approve, or type feedback: "
    _real_stdout.write(prompt)
    _real_stdout.flush()
    try:
        response = input("").strip()
    except EOFError:
        response = ""
    result = response if response else "approved"
    print(f"<solution>{result}</solution>")
    return result
