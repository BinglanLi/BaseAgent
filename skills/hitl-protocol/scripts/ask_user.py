"""Canonical ask_user tool for the HITL agent.

Register with the HITL agent before running:

    from skills.hitl_protocol.scripts.ask_user import ask_user
    hilt_agent.add_tool(ask_user)
"""


def ask_user(message: str) -> str:
    """Present a summary to the human user and collect approval or feedback."""
    prompt = f"\n{'─' * 60}\n{message}\n{'─' * 60}\nPress Enter to approve, or type feedback: "
    response = input(prompt).strip()
    result = response if response else "approved"
    print(f"<solution>{result}</solution>")
    return result
