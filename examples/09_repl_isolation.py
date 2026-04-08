"""Example: REPL Namespace Isolation between multiple BaseAgent instances.

Each BaseAgent instance owns:
  - self._repl_namespace  — isolated exec() dict for Python <execute> blocks
  - self._plot_capture    — isolated PlotCapture for matplotlib output

Variables set by one agent are completely invisible to other agents.
"""

from unittest.mock import MagicMock, patch

from BaseAgent import BaseAgent
from BaseAgent.tools.support_tools import run_python_repl


def make_mock_agent(**kwargs):
    """Helper: create a BaseAgent with a mock LLM (no API key required)."""
    mock_llm = MagicMock()
    mock_llm.model_name = "mock-model"
    resp = MagicMock()
    resp.content = "<solution>done</solution>"
    mock_llm.invoke.return_value = resp
    with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm)):
        return BaseAgent(**kwargs)


# ---------------------------------------------------------------------------
# 1. Variables in one agent's namespace are invisible to another
# ---------------------------------------------------------------------------

agent_a = make_mock_agent()
agent_b = make_mock_agent()

run_python_repl("secret = 'agent_a_data'", namespace=agent_a._repl_namespace)
run_python_repl("secret = 'agent_b_data'", namespace=agent_b._repl_namespace)

assert agent_a._repl_namespace["secret"] == "agent_a_data"
assert agent_b._repl_namespace["secret"] == "agent_b_data"

print("agent_a secret:", agent_a._repl_namespace["secret"])
print("agent_b secret:", agent_b._repl_namespace["secret"])


# ---------------------------------------------------------------------------
# 2. Custom tools are injected into the per-instance namespace only
# ---------------------------------------------------------------------------

def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

agent_a.add_tool(add)
agent_a._inject_custom_functions_to_repl()

# 'add' is available in agent_a's namespace but not agent_b's
assert "add" in agent_a._repl_namespace
assert "add" not in agent_b._repl_namespace

out = run_python_repl("print(add(3, 7))", namespace=agent_a._repl_namespace)
print("agent_a REPL output:", out.strip())  # -> 10


# ---------------------------------------------------------------------------
# 3. PlotCapture instances are independent
# ---------------------------------------------------------------------------

assert agent_a._plot_capture is not agent_b._plot_capture

agent_a._plot_capture._plots.append("fake_plot_data")
assert agent_b._plot_capture.get_plots() == []

agent_a._plot_capture.clear()
assert agent_a._plot_capture.get_plots() == []


# ---------------------------------------------------------------------------
# 4. Backward compat: run_python_repl without namespace uses global fallback
# ---------------------------------------------------------------------------

from BaseAgent.tools.support_tools import _persistent_namespace

_persistent_namespace.clear()
run_python_repl("global_var = 42")   # no namespace arg → uses _persistent_namespace
assert _persistent_namespace.get("global_var") == 42
_persistent_namespace.clear()

print("\nAll namespace isolation checks passed.")
