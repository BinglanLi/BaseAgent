"""
Example 10: Extract Subgraph (Feature 4)

Demonstrates get_subgraph() — returns an uncompiled StateGraph that can be
embedded in a parent LangGraph workflow for multi-agent composition.

Use cases:
1. Inspect the graph topology before running it.
2. Compile with a custom checkpointer (e.g., shared across agents).
3. Embed as a subgraph in a parent StateGraph (post-prototype Option B).
"""

from unittest.mock import MagicMock, patch

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

from BaseAgent import BaseAgent

# ---------------------------------------------------------------------------
# Setup: mock LLM so no API key is required to run this example
# ---------------------------------------------------------------------------
mock_llm = MagicMock()
mock_llm.model_name = "mock-model"
mock_response = MagicMock()
mock_response.content = "<solution>Hello!</solution>"
mock_llm.invoke.return_value = mock_response

# ---------------------------------------------------------------------------
# 1. Basic usage: get an uncompiled StateGraph
# ---------------------------------------------------------------------------
with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm)):
    agent = BaseAgent()

graph: StateGraph = agent.get_subgraph()
print(f"Type: {type(graph).__name__}")          # StateGraph (not compiled)
print(f"Nodes: {list(graph.nodes.keys())}")     # ['retrieve', 'generate', 'execute']

# ---------------------------------------------------------------------------
# 2. Compile with a custom checkpointer
# ---------------------------------------------------------------------------
custom_checkpointer = MemorySaver()
compiled = graph.compile(checkpointer=custom_checkpointer)
print(f"Compiled type: {type(compiled).__name__}")  # CompiledStateGraph

# ---------------------------------------------------------------------------
# 3. Two agents produce independent subgraphs
# ---------------------------------------------------------------------------
with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm)):
    agent_a = BaseAgent()
    agent_b = BaseAgent()

graph_a = agent_a.get_subgraph()
graph_b = agent_b.get_subgraph()
assert graph_a is not graph_b, "Each call produces a fresh StateGraph"
print("graph_a is not graph_b:", graph_a is not graph_b)

# ---------------------------------------------------------------------------
# 4. Topology variants
# ---------------------------------------------------------------------------
with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm)):
    agent_always = BaseAgent(require_approval="always")
    agent_critic = BaseAgent()

graph_always = agent_always.get_subgraph()
graph_critic = agent_critic.get_subgraph(self_critic=True)

print(f"\nWith require_approval='always': {list(graph_always.nodes.keys())}")
print(f"With self_critic=True: {list(graph_critic.nodes.keys())}")

print("\nAll assertions passed.")
