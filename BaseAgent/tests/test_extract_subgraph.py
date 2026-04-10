"""
Tests for Feature 4: Extract Subgraph.

Covers:
- get_subgraph() returns an uncompiled StateGraph
- configure() compiles the same graph (app is a CompiledStateGraph)
- get_subgraph() honours self_critic and require_approval topology variations
- configure() behaviour is unchanged (backwards compatibility)
- get_subgraph() can be called independently to embed in a parent graph
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph


def _make_agent(**kwargs):
    """Create a BaseAgent with a mocked LLM (no API key required)."""
    from BaseAgent.base_agent import BaseAgent

    mock_llm = MagicMock()
    mock_llm.model_name = "mock-model"
    mock_response = MagicMock()
    mock_response.content = "Mocked LLM response."
    mock_llm.invoke.return_value = mock_response

    with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm)):
        agent = BaseAgent(**kwargs)
    return agent


# ---------------------------------------------------------------------------
# get_subgraph() return type
# ---------------------------------------------------------------------------

class TestGetSubgraphReturnType:
    """get_subgraph() must return an uncompiled StateGraph."""

    @pytest.mark.unit
    def test_returns_state_graph(self):
        agent = _make_agent()
        graph = agent.get_subgraph()
        assert isinstance(graph, StateGraph)

    @pytest.mark.unit
    def test_not_compiled(self):
        agent = _make_agent()
        graph = agent.get_subgraph()
        # CompiledStateGraph is a subclass of StateGraph in LangGraph; verify it
        # is the raw StateGraph, not the compiled variant.
        assert not isinstance(graph, CompiledStateGraph)

    @pytest.mark.unit
    def test_configure_produces_compiled_graph(self):
        """configure() must compile — agent.app is a CompiledStateGraph."""
        agent = _make_agent()
        assert isinstance(agent.app, CompiledStateGraph)


# ---------------------------------------------------------------------------
# Node topology
# ---------------------------------------------------------------------------

class TestSubgraphTopology:
    """The graph wired by get_subgraph() must contain the expected nodes."""

    def _node_names(self, graph: StateGraph) -> set[str]:
        return set(graph.nodes.keys())

    @pytest.mark.unit
    def test_default_nodes(self):
        agent = _make_agent()
        graph = agent.get_subgraph()
        nodes = self._node_names(graph)
        assert {"retrieve", "generate", "execute", "approval_gate"}.issubset(nodes)
        assert "self_critic" not in nodes

    @pytest.mark.unit
    def test_self_critic_adds_node(self):
        agent = _make_agent()
        graph = agent.get_subgraph(self_critic=True)
        assert "self_critic" in self._node_names(graph)

    @pytest.mark.unit
    def test_require_approval_always_adds_gate(self):
        agent = _make_agent(require_approval="always")
        graph = agent.get_subgraph()
        assert "approval_gate" in self._node_names(graph)

    @pytest.mark.unit
    def test_require_approval_never_no_gate(self):
        agent = _make_agent(require_approval="never")
        graph = agent.get_subgraph()
        assert "approval_gate" not in self._node_names(graph)


# ---------------------------------------------------------------------------
# Side-effects: get_subgraph() sets up state
# ---------------------------------------------------------------------------

class TestSubgraphSideEffects:
    """get_subgraph() must set self.self_critic and self.system_prompt."""

    @pytest.mark.unit
    def test_sets_self_critic_flag(self):
        agent = _make_agent()
        agent.get_subgraph(self_critic=True)
        assert agent.self_critic is True

    @pytest.mark.unit
    def test_generates_system_prompt(self):
        agent = _make_agent()
        # Calling get_subgraph() independently should still produce a system prompt
        agent.system_prompt = ""
        agent.get_subgraph()
        assert len(agent.system_prompt) > 0

    @pytest.mark.unit
    def test_node_executor_set(self):
        agent = _make_agent()
        from BaseAgent.nodes import NodeExecutor
        agent.get_subgraph()
        assert isinstance(agent.node_executor, NodeExecutor)


# ---------------------------------------------------------------------------
# configure() delegates to get_subgraph()
# ---------------------------------------------------------------------------

class TestConfigureDelegates:
    """configure() calls get_subgraph() internally."""

    @pytest.mark.unit
    def test_configure_calls_get_subgraph(self):
        agent = _make_agent()
        with patch.object(agent, "get_subgraph", wraps=agent.get_subgraph) as mock_gs:
            agent.configure()
            mock_gs.assert_called_once()

    @pytest.mark.unit
    def test_configure_passes_args(self):
        agent = _make_agent()
        with patch.object(agent, "get_subgraph", wraps=agent.get_subgraph) as mock_gs:
            agent.configure(self_critic=True, test_time_scale_round=2)
            mock_gs.assert_called_once_with(True, 2)


# ---------------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------------

class TestBackwardsCompat:
    """configure() behaviour must be unchanged after the refactor."""

    @pytest.mark.unit
    def test_app_exists_after_init(self):
        agent = _make_agent()
        assert agent.app is not None

    @pytest.mark.unit
    def test_checkpointer_exists_after_init(self):
        agent = _make_agent()
        assert agent.checkpointer is not None

    @pytest.mark.unit
    def test_system_prompt_non_empty(self):
        agent = _make_agent()
        assert isinstance(agent.system_prompt, str)
        assert len(agent.system_prompt) > 0


# ---------------------------------------------------------------------------
# Independent get_subgraph() call (multi-agent embedding use-case)
# ---------------------------------------------------------------------------

class TestSubgraphEmbedding:
    """get_subgraph() returned graph can be compiled independently."""

    @pytest.mark.unit
    def test_returned_graph_is_compilable(self):
        """The uncompiled graph from get_subgraph() can be compiled standalone."""
        from langgraph.checkpoint.memory import MemorySaver

        agent = _make_agent()
        graph = agent.get_subgraph()
        # Should not raise
        compiled = graph.compile(checkpointer=MemorySaver())
        assert isinstance(compiled, CompiledStateGraph)

    @pytest.mark.unit
    def test_two_agents_independent_subgraphs(self):
        """Two BaseAgent instances produce independent StateGraph objects."""
        agent_a = _make_agent()
        agent_b = _make_agent()
        graph_a = agent_a.get_subgraph()
        graph_b = agent_b.get_subgraph()
        assert graph_a is not graph_b

    @pytest.mark.unit
    def test_no_resource_duplication_on_repeated_calls(self):
        """Calling get_subgraph() multiple times must not duplicate resources."""
        agent = _make_agent()
        tool_count = len(agent.resource_manager.get_all_tools())
        agent.get_subgraph()
        agent.get_subgraph()
        assert len(agent.resource_manager.get_all_tools()) == tool_count
