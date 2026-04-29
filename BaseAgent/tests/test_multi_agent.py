"""Unit tests for Feature 8: AgentTeam multi-agent orchestration."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END

from BaseAgent.agent_spec import AgentSpec
from BaseAgent.errors import BaseAgentError, MaxRoundsExceededError
from BaseAgent.multi_agent.orchestrator import AgentTeam, SupervisorDecision
from BaseAgent.multi_agent.state import MultiAgentState

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_mock_agent(name: str, role: str = "test agent"):
    """Return a BaseAgent wired with a mock LLM, ready for use in AgentTeam."""
    mock_llm = MagicMock()
    mock_llm.model_name = "mock-model"
    with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm)):
        from BaseAgent.base_agent import BaseAgent

        agent = BaseAgent(
            spec=AgentSpec(name=name, role=role),
            require_approval="never",
        )
    agent.arun = AsyncMock(return_value=([], f"result from {name}"))
    return agent


def make_team(*names: str) -> AgentTeam:
    """Return an AgentTeam with mock agents and a mock supervisor LLM."""
    agents = [make_mock_agent(n) for n in names]
    mock_supervisor = MagicMock()
    # Default: supervisor immediately returns FINISH
    mock_supervisor.with_structured_output.return_value.invoke.return_value = SupervisorDecision(
        next_agent="FINISH", sub_task=""
    )
    with patch("BaseAgent.multi_agent.orchestrator.get_llm", return_value=("Anthropic", mock_supervisor)):
        team = AgentTeam(agents, supervisor_llm="mock-model")
    return team


def _make_state(
    next_agent: str = "FINISH",
    sub_task: str = "",
    task: str = "test task",
    results: dict | None = None,
    round: int = 0,
) -> MultiAgentState:
    return {
        "messages": [HumanMessage(content=task)],
        "next_agent": next_agent,
        "sub_task": sub_task,
        "task": task,
        "results": results or {},
        "round": round,
    }


# ---------------------------------------------------------------------------
# 1. __init__ validation
# ---------------------------------------------------------------------------


class TestInitValidation:
    def test_duplicate_names_raise(self):
        a = make_mock_agent("agent_a")
        b = make_mock_agent("agent_a")
        mock_llm = MagicMock()
        with patch("BaseAgent.multi_agent.orchestrator.get_llm", return_value=("Anthropic", mock_llm)):
            with pytest.raises(ValueError, match="Duplicate name"):
                AgentTeam([a, b])

    def test_require_approval_not_never_raises(self):
        mock_llm_agent = MagicMock()
        mock_llm_agent.model_name = "mock-model"
        with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm_agent)):
            from BaseAgent.base_agent import BaseAgent

            agent = BaseAgent(
                spec=AgentSpec(name="x", role="r"),
                require_approval="always",  # wrong — should fail
            )
        mock_supervisor = MagicMock()
        with patch("BaseAgent.multi_agent.orchestrator.get_llm", return_value=("Anthropic", mock_supervisor)):
            with pytest.raises(ValueError, match="require_approval='never'"):
                AgentTeam([agent])


# ---------------------------------------------------------------------------
# 2. Routing
# ---------------------------------------------------------------------------


class TestRouting:
    def test_finish_routes_to_end(self):
        team = make_team("alpha", "beta")
        state = _make_state(next_agent="FINISH")
        assert team._route(state) == END

    def test_valid_name_routes_to_node(self):
        team = make_team("alpha", "beta")
        state = _make_state(next_agent="alpha")
        assert team._route(state) == "alpha"

    def test_unknown_name_routes_to_end_with_warning(self, caplog):
        team = make_team("alpha")
        state = _make_state(next_agent="ghost_agent")
        with caplog.at_level(logging.WARNING, logger="BaseAgent.multi_agent.orchestrator"):
            result = team._route(state)
        assert result == END
        assert "ghost_agent" in caplog.text


# ---------------------------------------------------------------------------
# 3. Agent node
# ---------------------------------------------------------------------------


class TestAgentNode:
    def test_writes_result_to_results(self):
        team = make_team("alpha")
        agent = team._agent_map["alpha"]
        agent.arun = AsyncMock(return_value=([], "the answer"))
        node_fn = team._make_agent_node(agent)
        state = _make_state(sub_task="do the thing")

        result = asyncio.run(node_fn(state))

        assert result["results"]["alpha"] == "the answer"
        assert any(
            isinstance(m, AIMessage) and m.name == "alpha"
            for m in result["messages"]
        )

    def test_reraises_base_agent_error(self):
        """Infrastructure errors surface to the caller rather than being swallowed."""
        team = make_team("alpha")
        agent = team._agent_map["alpha"]
        agent.arun = AsyncMock(side_effect=BaseAgentError("boom"))
        node_fn = team._make_agent_node(agent)
        state = _make_state(sub_task="do the thing")

        with pytest.raises(BaseAgentError, match="boom"):
            asyncio.run(node_fn(state))


# ---------------------------------------------------------------------------
# 4. Supervisor node
# ---------------------------------------------------------------------------


class TestSupervisorNode:
    def test_sets_next_agent_and_sub_task(self):
        team = make_team("alpha", "beta")
        team._supervisor_llm.with_structured_output.return_value.invoke.return_value = (
            SupervisorDecision(next_agent="alpha", sub_task="analyse the data")
        )
        state = _make_state(round=0)

        result = team._supervisor_node(state)

        assert result["next_agent"] == "alpha"
        assert result["sub_task"] == "analyse the data"
        assert result["round"] == 1

    def test_raises_max_rounds_exceeded(self):
        team = make_team("alpha")
        team.max_rounds = 2
        state = _make_state(round=2)  # round will be incremented to 3 > max_rounds=2

        with pytest.raises(MaxRoundsExceededError, match="max_rounds=2"):
            team._supervisor_node(state)

    def test_supervisor_prompt_contains_clean_result(self):
        """Supervisor receives stripped solution text, not raw XML tags."""
        team = make_team("analyst", "writer")
        team._supervisor_llm.with_structured_output.return_value.invoke.return_value = (
            SupervisorDecision(next_agent="writer", sub_task="write it")
        )
        state = _make_state(
            results={"analyst": "<think>reasoning</think><solution>the findings</solution>"},
            round=0,
        )
        team._supervisor_node(state)

        call_args = team._supervisor_llm.with_structured_output.return_value.invoke.call_args
        prompt_text = call_args[0][0][0].content  # HumanMessage.content
        assert "the findings" in prompt_text
        assert "<think>" not in prompt_text


# ---------------------------------------------------------------------------
# 5. End-to-end
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_one_round_returns_correct_result(self):
        team = make_team("alpha")
        call_count = 0

        def _supervisor_side_effect(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return SupervisorDecision(next_agent="alpha", sub_task="do it")
            return SupervisorDecision(next_agent="FINISH", sub_task="")

        team._supervisor_llm.with_structured_output.return_value.invoke.side_effect = (
            _supervisor_side_effect
        )
        team._agent_map["alpha"].arun = AsyncMock(return_value=([], "final answer"))

        log, result = team.run_sync("the task")

        assert result == "final answer"
        assert log == []

    def test_two_round_loop_routes_both_agents(self):
        team = make_team("agent_a", "agent_b")
        call_order = []

        def _supervisor_side_effect(messages):
            if not call_order:
                call_order.append("sup1")
                return SupervisorDecision(next_agent="agent_a", sub_task="first step")
            elif len(call_order) == 1:
                call_order.append("sup2")
                return SupervisorDecision(next_agent="agent_b", sub_task="second step")
            else:
                return SupervisorDecision(next_agent="FINISH", sub_task="")

        team._supervisor_llm.with_structured_output.return_value.invoke.side_effect = (
            _supervisor_side_effect
        )
        team._agent_map["agent_a"].arun = AsyncMock(return_value=([], "output A"))
        team._agent_map["agent_b"].arun = AsyncMock(return_value=([], "output B"))

        log, result = team.run_sync("two-step task")

        assert result == "output B"
        team_tid = team._current_thread_id
        team._agent_map["agent_a"].arun.assert_called_once_with(
            "first step", thread_id=f"{team_tid}:agent_a"
        )
        team._agent_map["agent_b"].arun.assert_called_once_with(
            "second step", thread_id=f"{team_tid}:agent_b"
        )
