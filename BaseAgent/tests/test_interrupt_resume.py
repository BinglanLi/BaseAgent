"""Unit tests for interrupt/resume and selective approval — Phases 2 & 3."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from BaseAgent.nodes import NodeExecutor
from BaseAgent.state import AgentState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_agent(require_approval="never", source="Anthropic"):
    agent = MagicMock()
    agent.source = source
    agent.system_prompt = "You are a helpful assistant."
    agent.use_tool_retriever = False
    agent.timeout_seconds = 30
    agent.critic_count = 0
    agent.user_task = "test task"
    agent.require_approval = require_approval
    agent.max_context_messages = None
    agent.max_iterations = None
    agent.max_cost = None
    agent.max_consecutive_errors = None
    agent._usage_metrics = []
    agent._run_usage_start = 0
    resp = MagicMock()
    resp.content = "<solution>answer</solution>"
    resp.usage_metadata = None
    agent.llm.invoke.return_value = resp
    return agent


def make_state(messages=None, pending_code=None, pending_language=None, next_step=None):
    return {
        "input": messages or [HumanMessage(content="test")],
        "next_step": next_step,
        "pending_code": pending_code,
        "pending_language": pending_language,
    }


# ---------------------------------------------------------------------------
# approval_gate node — unit tests
# ---------------------------------------------------------------------------


class TestApprovalGate:
    def test_approved_true_passes_state_through(self):
        agent = make_agent()
        executor = NodeExecutor(agent)
        state = make_state(
            pending_code="print('hi')",
            pending_language="python",
            next_step="execute",
        )

        with patch("BaseAgent.nodes.interrupt", return_value=True):
            result = executor.approval_gate(state)

        # State unchanged, next_step still "execute"
        assert result["next_step"] == "execute"
        assert result["pending_code"] == "print('hi')"

    def test_approved_dict_passes_state_through(self):
        agent = make_agent()
        executor = NodeExecutor(agent)
        state = make_state(pending_code="x=1", pending_language="python", next_step="execute")

        with patch("BaseAgent.nodes.interrupt", return_value={"approved": True}):
            result = executor.approval_gate(state)

        assert result["next_step"] == "execute"

    def test_rejected_injects_feedback_and_reroutes(self):
        agent = make_agent()
        executor = NodeExecutor(agent)
        state = make_state(pending_code="rm -rf /", pending_language="bash", next_step="execute")

        with patch("BaseAgent.nodes.interrupt", return_value={"approved": False, "feedback": "Too dangerous"}):
            result = executor.approval_gate(state)

        assert result["next_step"] == "generate"
        assert result["pending_code"] is None
        assert result["pending_language"] is None
        last_msg = result["input"][-1]
        assert isinstance(last_msg, HumanMessage)
        assert "Too dangerous" in last_msg.content

    def test_rejected_false_uses_default_feedback(self):
        agent = make_agent()
        executor = NodeExecutor(agent)
        state = make_state(pending_code="code", pending_language="bash", next_step="execute")

        with patch("BaseAgent.nodes.interrupt", return_value=False):
            result = executor.approval_gate(state)

        assert result["next_step"] == "generate"
        last_msg = result["input"][-1]
        assert isinstance(last_msg, HumanMessage)
        assert "rejected" in last_msg.content.lower()

    def test_interrupt_called_with_correct_payload(self):
        agent = make_agent()
        executor = NodeExecutor(agent)
        state = make_state(pending_code="echo hi", pending_language="bash", next_step="execute")

        with patch("BaseAgent.nodes.interrupt", return_value=True) as mock_interrupt:
            executor.approval_gate(state)

        mock_interrupt.assert_called_once()
        payload = mock_interrupt.call_args[0][0]
        assert payload["code"] == "echo hi"
        assert payload["language"] == "bash"
        assert "message" in payload



# ---------------------------------------------------------------------------
# generate() — pending_code population
# ---------------------------------------------------------------------------


class TestGeneratePendingCode:
    def _make_llm_response(self, content):
        resp = MagicMock()
        resp.content = content
        resp.usage_metadata = None
        return resp

    def test_execute_tag_populates_pending_code(self):
        agent = make_agent()
        agent.llm.invoke.return_value = self._make_llm_response(
            "<think>thinking</think><execute>print('hi')</execute>"
        )
        executor = NodeExecutor(agent)
        state = make_state()
        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None):
            result = executor.generate(state)
        assert result["pending_code"] == "print('hi')"
        assert result["pending_language"] is not None

    def test_solution_tag_clears_pending_code(self):
        agent = make_agent()
        agent.llm.invoke.return_value = self._make_llm_response("<solution>done</solution>")
        executor = NodeExecutor(agent)
        state = make_state(pending_code="old code", pending_language="python")
        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None):
            result = executor.generate(state)
        assert result["pending_code"] is None
        assert result["pending_language"] is None

    def test_think_tag_clears_pending_code(self):
        agent = make_agent()
        agent.llm.invoke.return_value = self._make_llm_response("<think>still thinking</think>")
        executor = NodeExecutor(agent)
        state = make_state(pending_code="old code", pending_language="bash")
        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None):
            result = executor.generate(state)
        assert result["pending_code"] is None
        assert result["pending_language"] is None

    def test_no_tags_clears_pending_code(self):
        agent = make_agent()
        agent.llm.invoke.return_value = self._make_llm_response("no tags at all")
        executor = NodeExecutor(agent)
        state = make_state(pending_code="old code", pending_language="python")
        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None):
            result = executor.generate(state)
        assert result["pending_code"] is None
        assert result["pending_language"] is None


# ---------------------------------------------------------------------------
# execute() — pending_code cleared after execution
# ---------------------------------------------------------------------------


class TestExecuteClearsPendingCode:
    def test_execute_clears_pending_fields(self):
        agent = make_agent()
        agent._execution_results = []
        msg = AIMessage(content="<execute>print('hi')</execute>")
        state = make_state(
            messages=[msg],
            pending_code="print('hi')",
            pending_language="python",
        )

        with patch("BaseAgent.nodes.run_with_timeout", return_value="hi"):
            executor = NodeExecutor(agent)
            result = executor.execute(state)

        assert result["pending_code"] is None
        assert result["pending_language"] is None

    def test_execute_clears_pending_even_when_no_execute_tag(self):
        agent = make_agent()
        state = make_state(
            messages=[AIMessage(content="no tags")],
            pending_code="some code",
            pending_language="bash",
        )

        executor = NodeExecutor(agent)
        result = executor.execute(state)

        assert result["pending_code"] is None
        assert result["pending_language"] is None


# ---------------------------------------------------------------------------
# is_interrupted property
# ---------------------------------------------------------------------------


class TestIsInterrupted:
    def test_not_interrupted_by_default(self, base_agent):
        assert not base_agent.is_interrupted

    def test_is_interrupted_false_after_normal_run(self, base_agent):
        mock_resp = MagicMock()
        mock_resp.content = "<solution>done</solution>"
        base_agent.llm.invoke.return_value = mock_resp

        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None):
            base_agent.run("test prompt")

        assert not base_agent.is_interrupted

    def test_resume_raises_when_not_interrupted(self, base_agent):
        assert not base_agent.is_interrupted
        with pytest.raises(RuntimeError, match="not in an interrupted state"):
            base_agent.resume()

    def test_reject_raises_when_not_interrupted(self, base_agent):
        assert not base_agent.is_interrupted
        with pytest.raises(RuntimeError, match="not in an interrupted state"):
            base_agent.reject()


# ---------------------------------------------------------------------------
# EventType.APPROVAL_REQUIRED
# ---------------------------------------------------------------------------


class TestApprovalRequiredEvent:
    def test_approval_required_in_event_type(self):
        from BaseAgent.events import EventType
        assert hasattr(EventType, "APPROVAL_REQUIRED")
        assert EventType.APPROVAL_REQUIRED.value == "approval_required"

    def test_approval_required_serialises(self):
        from BaseAgent.events import AgentEvent, EventType
        event = AgentEvent(
            event_type=EventType.APPROVAL_REQUIRED,
            content="print('hi')",
            node_name="approval_gate",
            metadata={"language": "python"},
        )
        d = event.to_dict()
        assert d["event_type"] == "approval_required"
        assert d["content"] == "print('hi')"
        assert d["metadata"]["language"] == "python"
