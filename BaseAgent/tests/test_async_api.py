"""Unit tests for the async API: arun(), aresume(), areject() (Feature 7)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_agent():
    """Return a BaseAgent wired with mock LLM and app."""
    mock_llm = MagicMock()
    mock_llm.model_name = "mock-model"
    mock_llm.invoke.return_value = MagicMock(
        content="<solution>done</solution>", usage_metadata=None
    )
    with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm)):
        from BaseAgent.base_agent import BaseAgent

        agent = BaseAgent()
    # Disable termination guards that require real attributes on MagicMock
    agent.max_context_messages = None
    agent.max_iterations = None
    agent.max_cost = None
    agent.max_consecutive_errors = None
    agent._usage_metrics = []
    agent._run_usage_start = 0
    return agent


def _make_state(content: str = "<solution>done</solution>"):
    """One LangGraph state snapshot with a single AIMessage."""
    return {"input": [AIMessage(content=content)], "next_step": "end"}


def _no_interrupt_graph_state():
    """Mock graph state with no pending interrupts."""
    gs = MagicMock()
    gs.tasks = []
    return gs


def _interrupted_graph_state(payload: dict):
    """Mock graph state with one pending interrupt task."""
    task = MagicMock()
    task.interrupts = [MagicMock(value=payload)]
    gs = MagicMock()
    gs.tasks = [task]
    return gs


# ---------------------------------------------------------------------------
# arun() tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestArun:
    def test_arun_returns_log_and_content(self):
        agent = make_agent()
        agent.app = MagicMock()
        agent.app.astream = MagicMock(return_value=_async_iter([_make_state()]))
        agent.app.aget_state = AsyncMock(return_value=_no_interrupt_graph_state())

        log, content = asyncio.run(agent.arun("test task"))

        assert isinstance(log, list)
        assert isinstance(content, str)
        assert content == "<solution>done</solution>"

    def test_arun_with_thread_id(self):
        agent = make_agent()
        agent.app = MagicMock()
        agent.app.astream = MagicMock(return_value=_async_iter([_make_state()]))
        agent.app.aget_state = AsyncMock(return_value=_no_interrupt_graph_state())

        asyncio.run(agent.arun("test task", thread_id="my-thread"))

        assert agent.thread_id == "my-thread"
        call_config = agent.app.astream.call_args[1]["config"]
        assert call_config["configurable"]["thread_id"] == "my-thread"

    def test_arun_interrupted(self):
        agent = make_agent()
        payload = {"code": "print('hi')", "language": "python", "message": "Approve?"}
        agent.app = MagicMock()
        agent.app.astream = MagicMock(return_value=_async_iter([_make_state()]))
        agent.app.aget_state = AsyncMock(
            return_value=_interrupted_graph_state(payload)
        )

        log, result = asyncio.run(agent.arun("test task"))

        assert agent.is_interrupted is True
        assert result == payload

    def test_arun_resets_counters_and_log(self):
        agent = make_agent()
        agent.app = MagicMock()
        agent.app.astream = MagicMock(return_value=_async_iter([_make_state()]))
        agent.app.aget_state = AsyncMock(return_value=_no_interrupt_graph_state())
        # Pre-seed dirty state
        agent.node_executor._iteration_count = 5
        agent.node_executor._consecutive_error_count = 3
        agent._usage_metrics = [MagicMock()]
        agent._run_usage_start = 99

        asyncio.run(agent.arun("test task"))

        assert agent.node_executor._iteration_count == 0
        assert agent.node_executor._consecutive_error_count == 0
        assert agent._run_usage_start == 1  # len of pre-seeded metrics list

    def test_arun_empty_stream(self):
        agent = make_agent()
        agent.app = MagicMock()
        agent.app.astream = MagicMock(return_value=_async_iter([]))
        agent.app.aget_state = AsyncMock(return_value=_no_interrupt_graph_state())

        log, content = asyncio.run(agent.arun("test task"))

        assert log == []
        assert content == ""

    def test_arun_propagates_stream_error(self):
        agent = make_agent()
        agent._interrupted = True  # pre-set stale state

        async def _failing_stream(*args, **kwargs):
            yield _make_state()
            raise RuntimeError("stream exploded")

        agent.app = MagicMock()
        agent.app.astream = MagicMock(return_value=_failing_stream())

        with pytest.raises(RuntimeError, match="stream exploded"):
            asyncio.run(agent.arun("test task"))

        # stale interrupt must be cleared on error
        assert agent._interrupted is False

    def test_arun_calls_astream_not_stream(self):
        agent = make_agent()
        agent.app = MagicMock()
        agent.app.astream = MagicMock(return_value=_async_iter([_make_state()]))
        agent.app.aget_state = AsyncMock(return_value=_no_interrupt_graph_state())
        agent.app.stream = MagicMock()  # must NOT be called

        asyncio.run(agent.arun("test task"))

        agent.app.astream.assert_called_once()
        agent.app.stream.assert_not_called()


# ---------------------------------------------------------------------------
# aresume() tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAresume:
    def _interrupted_agent(self):
        agent = make_agent()
        agent.app = MagicMock()
        agent.app.astream = MagicMock(return_value=_async_iter([_make_state()]))
        agent.app.aget_state = AsyncMock(return_value=_no_interrupt_graph_state())
        agent._interrupted = True
        agent._run_config = {"configurable": {"thread_id": "t1"}}
        agent.log = []  # normally set by arun(); set here for isolation
        return agent

    def test_aresume_returns_result(self):
        agent = self._interrupted_agent()

        log, content = asyncio.run(agent.aresume())

        assert isinstance(log, list)
        assert content == "<solution>done</solution>"
        assert agent.is_interrupted is False

    def test_aresume_raises_if_not_interrupted(self):
        agent = make_agent()
        agent._interrupted = False

        with pytest.raises(RuntimeError, match="not in an interrupted state"):
            asyncio.run(agent.aresume())

    def test_aresume_calls_astream_not_stream(self):
        agent = self._interrupted_agent()
        agent.app.stream = MagicMock()

        asyncio.run(agent.aresume())

        agent.app.astream.assert_called_once()
        agent.app.stream.assert_not_called()


# ---------------------------------------------------------------------------
# areject() tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAreject:
    def _interrupted_agent(self):
        agent = make_agent()
        agent.app = MagicMock()
        agent.app.astream = MagicMock(return_value=_async_iter([_make_state()]))
        agent.app.aget_state = AsyncMock(return_value=_no_interrupt_graph_state())
        agent._interrupted = True
        agent._run_config = {"configurable": {"thread_id": "t1"}}
        agent.log = []  # normally set by arun(); set here for isolation
        return agent

    def test_areject_returns_result(self):
        agent = self._interrupted_agent()

        log, content = asyncio.run(agent.areject("try again"))

        assert isinstance(log, list)
        assert content == "<solution>done</solution>"

    def test_areject_raises_if_not_interrupted(self):
        agent = make_agent()
        agent._interrupted = False

        with pytest.raises(RuntimeError, match="not in an interrupted state"):
            asyncio.run(agent.areject())

    def test_areject_passes_correct_command(self):
        agent = self._interrupted_agent()

        asyncio.run(agent.areject("do it differently"))

        call_args = agent.app.astream.call_args[0]
        command = call_args[0]
        assert isinstance(command, Command)
        assert command.resume == {"approved": False, "feedback": "do it differently"}

    def test_areject_calls_astream_not_stream(self):
        agent = self._interrupted_agent()
        agent.app.stream = MagicMock()

        asyncio.run(agent.areject("nope"))

        agent.app.astream.assert_called_once()
        agent.app.stream.assert_not_called()


# ---------------------------------------------------------------------------
# Async generator helper
# ---------------------------------------------------------------------------


async def _async_iter(items):
    """Yield items one by one as an async generator."""
    for item in items:
        yield item
