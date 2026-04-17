"""Unit tests for the async API: arun(), aresume(), areject() (Feature 7).

Also covers run() / arun() stream log deduplication (fix for double-HumanMessage
bug caused by LangGraph's __start__ snapshot + retrieve no-op snapshot).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from helpers.node_helpers import make_base_agent

pytestmark = pytest.mark.unit


def make_agent():
    return make_base_agent()


def make_interrupted_agent():
    """Return a BaseAgent in an interrupted state, ready for aresume()/areject()."""
    agent = make_agent()
    agent.app = MagicMock()
    agent.app.astream = MagicMock(return_value=_async_iter([_make_state()]))
    agent.app.aget_state = AsyncMock(return_value=_no_interrupt_graph_state())
    agent._interrupted = True
    agent._run_config = {"configurable": {"thread_id": "t1"}}
    agent.log = []  # normally set by arun(); set here for isolation
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
        agent.usage_metrics = [MagicMock()]
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

        # _setup_run() cleared _interrupted before the stream started, so it
        # stays False even though the stream raised before _post_stream_result ran
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


class TestAresume:
    def test_aresume_returns_result(self):
        agent = make_interrupted_agent()

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
        agent = make_interrupted_agent()
        agent.app.stream = MagicMock()

        asyncio.run(agent.aresume())

        agent.app.astream.assert_called_once()
        agent.app.stream.assert_not_called()

    def test_aresume_allows_retry_after_stream_error(self):
        """A stream crash in aresume() must not clear _interrupted.

        The LangGraph checkpointer still holds the pending interrupt, so the
        user should be able to call aresume() again to retry.
        """
        agent = make_interrupted_agent()

        async def _failing_stream(*args, **kwargs):
            yield _make_state()
            raise RuntimeError("transient error")

        agent.app.astream = MagicMock(return_value=_failing_stream())

        with pytest.raises(RuntimeError, match="transient error"):
            asyncio.run(agent.aresume())

        assert agent._interrupted is True  # still retryable


# ---------------------------------------------------------------------------
# areject() tests
# ---------------------------------------------------------------------------


class TestAreject:
    def test_areject_returns_result(self):
        agent = make_interrupted_agent()

        log, content = asyncio.run(agent.areject("try again"))

        assert isinstance(log, list)
        assert content == "<solution>done</solution>"

    def test_areject_raises_if_not_interrupted(self):
        agent = make_agent()
        agent._interrupted = False

        with pytest.raises(RuntimeError, match="not in an interrupted state"):
            asyncio.run(agent.areject())

    def test_areject_passes_correct_command(self):
        agent = make_interrupted_agent()

        asyncio.run(agent.areject("do it differently"))

        call_args = agent.app.astream.call_args[0]
        command = call_args[0]
        assert isinstance(command, Command)
        assert command.resume == {"approved": False, "feedback": "do it differently"}

    def test_areject_calls_astream_not_stream(self):
        agent = make_interrupted_agent()
        agent.app.stream = MagicMock()

        asyncio.run(agent.areject("nope"))

        agent.app.astream.assert_called_once()
        agent.app.stream.assert_not_called()

    def test_areject_allows_retry_after_stream_error(self):
        """A stream crash in areject() must not clear _interrupted.

        The LangGraph checkpointer still holds the pending interrupt, so the
        user should be able to call areject() again to retry.
        """
        agent = make_interrupted_agent()

        async def _failing_stream(*args, **kwargs):
            yield _make_state()
            raise RuntimeError("transient error")

        agent.app.astream = MagicMock(return_value=_failing_stream())

        with pytest.raises(RuntimeError, match="transient error"):
            asyncio.run(agent.areject("nope"))

        assert agent._interrupted is True  # still retryable


# ---------------------------------------------------------------------------
# Async generator helper
# ---------------------------------------------------------------------------


async def _async_iter(items):
    """Yield items one by one as an async generator."""
    for item in items:
        yield item


def _sync_iter(items):
    return iter(items)


# ---------------------------------------------------------------------------
# Snapshot helpers for stream logging tests
# ---------------------------------------------------------------------------

_HM = HumanMessage(content="solve this")
_AI1 = AIMessage(content="<solution>answer</solution>")
_AI_BAD = AIMessage(content="no tags at all")
_HM_CORRECTION = HumanMessage(content="Each response must include thinking...")
_AI2 = AIMessage(content="<solution>second answer</solution>")


def _snap(*msgs):
    return {"input": list(msgs), "next_step": "end"}


# ---------------------------------------------------------------------------
# Stream log deduplication — run() sync API
# ---------------------------------------------------------------------------


class TestRunStreamLogging:
    def _make_agent_with_stream(self, snapshots):
        agent = make_agent()
        agent.app = MagicMock()
        agent.app.stream = MagicMock(return_value=_sync_iter(snapshots))
        agent.app.get_state = MagicMock(return_value=_no_interrupt_graph_state())
        return agent

    def test_duplicate_snapshot_logged_once(self):
        """__start__ + retrieve both emit [HM] — HM must appear in log exactly once."""
        snapshots = [
            _snap(_HM),
            _snap(_HM),
            _snap(_HM, _AI1),
        ]
        agent = self._make_agent_with_stream(snapshots)

        log, content = agent.run("solve this")

        hm_entries = [e for e in log if "Human" in e]
        assert len(hm_entries) == 1
        assert content == "<solution>answer</solution>"

    def test_multi_message_step_logs_all(self):
        snapshots = [
            _snap(_HM),
            _snap(_HM),
            _snap(_HM, _AI_BAD, _HM_CORRECTION),
            _snap(_HM, _AI_BAD, _HM_CORRECTION, _AI2),
        ]
        agent = self._make_agent_with_stream(snapshots)

        log, _ = agent.run("solve this")

        assert len(log) == 4

    def test_empty_stream_returns_empty_log(self):
        agent = self._make_agent_with_stream([])

        log, content = agent.run("solve this")

        assert log == []
        assert content == ""


# ---------------------------------------------------------------------------
# Stream log deduplication — arun() async API
# ---------------------------------------------------------------------------


class TestArunStreamLogging:
    def _make_agent_with_astream(self, snapshots):
        agent = make_agent()
        agent.app = MagicMock()
        agent.app.astream = MagicMock(return_value=_async_iter(snapshots))
        agent.app.aget_state = AsyncMock(return_value=_no_interrupt_graph_state())
        return agent

    def test_duplicate_snapshot_logged_once(self):
        snapshots = [
            _snap(_HM),
            _snap(_HM),
            _snap(_HM, _AI1),
        ]
        agent = self._make_agent_with_astream(snapshots)

        log, content = asyncio.run(agent.arun("solve this"))

        hm_entries = [e for e in log if "Human" in e]
        assert len(hm_entries) == 1
        assert content == "<solution>answer</solution>"

    def test_multi_message_step_logs_all(self):
        snapshots = [
            _snap(_HM),
            _snap(_HM),
            _snap(_HM, _AI_BAD, _HM_CORRECTION),
            _snap(_HM, _AI_BAD, _HM_CORRECTION, _AI2),
        ]
        agent = self._make_agent_with_astream(snapshots)

        log, _ = asyncio.run(agent.arun("solve this"))

        assert len(log) == 4
