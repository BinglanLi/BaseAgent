"""Unit tests for stream log deduplication (fix for double-HumanMessage bug).

LangGraph's stream_mode="values" emits one state snapshot per step, including
a __start__ snapshot before any node runs. Combined with the retrieve node
(which returns state unchanged when there is nothing to retrieve), this caused
the initial HumanMessage to be logged twice.

Fix: log only messages that are new since the previous snapshot by tracking
_prev_len across the streaming loop. Each loop logs msgs[_prev_len:] and then
updates _prev_len = len(msgs).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command


# ---------------------------------------------------------------------------
# Helpers shared with test_async_api.py, duplicated here for isolation
# ---------------------------------------------------------------------------


def make_agent():
    mock_llm = MagicMock()
    mock_llm.model_name = "mock-model"
    mock_llm.invoke.return_value = MagicMock(
        content="<solution>done</solution>", usage_metadata=None
    )
    with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm)):
        from BaseAgent.base_agent import BaseAgent

        agent = BaseAgent()
    agent.max_context_messages = None
    agent.max_iterations = None
    agent.max_cost = None
    agent.max_consecutive_errors = None
    agent._usage_metrics = []
    agent._run_usage_start = 0
    return agent


def _no_interrupt_graph_state():
    gs = MagicMock()
    gs.tasks = []
    return gs


async def _async_iter(items):
    for item in items:
        yield item


def _sync_iter(items):
    """Simulate app.stream() as a plain iterable."""
    return iter(items)


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

HM = HumanMessage(content="solve this")
AI1 = AIMessage(content="<solution>answer</solution>")
AI_BAD = AIMessage(content="no tags at all")
HM_CORRECTION = HumanMessage(content="Each response must include thinking...")
AI2 = AIMessage(content="<solution>second answer</solution>")


def _snap(*msgs):
    """Build a minimal LangGraph state snapshot."""
    return {"input": list(msgs), "next_step": "end"}


# ---------------------------------------------------------------------------
# run() — sync API
# ---------------------------------------------------------------------------


@pytest.mark.unit
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
            _snap(HM),        # __start__ snapshot
            _snap(HM),        # retrieve no-op snapshot (identical)
            _snap(HM, AI1),   # generate snapshot
        ]
        agent = self._make_agent_with_stream(snapshots)

        log, content = agent.run("solve this")

        hm_entries = [e for e in log if "Human" in e]
        assert len(hm_entries) == 1, f"Expected 1 HumanMessage log entry, got {len(hm_entries)}"
        assert content == "<solution>answer</solution>"

    def test_multi_message_step_logs_all(self):
        """When generate appends two messages in one step, both must be logged."""
        snapshots = [
            _snap(HM),                      # __start__
            _snap(HM),                      # retrieve no-op
            _snap(HM, AI_BAD, HM_CORRECTION),  # generate: parse error → 2 appends
            _snap(HM, AI_BAD, HM_CORRECTION, AI2),  # generate retry
        ]
        agent = self._make_agent_with_stream(snapshots)

        log, _ = agent.run("solve this")

        assert len(log) == 4, f"Expected 4 log entries, got {len(log)}"

    def test_empty_stream_returns_empty_log(self):
        agent = self._make_agent_with_stream([])

        log, content = agent.run("solve this")

        assert log == []
        assert content == ""


# ---------------------------------------------------------------------------
# arun() — async API
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestArunStreamLogging:
    def _make_agent_with_astream(self, snapshots):
        agent = make_agent()
        agent.app = MagicMock()
        agent.app.astream = MagicMock(return_value=_async_iter(snapshots))
        agent.app.aget_state = AsyncMock(return_value=_no_interrupt_graph_state())
        return agent

    def test_duplicate_snapshot_logged_once(self):
        snapshots = [
            _snap(HM),
            _snap(HM),
            _snap(HM, AI1),
        ]
        agent = self._make_agent_with_astream(snapshots)

        log, content = asyncio.run(agent.arun("solve this"))

        hm_entries = [e for e in log if "Human" in e]
        assert len(hm_entries) == 1
        assert content == "<solution>answer</solution>"

    def test_multi_message_step_logs_all(self):
        snapshots = [
            _snap(HM),
            _snap(HM),
            _snap(HM, AI_BAD, HM_CORRECTION),
            _snap(HM, AI_BAD, HM_CORRECTION, AI2),
        ]
        agent = self._make_agent_with_astream(snapshots)

        log, _ = asyncio.run(agent.arun("solve this"))

        assert len(log) == 4


# ---------------------------------------------------------------------------
# resume() / aresume() — must not re-log messages from the prior run()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResumeStreamLogging:
    def _interrupted_graph_state(self, payload):
        task = MagicMock()
        task.interrupts = [MagicMock(value=payload)]
        gs = MagicMock()
        gs.tasks = [task]
        return gs

    def _setup_interrupted_agent(self):
        """Return an agent whose run() already logged [HM, AI_PENDING]."""
        agent = make_agent()

        AI_PENDING = AIMessage(content="<execute>print('hi')</execute>")
        prior_state = _snap(HM, AI_PENDING)
        payload = {"code": "print('hi')", "language": "python", "message": "Approve?"}

        # Simulate a completed run() that interrupted
        agent.app = MagicMock()
        agent.app.stream = MagicMock(return_value=_sync_iter([
            _snap(HM),
            _snap(HM),
            _snap(HM, AI_PENDING),
        ]))
        agent.app.get_state = MagicMock(return_value=self._interrupted_graph_state(payload))
        agent.run("solve this")  # populates self.log and self._conversation_state

        assert agent.is_interrupted
        assert len(agent.log) == 2   # HM + AI_PENDING

        # Now wire up the resume stream
        AI_OBS = AIMessage(content="<observation>hi</observation>")
        AI_FINAL = AIMessage(content="<solution>done</solution>")
        resume_snapshots = [
            _snap(HM, AI_PENDING),           # replayed interrupt-point state
            _snap(HM, AI_PENDING),           # approval_gate unchanged
            _snap(HM, AI_PENDING, AI_OBS),   # after execute
            _snap(HM, AI_PENDING, AI_OBS, AI_FINAL),  # after generate
        ]
        agent.app.stream = MagicMock(return_value=_sync_iter(resume_snapshots))
        agent.app.get_state = MagicMock(return_value=_no_interrupt_graph_state())
        return agent

    def test_resume_logs_only_new_messages(self):
        agent = self._setup_interrupted_agent()
        pre_resume_count = len(agent.log)  # 2: HM + AI_PENDING

        agent.resume()

        new_entries = agent.log[pre_resume_count:]
        # Should have logged: AI_OBS + AI_FINAL (2 new messages)
        assert len(new_entries) == 2, (
            f"Expected 2 new log entries after resume, got {len(new_entries)}"
        )

    def test_aresume_logs_only_new_messages(self):
        agent = make_agent()

        AI_PENDING = AIMessage(content="<execute>print('hi')</execute>")
        payload = {"code": "print('hi')", "language": "python", "message": "Approve?"}

        # Simulate prior arun() that interrupted
        agent.app = MagicMock()
        agent.app.astream = MagicMock(return_value=_async_iter([
            _snap(HM),
            _snap(HM),
            _snap(HM, AI_PENDING),
        ]))
        agent.app.aget_state = AsyncMock(return_value=MagicMock(
            tasks=[MagicMock(interrupts=[MagicMock(value=payload)])]
        ))
        asyncio.run(agent.arun("solve this"))
        assert agent.is_interrupted
        pre_resume_count = len(agent.log)

        # Wire up the aresume stream
        AI_OBS = AIMessage(content="<observation>hi</observation>")
        AI_FINAL = AIMessage(content="<solution>done</solution>")
        agent.app.astream = MagicMock(return_value=_async_iter([
            _snap(HM, AI_PENDING),
            _snap(HM, AI_PENDING),
            _snap(HM, AI_PENDING, AI_OBS),
            _snap(HM, AI_PENDING, AI_OBS, AI_FINAL),
        ]))
        agent.app.aget_state = AsyncMock(return_value=_no_interrupt_graph_state())

        asyncio.run(agent.aresume())

        new_entries = agent.log[pre_resume_count:]
        assert len(new_entries) == 2
