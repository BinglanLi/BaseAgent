"""Tests for the patterns demonstrated in examples 15 (streaming) and 16 (persistence).

These tests use mock LLMs so no API key is required.
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from BaseAgent.events import AgentEvent, EventType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(checkpoint_db_path: str = ":memory:"):
    mock_llm = MagicMock()
    mock_llm.model_name = "mock-model"
    mock_llm.invoke.return_value = MagicMock(content="<solution>done</solution>")
    with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm)):
        from BaseAgent.base_agent import BaseAgent
        return BaseAgent(checkpoint_db_path=checkpoint_db_path, require_approval="never")


async def _collect(agen) -> list[AgentEvent]:
    return [item async for item in agen]


def _make_raw_events(events: list[dict]):
    async def _gen():
        for e in events:
            yield e
    return lambda *args, **kwargs: _gen()


# ---------------------------------------------------------------------------
# Streaming (example 15) patterns
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStreamingExample:
    @pytest.fixture
    def agent(self):
        return _make_agent()

    def _thinking_events(self):
        ai_msg = AIMessage(content="<think>reasoning</think>")
        state = {"input": [HumanMessage(content="task"), ai_msg], "next_step": "generate"}
        return [
            {"event": "on_chain_end", "metadata": {"langgraph_node": "generate"}, "data": {"output": state}},
        ]

    def test_stream_all_events_yields_thinking(self, agent):
        agent.app.astream_events = _make_raw_events(self._thinking_events())
        collected = asyncio.run(_collect(agent.run_stream("task")))
        assert len(collected) == 1
        assert collected[0].event_type == EventType.THINKING

    def test_stream_filter_final_answer_only(self, agent):
        ai_msg = AIMessage(content="<solution>answer</solution>")
        state = {"input": [HumanMessage(content="task"), ai_msg], "next_step": "end"}
        raw = [{"event": "on_chain_end", "metadata": {"langgraph_node": "generate"}, "data": {"output": state}}]
        agent.app.astream_events = _make_raw_events(raw)

        collected = asyncio.run(
            _collect(agent.run_stream("task", event_types={EventType.FINAL_ANSWER}))
        )
        assert len(collected) == 1
        assert collected[0].event_type == EventType.FINAL_ANSWER

    def test_stream_filter_excludes_other_types(self, agent):
        ai_msg = AIMessage(content="<think>thinking</think>")
        state = {"input": [HumanMessage(content="task"), ai_msg], "next_step": "end"}
        raw = [{"event": "on_chain_end", "metadata": {"langgraph_node": "generate"}, "data": {"output": state}}]
        agent.app.astream_events = _make_raw_events(raw)

        collected = asyncio.run(
            _collect(agent.run_stream("task", event_types={EventType.FINAL_ANSWER}))
        )
        assert collected == []

    def test_event_to_json_is_serialisable(self, agent):
        import json
        agent.app.astream_events = _make_raw_events(self._thinking_events())
        collected = asyncio.run(_collect(agent.run_stream("task")))
        assert collected
        payload = collected[0].to_json()
        parsed = json.loads(payload)
        assert parsed["event_type"] == "thinking"

    def test_stream_with_thread_id(self, agent):
        agent.app.astream_events = _make_raw_events([])
        tid = str(uuid.uuid4())
        asyncio.run(_collect(agent.run_stream("task", thread_id=tid)))
        assert agent.thread_id == tid


# ---------------------------------------------------------------------------
# Persistence (example 16) patterns
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPersistenceExample:
    def test_file_checkpoint_path_is_stored(self, tmp_path):
        db = str(tmp_path / "conv.db")
        agent = _make_agent(checkpoint_db_path=db)
        assert agent.checkpoint_db_path == db

    def test_same_thread_id_reused_across_runs(self, tmp_path):
        db = str(tmp_path / "conv.db")
        agent = _make_agent(checkpoint_db_path=db)

        mock_resp = MagicMock()
        mock_resp.content = "<solution>done</solution>"
        agent.llm.invoke.return_value = mock_resp

        tid = "my-session"
        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None):
            agent.run("first question", thread_id=tid)
            assert agent.thread_id == tid
            agent.run("follow-up question", thread_id=tid)
            assert agent.thread_id == tid

    def test_different_thread_ids_produce_separate_histories(self, tmp_path):
        db = str(tmp_path / "conv.db")
        agent = _make_agent(checkpoint_db_path=db)

        mock_resp = MagicMock()
        mock_resp.content = "<solution>done</solution>"
        agent.llm.invoke.return_value = mock_resp

        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None):
            agent.run("question A", thread_id="session-a")
            tid_a = agent.thread_id
            agent.run("question B", thread_id="session-b")
            tid_b = agent.thread_id

        assert tid_a == "session-a"
        assert tid_b == "session-b"
        assert tid_a != tid_b

    def test_close_cleans_up_without_error(self, tmp_path):
        db = str(tmp_path / "conv.db")
        agent = _make_agent(checkpoint_db_path=db)
        agent.close()  # must not raise
