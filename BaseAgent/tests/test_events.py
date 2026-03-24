"""Unit tests for the streaming event architecture (BaseAgent/events.py).

Tests cover:
- AgentEvent creation and serialisation
- BaseAgent._map_langgraph_event() mapping for all node types
- BaseAgent.run_stream() integration with a mocked astream_events
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from BaseAgent.events import AgentEvent, EventType


# ---------------------------------------------------------------------------
# AgentEvent unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAgentEvent:
    def test_creation_defaults(self):
        event = AgentEvent(event_type=EventType.THINKING, content="reasoning text")
        assert event.event_type == EventType.THINKING
        assert event.content == "reasoning text"
        assert event.node_name == ""
        assert event.metadata == {}
        assert event.timestamp  # non-empty

    def test_creation_all_fields(self):
        event = AgentEvent(
            event_type=EventType.CODE_EXECUTING,
            content="print('hi')",
            node_name="execute",
            timestamp="2024-01-01T00:00:00+00:00",
            metadata={"language": "python"},
        )
        assert event.node_name == "execute"
        assert event.metadata == {"language": "python"}
        assert event.timestamp == "2024-01-01T00:00:00+00:00"

    def test_to_dict_event_type_is_string(self):
        event = AgentEvent(event_type=EventType.FINAL_ANSWER, content="42")
        d = event.to_dict()
        assert d["event_type"] == "final_answer"
        assert isinstance(d["event_type"], str)

    def test_to_dict_roundtrip(self):
        event = AgentEvent(
            event_type=EventType.CODE_RESULT,
            content="output",
            node_name="execute",
            metadata={"lines": 10},
        )
        d = event.to_dict()
        assert d["content"] == "output"
        assert d["node_name"] == "execute"
        assert d["metadata"] == {"lines": 10}

    def test_to_json_valid(self):
        event = AgentEvent(event_type=EventType.ERROR, content="oops")
        payload = event.to_json()
        parsed = json.loads(payload)
        assert parsed["event_type"] == "error"
        assert parsed["content"] == "oops"

    @pytest.mark.parametrize("et", list(EventType))
    def test_all_event_types_serialise(self, et):
        event = AgentEvent(event_type=et, content="x")
        d = event.to_dict()
        assert d["event_type"] == et.value
        json.loads(event.to_json())  # must not raise


# ---------------------------------------------------------------------------
# _map_langgraph_event unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMapLangGraphEvent:
    """Tests for BaseAgent._map_langgraph_event()."""

    @pytest.fixture
    def agent(self):
        mock_llm = MagicMock()
        mock_llm.model_name = "mock-model"
        mock_llm.invoke.return_value = MagicMock(content="<solution>done</solution>")
        with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm)):
            from BaseAgent.base_agent import BaseAgent
            return BaseAgent()

    def _make_event(self, event_name: str, node_name: str, data: dict) -> dict:
        return {
            "event": event_name,
            "metadata": {"langgraph_node": node_name},
            "data": data,
        }

    # -- retrieve node -------------------------------------------------------

    def test_retrieve_start(self, agent):
        raw = self._make_event("on_chain_start", "retrieve", {})
        result = agent._map_langgraph_event(raw)
        assert result is not None
        assert result.event_type == EventType.RETRIEVAL_START
        assert result.node_name == "retrieve"

    def test_retrieve_end(self, agent):
        state = {"input": [HumanMessage(content="task")], "next_step": None}
        raw = self._make_event("on_chain_end", "retrieve", {"output": state})
        result = agent._map_langgraph_event(raw)
        assert result is not None
        assert result.event_type == EventType.RETRIEVAL_COMPLETE

    # -- generate node -------------------------------------------------------

    def test_generate_end_thinking(self, agent):
        ai_msg = AIMessage(content="<think>I need to reason</think>")
        state = {"input": [HumanMessage(content="task"), ai_msg], "next_step": "generate"}
        raw = self._make_event("on_chain_end", "generate", {"output": state})
        result = agent._map_langgraph_event(raw)
        assert result is not None
        assert result.event_type == EventType.THINKING
        assert result.content == "I need to reason"

    def test_generate_end_final_answer(self, agent):
        ai_msg = AIMessage(content="<think>done</think><solution>42</solution>")
        state = {"input": [HumanMessage(content="task"), ai_msg], "next_step": "end"}
        raw = self._make_event("on_chain_end", "generate", {"output": state})
        result = agent._map_langgraph_event(raw)
        assert result is not None
        assert result.event_type == EventType.FINAL_ANSWER
        assert result.content == "42"

    def test_generate_end_error(self, agent):
        ai_msg = AIMessage(content="Execution terminated due to repeated parsing errors.")
        state = {"input": [HumanMessage(content="task"), ai_msg], "next_step": "end"}
        raw = self._make_event("on_chain_end", "generate", {"output": state})
        result = agent._map_langgraph_event(raw)
        assert result is not None
        assert result.event_type == EventType.ERROR

    def test_generate_end_execute_tag_no_event(self, agent):
        # When the response contains <execute> but no <think>, no event is emitted
        # from generate — CODE_EXECUTING fires from execute's on_chain_start instead.
        ai_msg = AIMessage(content="<execute>print(1)</execute>")
        state = {"input": [HumanMessage(content="task"), ai_msg], "next_step": "execute"}
        raw = self._make_event("on_chain_end", "generate", {"output": state})
        result = agent._map_langgraph_event(raw)
        assert result is None

    def test_generate_end_think_before_execute(self, agent):
        # When both <think> and <execute> are present, THINKING fires from generate;
        # CODE_EXECUTING fires later from execute's on_chain_start.
        ai_msg = AIMessage(content="<think>plan</think><execute>print(1)</execute>")
        state = {"input": [HumanMessage(content="task"), ai_msg], "next_step": "execute"}
        raw = self._make_event("on_chain_end", "generate", {"output": state})
        result = agent._map_langgraph_event(raw)
        assert result is not None
        assert result.event_type == EventType.THINKING
        assert result.content == "plan"

    # -- execute node --------------------------------------------------------

    def test_execute_start_code_executing(self, agent):
        ai_msg = AIMessage(content="<think>plan</think><execute>x = 1 + 1</execute>")
        state = {"input": [HumanMessage(content="task"), ai_msg], "next_step": "execute"}
        raw = self._make_event("on_chain_start", "execute", {"input": state})
        result = agent._map_langgraph_event(raw)
        assert result is not None
        assert result.event_type == EventType.CODE_EXECUTING
        assert result.content == "x = 1 + 1"

    def test_execute_end_code_result(self, agent):
        obs_msg = AIMessage(content="<observation>2</observation>")
        state = {
            "input": [HumanMessage(content="task"), AIMessage(content="<execute>x=1+1</execute>"), obs_msg],
            "next_step": "generate",
        }
        raw = self._make_event("on_chain_end", "execute", {"output": state})
        result = agent._map_langgraph_event(raw)
        assert result is not None
        assert result.event_type == EventType.CODE_RESULT
        assert result.content == "2"

    # -- ignored events ------------------------------------------------------

    def test_unknown_node_returns_none(self, agent):
        raw = self._make_event("on_chain_start", "LangGraph", {})
        assert agent._map_langgraph_event(raw) is None

    def test_empty_node_name_returns_none(self, agent):
        raw = {"event": "on_chain_end", "metadata": {}, "data": {}}
        assert agent._map_langgraph_event(raw) is None

    def test_non_chain_event_returns_none(self, agent):
        raw = self._make_event("on_chat_model_stream", "generate", {"chunk": "tok"})
        assert agent._map_langgraph_event(raw) is None

    def test_missing_messages_returns_none(self, agent):
        raw = self._make_event("on_chain_end", "generate", {"output": {"input": []}})
        assert agent._map_langgraph_event(raw) is None


# ---------------------------------------------------------------------------
# run_stream() integration tests (mocked astream_events)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunStream:
    """Tests for BaseAgent.run_stream() with a mocked LangGraph app."""

    @pytest.fixture
    def agent(self):
        mock_llm = MagicMock()
        mock_llm.model_name = "mock-model"
        mock_llm.invoke.return_value = MagicMock(content="<solution>done</solution>")
        with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm)):
            from BaseAgent.base_agent import BaseAgent
            return BaseAgent()

    def _make_raw_events(self, events: list[dict]):
        """Build an async generator that yields the given raw event dicts."""
        async def _gen():
            for e in events:
                yield e
        return lambda *_, **__: _gen()

    def test_run_stream_yields_events(self, agent):
        ai_think = AIMessage(content="<think>planning</think>")
        state_after_generate = {
            "input": [HumanMessage(content="task"), ai_think],
            "next_step": "end",
        }
        raw_events = [
            {
                "event": "on_chain_end",
                "metadata": {"langgraph_node": "generate"},
                "data": {"output": state_after_generate},
            }
        ]
        agent.app.astream_events = self._make_raw_events(raw_events)

        collected = asyncio.run(_collect(agent.run_stream("task")))
        assert len(collected) == 1
        assert collected[0].event_type == EventType.THINKING
        assert collected[0].content == "planning"

    def test_run_stream_filter_by_event_type(self, agent):
        ai_think = AIMessage(content="<think>planning</think>")
        ai_answer = AIMessage(content="<solution>42</solution>")
        state_think = {"input": [HumanMessage(content="task"), ai_think], "next_step": "end"}
        state_answer = {"input": [HumanMessage(content="task"), ai_answer], "next_step": "end"}
        raw_events = [
            {"event": "on_chain_end", "metadata": {"langgraph_node": "generate"}, "data": {"output": state_think}},
            {"event": "on_chain_end", "metadata": {"langgraph_node": "generate"}, "data": {"output": state_answer}},
        ]
        agent.app.astream_events = self._make_raw_events(raw_events)

        collected = asyncio.run(
            _collect(agent.run_stream("task", event_types={EventType.FINAL_ANSWER}))
        )
        assert len(collected) == 1
        assert collected[0].event_type == EventType.FINAL_ANSWER

    def test_run_stream_skips_unknown_nodes(self, agent):
        raw_events = [
            {"event": "on_chain_start", "metadata": {"langgraph_node": "LangGraph"}, "data": {}},
            {"event": "on_chat_model_stream", "metadata": {"langgraph_node": "generate"}, "data": {}},
        ]
        agent.app.astream_events = self._make_raw_events(raw_events)

        collected = asyncio.run(_collect(agent.run_stream("task")))
        assert collected == []

    def test_run_stream_sets_user_task(self, agent):
        agent.app.astream_events = self._make_raw_events([])
        asyncio.run(_collect(agent.run_stream("my task")))
        assert agent.user_task == "my task"


async def _collect(agen) -> list[AgentEvent]:
    """Collect all items from an async generator into a list."""
    return [item async for item in agen]
