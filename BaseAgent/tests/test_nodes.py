"""Unit tests for BaseAgent.nodes.NodeExecutor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from BaseAgent.nodes import NodeExecutor


def make_agent(source="Anthropic", use_tool_retriever=False):
    """Build a minimal mock agent suitable for NodeExecutor."""
    agent = MagicMock()
    agent.source = source
    agent.system_prompt = "You are a helpful assistant."
    agent.use_tool_retriever = use_tool_retriever
    agent.timeout_seconds = 30
    agent.critic_count = 0
    agent.user_task = "test task"
    agent.max_context_messages = None
    # llm.invoke returns a mock response
    agent.llm.invoke.return_value = MagicMock(content="<solution>answer</solution>", usage_metadata=None)
    return agent


def make_state(messages=None):
    """Build a minimal AgentState dict."""
    return {
        "input": messages or [HumanMessage(content="test")],
        "next_step": None,
        "pending_code": None,
        "pending_language": None,
    }


class TestRouting:
    """Tests for routing_function()."""

    def test_routes_execute(self):
        executor = NodeExecutor(make_agent())
        state = make_state()
        state["next_step"] = "execute"
        assert executor.routing_function(state) == "execute"

    def test_routes_generate(self):
        executor = NodeExecutor(make_agent())
        state = make_state()
        state["next_step"] = "generate"
        assert executor.routing_function(state) == "generate"

    def test_routes_end(self):
        executor = NodeExecutor(make_agent())
        state = make_state()
        state["next_step"] = "end"
        assert executor.routing_function(state) == "end"

    def test_raises_on_unknown(self):
        executor = NodeExecutor(make_agent())
        state = make_state()
        state["next_step"] = "bogus"
        with pytest.raises(ValueError):
            executor.routing_function(state)


class TestRoutingSelfCritic:
    """Tests for routing_function_self_critic()."""

    def test_generate(self):
        executor = NodeExecutor(make_agent())
        state = {"next_step": "generate"}
        assert executor.routing_function_self_critic(state) == "generate"

    def test_end(self):
        executor = NodeExecutor(make_agent())
        state = {"next_step": "end"}
        assert executor.routing_function_self_critic(state) == "end"

    def test_raises_on_unknown(self):
        executor = NodeExecutor(make_agent())
        state = {"next_step": "bad"}
        with pytest.raises(ValueError):
            executor.routing_function_self_critic(state)


class TestGenerate:
    """Tests for generate() node."""

    def _make_llm_response(self, content):
        resp = MagicMock()
        resp.content = content
        resp.usage_metadata = None
        return resp

    def test_solution_tag_sets_end(self):
        agent = make_agent()
        agent.llm.invoke.return_value = self._make_llm_response(
            "<think>reasoning</think><solution>final answer</solution>"
        )
        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None):
            executor = NodeExecutor(agent)
            state = make_state()
            result = executor.generate(state)
        assert result["next_step"] == "end"

    def test_execute_tag_sets_execute(self):
        agent = make_agent()
        agent.llm.invoke.return_value = self._make_llm_response(
            "<think>let me code</think><execute>print('hi')</execute>"
        )
        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None):
            executor = NodeExecutor(agent)
            state = make_state()
            result = executor.generate(state)
        assert result["next_step"] == "execute"

    def test_think_only_loops_generate(self):
        agent = make_agent()
        agent.llm.invoke.return_value = self._make_llm_response(
            "<think>still thinking</think>"
        )
        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None):
            executor = NodeExecutor(agent)
            state = make_state()
            result = executor.generate(state)
        assert result["next_step"] == "generate"

    def test_no_tags_adds_correction_message(self):
        agent = make_agent()
        agent.llm.invoke.return_value = self._make_llm_response(
            "plain response with no tags at all"
        )
        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None):
            executor = NodeExecutor(agent)
            state = make_state()
            result = executor.generate(state)
        # Should still continue (not crash)
        assert result["next_step"] in ("generate", "end")

    def test_appends_ai_message_to_state(self):
        agent = make_agent()
        agent.llm.invoke.return_value = self._make_llm_response(
            "<solution>done</solution>"
        )
        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None):
            executor = NodeExecutor(agent)
            state = make_state()
            result = executor.generate(state)
        last_msg = result["input"][-1]
        assert isinstance(last_msg, AIMessage)


class TestRetrieve:
    """Tests for retrieve() node."""

    def test_passthrough_when_retriever_disabled(self):
        agent = make_agent(use_tool_retriever=False)
        executor = NodeExecutor(agent)
        state = make_state([HumanMessage(content="hello")])
        result = executor.retrieve(state)
        assert result == state
        agent._select_resources_for_prompt.assert_not_called()

    def test_calls_select_resources_when_enabled(self):
        agent = make_agent(use_tool_retriever=True)
        agent._generate_system_prompt.return_value = "updated system prompt"
        executor = NodeExecutor(agent)
        state = make_state([HumanMessage(content="task description")])
        result = executor.retrieve(state)
        agent._select_resources_for_prompt.assert_called_once_with("task description")


class TestExecute:
    """Tests for execute() node — Python code execution path."""

    def test_python_code_executes(self):
        agent = make_agent()
        executor = NodeExecutor(agent)
        code = "result = 1 + 1"
        msg = AIMessage(content=f"<execute>\n{code}\n</execute>")
        state = make_state([msg])

        with patch("BaseAgent.nodes.run_with_timeout", return_value="") as mock_run, \
             patch("BaseAgent.nodes.run_python_repl"):
            mock_run.return_value = ""
            result = executor.execute(state)

        # An observation message should have been appended
        last = result["input"][-1]
        assert isinstance(last, AIMessage)
        assert "<observation>" in last.content

    def test_no_execute_tag_state_unchanged(self):
        agent = make_agent()
        executor = NodeExecutor(agent)
        state = make_state([AIMessage(content="no tags here")])
        result = executor.execute(state)
        # No new message added
        assert result["input"][-1].content == "no tags here"


class TestGenerateEdgeCases:
    """Additional edge-case tests for generate()."""

    def _make_llm_response(self, content):
        resp = MagicMock()
        resp.content = content
        resp.usage_metadata = None
        return resp

    def test_incomplete_execute_tag_repaired(self):
        """Unclosed <execute> tag is auto-closed before parsing."""
        agent = make_agent()
        agent.llm.invoke.return_value = self._make_llm_response(
            "<think>thinking</think><execute>print('hi')"
        )
        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None):
            executor = NodeExecutor(agent)
            state = make_state()
            result = executor.generate(state)
        assert result["next_step"] == "execute"
        assert "</execute>" in result["input"][-1].content

    def test_incomplete_solution_tag_repaired(self):
        """Unclosed <solution> tag is auto-closed before parsing."""
        agent = make_agent()
        agent.llm.invoke.return_value = self._make_llm_response(
            "<solution>my final answer"
        )
        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None):
            executor = NodeExecutor(agent)
            state = make_state()
            result = executor.generate(state)
        assert result["next_step"] == "end"

    def test_incomplete_think_tag_repaired(self):
        """Unclosed <think> tag is auto-closed before parsing."""
        agent = make_agent()
        agent.llm.invoke.return_value = self._make_llm_response(
            "<think>still thinking"
        )
        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None):
            executor = NodeExecutor(agent)
            state = make_state()
            result = executor.generate(state)
        assert result["next_step"] == "generate"

    def test_no_tags_first_error_adds_correction_message(self):
        """First parse error appends a HumanMessage correction and loops."""
        agent = make_agent()
        agent.llm.invoke.return_value = self._make_llm_response("no tags at all")
        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None):
            executor = NodeExecutor(agent)
            state = make_state()
            result = executor.generate(state)
        assert result["next_step"] == "generate"
        last_msg = result["input"][-1]
        assert isinstance(last_msg, HumanMessage)
        assert "there are no tags" in last_msg.content.lower()

    def test_no_tags_repeated_errors_terminate(self):
        """When state already has 2 AIMessages with 'There are no tags', conversation ends."""
        agent = make_agent()
        agent.llm.invoke.return_value = self._make_llm_response("still no tags")
        prior_errors = [
            AIMessage(content="There are no tags in this response"),
            AIMessage(content="There are no tags again"),
        ]
        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None):
            executor = NodeExecutor(agent)
            state = make_state([HumanMessage(content="task")] + prior_errors)
            result = executor.generate(state)
        assert result["next_step"] == "end"

    def test_usage_metrics_recorded_when_returned(self):
        """When extract_usage_metrics returns a value, agent._record_usage is called."""
        agent = make_agent()
        agent.llm.invoke.return_value = self._make_llm_response("<solution>done</solution>")
        fake_metrics = {"input_tokens": 10, "output_tokens": 5}
        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=fake_metrics):
            executor = NodeExecutor(agent)
            state = make_state()
            executor.generate(state)
        agent._record_usage.assert_called_once_with(fake_metrics)

    def test_usage_metrics_not_recorded_when_none(self):
        """When extract_usage_metrics returns None, agent._record_usage is not called."""
        agent = make_agent()
        agent.llm.invoke.return_value = self._make_llm_response("<solution>done</solution>")
        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None):
            executor = NodeExecutor(agent)
            state = make_state()
            executor.generate(state)
        agent._record_usage.assert_not_called()

    def test_anthropic_source_adds_cache_control(self):
        """For Anthropic source, SystemMessage gets cache_control additional_kwargs."""
        agent = make_agent(source="Anthropic")
        agent.llm.invoke.return_value = self._make_llm_response("<solution>done</solution>")
        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None):
            executor = NodeExecutor(agent)
            state = make_state()
            executor.generate(state)
        # Inspect the SystemMessage passed to llm.invoke
        call_args = agent.llm.invoke.call_args[0][0]
        system_msg = call_args[0]
        assert isinstance(system_msg, SystemMessage)
        assert system_msg.additional_kwargs.get("cache_control") == {"type": "ephemeral"}

    def test_non_anthropic_source_no_cache_control(self):
        """For non-Anthropic source, no cache_control is added to SystemMessage."""
        agent = make_agent(source="OpenAI")
        agent.llm.invoke.return_value = self._make_llm_response("<solution>done</solution>")
        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None):
            executor = NodeExecutor(agent)
            state = make_state()
            executor.generate(state)
        call_args = agent.llm.invoke.call_args[0][0]
        system_msg = call_args[0]
        assert isinstance(system_msg, SystemMessage)
        assert "cache_control" not in system_msg.additional_kwargs


class TestExecuteCodePaths:
    """Tests for execute() code dispatch logic."""

    def test_r_code_dispatches_to_run_r_code(self):
        agent = make_agent()
        agent._execution_results = []
        msg = AIMessage(content="<execute>#!R\nprint('hello')\n</execute>")
        state = make_state([msg])

        with patch("BaseAgent.nodes.run_with_timeout", return_value="[1] hello") as mock_run, \
             patch("BaseAgent.nodes.run_r_code") as mock_r:
            result = executor = NodeExecutor(agent)
            result = executor.execute(state)
            # run_with_timeout was called with run_r_code
            args = mock_run.call_args[0]
            assert args[0] is mock_r

    def test_bash_code_dispatches_to_run_bash_script(self):
        agent = make_agent()
        agent._execution_results = []
        msg = AIMessage(content="<execute>#!BASH\necho hi\n</execute>")
        state = make_state([msg])

        with patch("BaseAgent.nodes.run_with_timeout", return_value="hi") as mock_run, \
             patch("BaseAgent.nodes.run_bash_script") as mock_bash:
            executor = NodeExecutor(agent)
            executor.execute(state)
            args = mock_run.call_args[0]
            assert args[0] is mock_bash

    def test_cli_command_joins_newlines(self):
        """#!CLI code has newlines replaced with spaces before execution."""
        agent = make_agent()
        agent._execution_results = []
        msg = AIMessage(content="<execute>#!CLI\necho\nhi\n</execute>")
        state = make_state([msg])

        with patch("BaseAgent.nodes.run_with_timeout", return_value="hi") as mock_run, \
             patch("BaseAgent.nodes.run_bash_script"):
            executor = NodeExecutor(agent)
            executor.execute(state)
            # The code passed to run_with_timeout should have no newlines
            stripped_code = mock_run.call_args[0][1][0]
            assert "\n" not in stripped_code

    def test_long_output_is_truncated(self):
        """Output exceeding 10000 chars is truncated."""
        agent = make_agent()
        agent._execution_results = []
        code = "print('x' * 20000)"
        msg = AIMessage(content=f"<execute>{code}</execute>")
        state = make_state([msg])

        long_output = "x" * 20000
        with patch("BaseAgent.nodes.run_with_timeout", return_value=long_output):
            executor = NodeExecutor(agent)
            result = executor.execute(state)

        last_content = result["input"][-1].content
        assert "too long" in last_content
        assert len(last_content) < 20000

    def test_execution_results_stored(self):
        """Each execution appends an entry to agent._execution_results."""
        agent = make_agent()
        agent._execution_results = []
        code = "x = 1"
        msg = AIMessage(content=f"<execute>{code}</execute>")
        state = make_state([msg])

        with patch("BaseAgent.nodes.run_with_timeout", return_value=""):
            executor = NodeExecutor(agent)
            executor.execute(state)

        assert len(agent._execution_results) == 1
        entry = agent._execution_results[0]
        assert "triggering_message" in entry
        assert "timestamp" in entry
        assert "images" in entry


class TestExecuteSelfCritic:
    """Tests for execute_self_critic()."""

    def test_generates_feedback_when_under_limit(self):
        """critic_count < limit → feedback appended as HumanMessage, next_step = generate."""
        agent = make_agent()
        agent.critic_count = 0
        feedback_resp = MagicMock()
        feedback_resp.content = "You should improve X."
        agent.llm.invoke.return_value = feedback_resp
        executor = NodeExecutor(agent)
        state = make_state()

        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None), \
             patch("BaseAgent.nodes.get_feedback_prompt", return_value="Give feedback."):
            result = executor.execute_self_critic(state, test_time_scale_round=2)

        assert result["next_step"] == "generate"
        assert agent.critic_count == 1
        last_msg = result["input"][-1]
        assert isinstance(last_msg, HumanMessage)
        assert "You should improve X." in last_msg.content

    def test_terminates_when_limit_reached(self):
        """critic_count == limit → next_step = end, no additional message."""
        agent = make_agent()
        agent.critic_count = 2
        executor = NodeExecutor(agent)
        state = make_state()
        initial_length = len(state["input"])

        result = executor.execute_self_critic(state, test_time_scale_round=2)

        assert result["next_step"] == "end"
        assert len(result["input"]) == initial_length

    def test_self_critic_records_usage(self):
        """extract_usage_metrics result is forwarded to agent._record_usage."""
        agent = make_agent()
        agent.critic_count = 0
        feedback_resp = MagicMock()
        feedback_resp.content = "feedback"
        agent.llm.invoke.return_value = feedback_resp
        executor = NodeExecutor(agent)
        state = make_state()

        fake_metrics = {"input_tokens": 20}
        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=fake_metrics), \
             patch("BaseAgent.nodes.get_feedback_prompt", return_value="Give feedback."):
            executor.execute_self_critic(state, test_time_scale_round=3)

        agent._record_usage.assert_called_once_with(fake_metrics)


class TestRetrieveSystemMessageUpdate:
    """Extended tests for retrieve() node — system message update path."""

    def test_updates_system_message_in_state_when_present(self):
        """When state[input][0] is a SystemMessage, it gets replaced with updated prompt."""
        agent = make_agent(use_tool_retriever=True)
        agent._generate_system_prompt.return_value = "updated prompt"
        executor = NodeExecutor(agent)
        state = {
            "input": [
                SystemMessage(content="old prompt"),
                HumanMessage(content="user task"),
            ],
            "next_step": None,
        }
        result = executor.retrieve(state)
        assert isinstance(result["input"][0], SystemMessage)
        assert result["input"][0].content == "updated prompt"

    def test_does_not_update_when_first_message_is_not_system(self):
        """When state[input][0] is not a SystemMessage, it is left unchanged."""
        agent = make_agent(use_tool_retriever=True)
        agent._generate_system_prompt.return_value = "updated prompt"
        executor = NodeExecutor(agent)
        state = make_state([HumanMessage(content="just a user message")])
        result = executor.retrieve(state)
        assert isinstance(result["input"][0], HumanMessage)
