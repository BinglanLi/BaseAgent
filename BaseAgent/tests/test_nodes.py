"""Unit tests for BaseAgent.nodes.NodeExecutor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from BaseAgent.nodes import NodeExecutor
from helpers.node_helpers import make_mock_agent_attrs as make_agent, make_state

pytestmark = pytest.mark.unit


class TestRouting:
    """Tests for routing_function()."""

    def test_raises_on_unknown(self):
        executor = NodeExecutor(make_agent())
        state = make_state()
        state["next_step"] = "bogus"
        with pytest.raises(ValueError):
            executor.routing_function(state)


class TestRoutingSelfCritic:
    """Tests for routing_function_self_critic()."""

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

        # An observation message should have been appended as a HumanMessage
        last = result["input"][-1]
        assert isinstance(last, HumanMessage)
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
            HumanMessage(content="There are no tags in this response"),
            HumanMessage(content="There are no tags again"),
        ]
        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None):
            executor = NodeExecutor(agent)
            state = make_state([HumanMessage(content="task")] + prior_errors)
            result = executor.generate(state)
        assert result["next_step"] == "end"

    def testusage_metrics_recorded_when_returned(self):
        """When extract_usage_metrics returns a value, agent._record_usage is called."""
        agent = make_agent()
        agent.llm.invoke.return_value = self._make_llm_response("<solution>done</solution>")
        fake_metrics = {"input_tokens": 10, "output_tokens": 5}
        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=fake_metrics):
            executor = NodeExecutor(agent)
            state = make_state()
            executor.generate(state)
        agent._record_usage.assert_called_once_with(fake_metrics)

    def testusage_metrics_not_recorded_when_none(self):
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


# ---------------------------------------------------------------------------
# Feature 6: Error Handling + Termination Conditions
# ---------------------------------------------------------------------------

def make_agent_with_limits(**limits):
    """Build a mock agent with termination limit attributes set."""
    agent = make_agent()
    agent.max_iterations = limits.get("max_iterations", None)
    agent.max_cost = limits.get("max_cost", None)
    agent.max_consecutive_errors = limits.get("max_consecutive_errors", None)
    agent.usage_metrics = limits.get("usage_metrics", [])
    agent._run_usage_start = limits.get("_run_usage_start", 0)
    return agent


def _llm_response(content):
    resp = MagicMock()
    resp.content = content
    resp.usage_metadata = None
    return resp


class TestTerminationMaxIterations:
    """generate() terminates early when max_iterations is exceeded."""

    def test_terminates_when_over_limit(self):
        agent = make_agent_with_limits(max_iterations=2)
        executor = NodeExecutor(agent)
        # Simulate _iteration_count already at limit
        executor._iteration_count = 2
        state = make_state()
        result = executor.generate(state)
        assert result["next_step"] == "end"
        assert "terminated due to max_iterations" in result["input"][-1].content

    def test_does_not_terminate_within_limit(self):
        agent = make_agent_with_limits(max_iterations=5)
        agent.llm.invoke.return_value = _llm_response("<solution>done</solution>")
        executor = NodeExecutor(agent)
        executor._iteration_count = 0  # first call
        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None):
            state = make_state()
            result = executor.generate(state)
        assert result["next_step"] == "end"  # ended by solution, not limit
        assert "terminated due to" not in result["input"][-1].content

    def test_none_disables_limit(self):
        agent = make_agent_with_limits(max_iterations=None)
        agent.llm.invoke.return_value = _llm_response("<solution>done</solution>")
        executor = NodeExecutor(agent)
        executor._iteration_count = 9999  # would exceed any limit
        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None):
            state = make_state()
            result = executor.generate(state)
        # Should NOT terminate due to iterations — the solution tag fires instead
        assert result["next_step"] == "end"
        assert "terminated due to" not in result["input"][-1].content


class TestTerminationMaxCost:
    """generate() terminates early when per-run cost budget is exceeded."""

    def _make_metric(self, cost):
        m = MagicMock()
        m.cost = cost
        return m

    def test_terminates_when_over_budget(self):
        metrics = [self._make_metric(0.6), self._make_metric(0.6)]
        agent = make_agent_with_limits(max_cost=1.0, usage_metrics=metrics, _run_usage_start=0)
        executor = NodeExecutor(agent)
        state = make_state()
        result = executor.generate(state)
        assert result["next_step"] == "end"
        assert "terminated due to cost budget" in result["input"][-1].content

    def test_does_not_terminate_under_budget(self):
        metrics = [self._make_metric(0.3)]
        agent = make_agent_with_limits(max_cost=1.0, usage_metrics=metrics, _run_usage_start=0)
        agent.llm.invoke.return_value = _llm_response("<solution>done</solution>")
        executor = NodeExecutor(agent)
        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None):
            state = make_state()
            result = executor.generate(state)
        assert "terminated due to cost budget" not in result["input"][-1].content

    def test_none_cost_metrics_skipped(self):
        """Metrics with cost=None do not cause TypeError and do not count toward budget."""
        metrics = [self._make_metric(None), self._make_metric(None)]
        agent = make_agent_with_limits(max_cost=0.01, usage_metrics=metrics, _run_usage_start=0)
        agent.llm.invoke.return_value = _llm_response("<solution>done</solution>")
        executor = NodeExecutor(agent)
        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None):
            state = make_state()
            result = executor.generate(state)
        # None costs sum to 0.0, which is < 0.01, so budget not exceeded
        assert "terminated due to cost budget" not in result["input"][-1].content

    def test_run_usage_start_isolates_previous_run(self):
        """Only metrics from _run_usage_start onwards count toward the per-run budget."""
        prev_run = [self._make_metric(5.0), self._make_metric(5.0)]  # previous run: $10
        metrics = prev_run + [self._make_metric(0.1)]  # current run: $0.10
        agent = make_agent_with_limits(
            max_cost=1.0,
            usage_metrics=metrics,
            _run_usage_start=2,  # current run starts at index 2
        )
        agent.llm.invoke.return_value = _llm_response("<solution>done</solution>")
        executor = NodeExecutor(agent)
        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None):
            state = make_state()
            result = executor.generate(state)
        assert "terminated due to cost budget" not in result["input"][-1].content


class TestTerminationMaxConsecutiveErrors:
    """generate() terminates when consecutive infra errors reach the limit."""

    def test_terminates_when_at_limit(self):
        agent = make_agent_with_limits(max_consecutive_errors=3)
        executor = NodeExecutor(agent)
        executor._consecutive_error_count = 3
        state = make_state()
        result = executor.generate(state)
        assert result["next_step"] == "end"
        assert "terminated due to too many consecutive" in result["input"][-1].content

    def test_does_not_terminate_below_limit(self):
        agent = make_agent_with_limits(max_consecutive_errors=3)
        agent.llm.invoke.return_value = _llm_response("<solution>done</solution>")
        executor = NodeExecutor(agent)
        executor._consecutive_error_count = 2  # one below limit
        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None):
            state = make_state()
            result = executor.generate(state)
        assert "terminated due to too many consecutive" not in result["input"][-1].content

    def test_none_disables_check(self):
        agent = make_agent_with_limits(max_consecutive_errors=None)
        agent.llm.invoke.return_value = _llm_response("<solution>done</solution>")
        executor = NodeExecutor(agent)
        executor._consecutive_error_count = 9999
        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None):
            state = make_state()
            result = executor.generate(state)
        assert "terminated due to too many consecutive" not in result["input"][-1].content


class TestLLMErrorHandling:
    """generate() raises LLMError when llm.invoke() fails."""

    def test_llm_invoke_failure_raises_llm_error(self):
        from BaseAgent.errors import LLMError
        agent = make_agent_with_limits()
        agent.llm.invoke.side_effect = RuntimeError("connection refused")
        executor = NodeExecutor(agent)
        state = make_state()
        with pytest.raises(LLMError, match="connection refused"):
            executor.generate(state)


class TestExecuteErrorCounter:
    """execute() updates _consecutive_error_count based on run_with_timeout result."""

    def test_timeout_prefix_increments_counter(self):
        from BaseAgent.utils.execution import TIMEOUT_ERROR_PREFIX
        agent = make_agent()
        executor = NodeExecutor(agent)
        assert executor._consecutive_error_count == 0
        msg = AIMessage(content="<execute>print('hi')</execute>")
        state = make_state([msg])
        timeout_result = f"{TIMEOUT_ERROR_PREFIX} after 600 seconds."
        with patch("BaseAgent.nodes.run_with_timeout", return_value=timeout_result):
            executor.execute(state)
        assert executor._consecutive_error_count == 1

    def test_execution_error_prefix_increments_counter(self):
        from BaseAgent.utils.execution import EXECUTION_ERROR_PREFIX
        agent = make_agent()
        executor = NodeExecutor(agent)
        msg = AIMessage(content="<execute>print('hi')</execute>")
        state = make_state([msg])
        error_result = f"{EXECUTION_ERROR_PREFIX} ZeroDivisionError"
        with patch("BaseAgent.nodes.run_with_timeout", return_value=error_result):
            executor.execute(state)
        assert executor._consecutive_error_count == 1

    def test_clean_result_resets_counter(self):
        agent = make_agent()
        executor = NodeExecutor(agent)
        executor._consecutive_error_count = 5  # some prior errors
        msg = AIMessage(content="<execute>print('hi')</execute>")
        state = make_state([msg])
        with patch("BaseAgent.nodes.run_with_timeout", return_value="2\n"):
            executor.execute(state)
        assert executor._consecutive_error_count == 0

    def test_normal_code_error_string_does_not_increment(self):
        """A code error returned as plain string (e.g. SyntaxError from REPL)
        does NOT increment the counter — agent is expected to fix its own code."""
        agent = make_agent()
        executor = NodeExecutor(agent)
        msg = AIMessage(content="<execute>x = 1/0</execute>")
        state = make_state([msg])
        # run_python_repl returns the traceback as a plain string
        with patch("BaseAgent.nodes.run_with_timeout", return_value="ZeroDivisionError: division by zero"):
            executor.execute(state)
        assert executor._consecutive_error_count == 0
