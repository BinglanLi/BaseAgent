"""Unit tests for Feature 5: Context Window Management.

Tests cover:
- _truncate_messages() logic (edge cases, boundary conditions)
- BaseAgentConfig validation for max_context_messages
- Wiring: generate() and execute_self_critic() pass truncated input to llm.invoke()
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from BaseAgent.config import BaseAgentConfig
from BaseAgent.nodes import NodeExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_agent(max_context_messages=None):
    """Minimal mock agent for NodeExecutor tests."""
    agent = MagicMock()
    agent.source = "Anthropic"
    agent.system_prompt = "You are a helpful assistant."
    agent.use_tool_retriever = False
    agent.timeout_seconds = 30
    agent.critic_count = 0
    agent.user_task = "test task"
    agent.max_context_messages = max_context_messages
    agent.llm.invoke.return_value = MagicMock(
        content="<solution>answer</solution>", usage_metadata=None
    )
    return agent


def make_messages(n: int) -> list:
    """Return a list of n alternating HumanMessage / AIMessage objects."""
    msgs = []
    for i in range(n):
        if i % 2 == 0:
            msgs.append(HumanMessage(content=f"human {i}"))
        else:
            msgs.append(AIMessage(content=f"ai {i}"))
    return msgs


def make_state(messages=None):
    return {
        "input": messages if messages is not None else [HumanMessage(content="task")],
        "next_step": None,
        "pending_code": None,
        "pending_language": None,
    }


# ---------------------------------------------------------------------------
# _truncate_messages unit tests
# ---------------------------------------------------------------------------


class TestTruncateMessages:
    def test_empty_list(self):
        executor = NodeExecutor(make_agent())
        assert executor._truncate_messages([]) == []

    def test_disabled_by_default(self):
        """max_context_messages=None passes all messages through unchanged."""
        executor = NodeExecutor(make_agent(max_context_messages=None))
        msgs = make_messages(20)
        result = executor._truncate_messages(msgs)
        assert result is msgs  # same object, no copy

    def test_no_truncation_below_limit(self):
        """Fewer messages than the limit: no truncation."""
        executor = NodeExecutor(make_agent(max_context_messages=10))
        msgs = make_messages(5)
        result = executor._truncate_messages(msgs)
        assert result == msgs

    def test_no_truncation_at_exact_limit(self):
        """Exactly at the limit: no truncation (boundary condition)."""
        executor = NodeExecutor(make_agent(max_context_messages=5))
        msgs = make_messages(5)
        result = executor._truncate_messages(msgs)
        assert result == msgs

    def test_truncation_above_limit(self):
        """10 messages, limit 5: keeps first + last 4."""
        executor = NodeExecutor(make_agent(max_context_messages=5))
        msgs = make_messages(10)
        result = executor._truncate_messages(msgs)
        assert len(result) == 5
        assert result[0] is msgs[0]   # first message preserved
        assert result[1:] == msgs[-4:]  # last 4 messages

    def test_first_message_always_preserved(self):
        """The initial user task (index 0) is always the first element."""
        executor = NodeExecutor(make_agent(max_context_messages=3))
        msgs = make_messages(20)
        result = executor._truncate_messages(msgs)
        assert result[0] is msgs[0]

    def test_state_input_not_mutated_by_generate(self):
        """state["input"] must not be modified during generate()."""
        agent = make_agent(max_context_messages=3)
        executor = NodeExecutor(agent)
        msgs = make_messages(10)
        state = make_state(list(msgs))  # copy so we can compare
        original_len = len(state["input"])
        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None):
            executor.generate(state)
        # state["input"] grows by 1 (the AI response is appended), but is not truncated
        assert len(state["input"]) == original_len + 1


# ---------------------------------------------------------------------------
# BaseAgentConfig validation tests
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_default_is_none(self):
        config = BaseAgentConfig()
        assert config.max_context_messages is None

    def test_valid_value(self):
        config = BaseAgentConfig(max_context_messages=10)
        assert config.max_context_messages == 10

    def test_value_2_is_minimum(self):
        config = BaseAgentConfig(max_context_messages=2)
        assert config.max_context_messages == 2

    def test_value_1_raises(self):
        with pytest.raises(ValueError, match="max_context_messages"):
            BaseAgentConfig(max_context_messages=1)

    def test_value_0_raises(self):
        with pytest.raises(ValueError, match="max_context_messages"):
            BaseAgentConfig(max_context_messages=0)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="max_context_messages"):
            BaseAgentConfig(max_context_messages=-5)

    def test_env_var_valid(self, monkeypatch):
        monkeypatch.setenv("BASE_AGENT_MAX_CONTEXT_MESSAGES", "20")
        config = BaseAgentConfig()
        assert config.max_context_messages == 20
        monkeypatch.delenv("BASE_AGENT_MAX_CONTEXT_MESSAGES", raising=False)

    def test_env_var_too_small_raises(self, monkeypatch):
        monkeypatch.setenv("BASE_AGENT_MAX_CONTEXT_MESSAGES", "1")
        with pytest.raises(ValueError, match="BASE_AGENT_MAX_CONTEXT_MESSAGES"):
            BaseAgentConfig()
        monkeypatch.delenv("BASE_AGENT_MAX_CONTEXT_MESSAGES", raising=False)

    def test_env_var_non_integer_raises(self, monkeypatch):
        monkeypatch.setenv("BASE_AGENT_MAX_CONTEXT_MESSAGES", "abc")
        with pytest.raises(ValueError, match="BASE_AGENT_MAX_CONTEXT_MESSAGES"):
            BaseAgentConfig()
        monkeypatch.delenv("BASE_AGENT_MAX_CONTEXT_MESSAGES", raising=False)

    def test_to_dict_includes_field(self):
        config = BaseAgentConfig(max_context_messages=15)
        d = config.to_dict()
        assert "max_context_messages" in d
        assert d["max_context_messages"] == 15


# ---------------------------------------------------------------------------
# Wiring tests: generate() passes truncated input to llm.invoke()
# ---------------------------------------------------------------------------


class TestGenerateNodeWiring:
    def test_generate_truncates_llm_input(self):
        """generate() must pass a truncated list to llm.invoke() when limit is set."""
        agent = make_agent(max_context_messages=3)
        executor = NodeExecutor(agent)

        msgs = make_messages(10)
        state = make_state(list(msgs))

        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None):
            executor.generate(state)

        call_args = agent.llm.invoke.call_args[0][0]
        # call_args = [SystemMessage] + truncated(state["input"])
        # truncated = [msgs[0]] + msgs[-2:] = 3 messages
        # total = 1 (system) + 3 = 4
        assert len(call_args) == 4
        assert isinstance(call_args[0], SystemMessage)
        assert call_args[1] is msgs[0]  # first message preserved

    def test_generate_no_truncation_when_disabled(self):
        """generate() passes all messages when max_context_messages=None."""
        agent = make_agent(max_context_messages=None)
        executor = NodeExecutor(agent)

        msgs = make_messages(10)
        state = make_state(list(msgs))

        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None):
            executor.generate(state)

        call_args = agent.llm.invoke.call_args[0][0]
        # 1 system + 10 messages = 11
        assert len(call_args) == 11


# ---------------------------------------------------------------------------
# Wiring tests: execute_self_critic() passes truncated input to llm.invoke()
# ---------------------------------------------------------------------------


class TestSelfCriticWiring:
    def test_self_critic_truncates_llm_input(self):
        """execute_self_critic() must pass a truncated list to llm.invoke()."""
        agent = make_agent(max_context_messages=3)
        # Set critic_count=0 and test_time_scale_round=1 to exercise the branch
        agent.critic_count = 0
        agent.llm.invoke.return_value = MagicMock(content="feedback text", usage_metadata=None)

        executor = NodeExecutor(agent)
        msgs = make_messages(10)
        state = make_state(list(msgs))
        state["next_step"] = "generate"

        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None):
            with patch("BaseAgent.nodes.get_feedback_prompt", return_value="give feedback"):
                executor.execute_self_critic(state, test_time_scale_round=1)

        call_args = agent.llm.invoke.call_args[0][0]
        # truncated = [msgs[0]] + msgs[-2:] = 3 messages
        # + 1 feedback HumanMessage appended after truncation = 4 total
        assert len(call_args) == 4
        assert call_args[0] is msgs[0]  # first message preserved
        assert isinstance(call_args[-1], HumanMessage)  # feedback appended last

    def test_self_critic_no_truncation_when_disabled(self):
        """execute_self_critic() passes all messages when max_context_messages=None."""
        agent = make_agent(max_context_messages=None)
        agent.critic_count = 0
        agent.llm.invoke.return_value = MagicMock(content="feedback", usage_metadata=None)

        executor = NodeExecutor(agent)
        msgs = make_messages(10)
        state = make_state(list(msgs))

        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None):
            with patch("BaseAgent.nodes.get_feedback_prompt", return_value="give feedback"):
                executor.execute_self_critic(state, test_time_scale_round=1)

        call_args = agent.llm.invoke.call_args[0][0]
        # 10 messages + 1 feedback HumanMessage = 11
        assert len(call_args) == 11
