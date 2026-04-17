"""Unit tests for BaseAgent.errors — structured error hierarchy."""

from __future__ import annotations

import pytest

from BaseAgent.errors import AgentTimeoutError, BaseAgentError, BudgetExceededError, LLMError

pytestmark = pytest.mark.unit


class TestErrorHierarchy:
    def test_timeout_error_is_base(self):
        assert issubclass(AgentTimeoutError, BaseAgentError)

    def test_llm_error_is_base(self):
        assert issubclass(LLMError, BaseAgentError)

    def test_budget_error_is_base(self):
        assert issubclass(BudgetExceededError, BaseAgentError)

    def test_base_agent_error_is_exception(self):
        assert issubclass(BaseAgentError, Exception)

    def test_catch_all_as_base(self):
        errors = [AgentTimeoutError("t"), LLMError("l"), BudgetExceededError("b")]
        for err in errors:
            with pytest.raises(BaseAgentError):
                raise err


class TestAgentTimeoutError:
    def test_message(self):
        e = AgentTimeoutError("timed out")
        assert str(e) == "timed out"

    def test_default_timeout(self):
        e = AgentTimeoutError("msg")
        assert e.timeout_seconds == 0.0

    def test_custom_timeout(self):
        e = AgentTimeoutError("msg", timeout_seconds=300.0)
        assert e.timeout_seconds == 300.0


class TestBudgetExceededError:
    def test_message(self):
        e = BudgetExceededError("over budget")
        assert str(e) == "over budget"

    def test_defaults(self):
        e = BudgetExceededError("msg")
        assert e.cost == 0.0
        assert e.budget == 0.0

    def test_custom_attrs(self):
        e = BudgetExceededError("msg", cost=1.5, budget=1.0)
        assert e.cost == 1.5
        assert e.budget == 1.0


class TestLLMError:
    def test_message(self):
        e = LLMError("API rate limit")
        assert str(e) == "API rate limit"
