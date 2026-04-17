"""Shared mock factories for BaseAgent tests.

Two families:
- make_mock_agent_attrs: Family A — MagicMock agent for NodeExecutor unit tests
- make_base_agent:       Family B — real BaseAgent with patched get_llm
- make_state:            minimal AgentState dict for NodeExecutor tests
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage


def make_mock_agent_attrs(**overrides) -> MagicMock:
    """Return a MagicMock with safe attribute defaults for NodeExecutor tests.

    Pass keyword arguments to override any default attribute, e.g.::

        make_mock_agent_attrs(source="openai", require_approval="always")
    """
    llm_content = overrides.pop("llm_content", "<solution>answer</solution>")
    attrs = dict(
        source="Anthropic",
        system_prompt="You are a helpful assistant.",
        use_tool_retriever=False,
        timeout_seconds=30,
        critic_count=0,
        user_task="test task",
        max_context_messages=None,
        require_approval="never",
        max_iterations=None,
        max_cost=None,
        max_consecutive_errors=None,
        _usage_metrics=[],
        _run_usage_start=0,
    )
    attrs.update(overrides)
    agent = MagicMock()
    for k, v in attrs.items():
        setattr(agent, k, v)
    resp = MagicMock()
    resp.content = llm_content
    resp.usage_metadata = None
    agent.llm.invoke.return_value = resp
    return agent


def make_state(messages=None, pending_code=None, pending_language=None, next_step=None) -> dict:
    """Return a minimal AgentState dict for NodeExecutor tests."""
    return {
        "input": messages if messages is not None else [HumanMessage(content="test")],
        "next_step": next_step,
        "pending_code": pending_code,
        "pending_language": pending_language,
    }


def make_base_agent(llm_content: str = "<solution>done</solution>", **kwargs):
    """Patch get_llm and return a real BaseAgent instance.

    Pass keyword arguments to forward to the BaseAgent constructor, e.g.::

        make_base_agent(skills_directory="/path/to/skills")
        make_base_agent(llm_content="SKILLS: []")
    """
    mock_llm = MagicMock()
    mock_llm.model_name = "mock-model"
    mock_llm.invoke.return_value = MagicMock(content=llm_content)
    with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm)):
        from BaseAgent.base_agent import BaseAgent

        agent = BaseAgent(**kwargs)
    return agent
