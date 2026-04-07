"""
Tests for Feature 2: AgentSpec + System Prompt Parameterization.

Covers:
- AgentSpec dataclass fields and defaults
- BaseAgent(spec=...) wires name, role, llm, source, temperature
- role is injected into the system prompt
- system_prompt_override replaces the generated prompt entirely
- tool_names / skill_names filter loaded resources
- spec=None preserves identical behaviour to no-spec baseline
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from BaseAgent.agent_spec import AgentSpec
from BaseAgent.prompts import _DEFAULT_ROLE_DESCRIPTION
from BaseAgent.resources import Skill


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_llm():
    mock_llm = MagicMock()
    mock_llm.model_name = "mock-model"
    mock_response = MagicMock()
    mock_response.content = "Mocked LLM response."
    mock_llm.invoke.return_value = mock_response
    return mock_llm


def _make_agent(**kwargs):
    """Create a BaseAgent with a mocked LLM (no API key required)."""
    from BaseAgent.base_agent import BaseAgent

    mock_llm = _make_mock_llm()
    with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm)):
        agent = BaseAgent(**kwargs)
    return agent


# ===========================================================================
# TestAgentSpecDataclass
# ===========================================================================

class TestAgentSpecDataclass:
    def test_required_fields(self):
        spec = AgentSpec(name="my_agent", role="a test agent")
        assert spec.name == "my_agent"
        assert spec.role == "a test agent"

    def test_optional_fields_default_to_none(self):
        spec = AgentSpec(name="a", role="b")
        assert spec.system_prompt_override is None
        assert spec.tool_names is None
        assert spec.skill_names is None
        assert spec.llm is None
        assert spec.source is None
        assert spec.temperature is None

    def test_all_fields_set(self):
        spec = AgentSpec(
            name="specialist",
            role="a specialist agent",
            system_prompt_override="Custom prompt.",
            tool_names=["run_python_repl"],
            skill_names=["data-analysis"],
            llm="gpt-4o",
            source="OpenAI",
            temperature=0.2,
        )
        assert spec.name == "specialist"
        assert spec.role == "a specialist agent"
        assert spec.system_prompt_override == "Custom prompt."
        assert spec.tool_names == ["run_python_repl"]
        assert spec.skill_names == ["data-analysis"]
        assert spec.llm == "gpt-4o"
        assert spec.source == "OpenAI"
        assert spec.temperature == 0.2


# ===========================================================================
# TestAgentSpecIntegration — BaseAgent wiring
# ===========================================================================

class TestAgentSpecIntegration:
    def test_no_spec_defaults(self):
        """Without spec, agent name defaults to 'agent' and spec is None."""
        agent = _make_agent()
        assert agent.name == "agent"
        assert agent.spec is None

    def test_spec_sets_name(self):
        spec = AgentSpec(name="ontology_analyst", role="an ontology analyst")
        agent = _make_agent(spec=spec)
        assert agent.name == "ontology_analyst"

    def test_spec_stored_on_agent(self):
        spec = AgentSpec(name="qa_agent", role="a QA agent")
        agent = _make_agent(spec=spec)
        assert agent.spec is spec

    def test_spec_llm_override(self):
        """spec.llm is used when no explicit llm kwarg is given."""
        spec = AgentSpec(name="a", role="b", llm="gpt-4o")
        agent = _make_agent(spec=spec)
        assert agent.llm_model_name == "gpt-4o"

    def test_explicit_llm_takes_priority_over_spec(self):
        """Explicit llm= kwarg beats spec.llm."""
        spec = AgentSpec(name="a", role="b", llm="gpt-4o")
        agent = _make_agent(llm="claude-sonnet-4-20250514", spec=spec)
        assert agent.llm_model_name == "claude-sonnet-4-20250514"

    def test_spec_source_override(self):
        """spec.source is forwarded to get_llm when no explicit source kwarg is given."""
        from BaseAgent.base_agent import BaseAgent

        mock_llm = _make_mock_llm()
        spec = AgentSpec(name="a", role="b", source="OpenAI")
        with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm)) as mock_get_llm:
            agent = BaseAgent(spec=spec)
        # Verify spec.source was passed into get_llm
        assert mock_get_llm.call_args.kwargs.get("source") == "OpenAI"

    def test_explicit_source_takes_priority_over_spec(self):
        """Explicit source= kwarg beats spec.source."""
        from BaseAgent.base_agent import BaseAgent

        mock_llm = _make_mock_llm()
        spec = AgentSpec(name="a", role="b", source="OpenAI")
        with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm)) as mock_get_llm:
            agent = BaseAgent(source="Anthropic", spec=spec)
        # Explicit kwarg wins
        assert mock_get_llm.call_args.kwargs.get("source") == "Anthropic"


# ===========================================================================
# TestRoleInjection — system prompt content
# ===========================================================================

class TestRoleInjection:
    def test_default_role_in_prompt_without_spec(self):
        """Without a spec, the default role string appears in the system prompt."""
        agent = _make_agent()
        assert _DEFAULT_ROLE_DESCRIPTION in agent.system_prompt

    def test_custom_role_in_prompt(self):
        """spec.role is injected into the system prompt."""
        role = "a biomedical ontology analyst that reads OWL reference ontologies"
        spec = AgentSpec(name="analyst", role=role)
        agent = _make_agent(spec=spec)
        assert role in agent.system_prompt

    def test_custom_role_replaces_default(self):
        """Default role description does NOT appear when a spec role is set."""
        spec = AgentSpec(name="analyst", role="a custom role")
        agent = _make_agent(spec=spec)
        assert _DEFAULT_ROLE_DESCRIPTION not in agent.system_prompt
        assert "a custom role" in agent.system_prompt

    def test_system_prompt_override_returned_directly(self):
        """system_prompt_override bypasses all generation and is returned as-is."""
        override = "You are fully overridden."
        spec = AgentSpec(name="a", role="b", system_prompt_override=override)
        agent = _make_agent(spec=spec)
        assert agent.system_prompt == override

    def test_system_prompt_override_excludes_resources(self):
        """With override, the normal environment resources section is absent."""
        override = "Minimal override prompt."
        spec = AgentSpec(name="a", role="b", system_prompt_override=override)
        agent = _make_agent(spec=spec)
        # The environment resources section header should not appear
        assert "Environment Resources" not in agent.system_prompt
        assert "Function Dictionary" not in agent.system_prompt


# ===========================================================================
# TestResourceFiltering
# ===========================================================================

class TestResourceFiltering:
    def test_tool_names_filters_tools(self):
        """Only the named tools remain selected when spec.tool_names is set."""
        spec = AgentSpec(name="a", role="b", tool_names=["run_python_repl"])
        agent = _make_agent(spec=spec)
        selected = [t.name for t in agent.resource_manager.get_selected_tools()]
        # run_python_repl may be excluded from the prompt description but is still selected
        all_selected = [t.name for t in
                        agent.resource_manager.collection.tools +
                        agent.resource_manager.collection.custom_tools
                        if t.selected]
        assert all_selected == ["run_python_repl"]

    def test_tool_names_none_keeps_all_tools(self):
        """spec.tool_names=None leaves all tools selected."""
        spec = AgentSpec(name="a", role="b", tool_names=None)
        agent = _make_agent(spec=spec)
        # All built-in tools are selected by default
        all_tools = agent.resource_manager.collection.tools
        assert all(t.selected for t in all_tools)

    def test_skill_names_filters_skills(self):
        """Only the named skills remain selected when spec.skill_names is set."""
        from BaseAgent.base_agent import BaseAgent

        mock_llm = _make_mock_llm()
        skill_a = Skill(name="skill-a", description="Skill A", instructions="## A")
        skill_b = Skill(name="skill-b", description="Skill B", instructions="## B")

        spec = AgentSpec(name="a", role="b", skill_names=["skill-a"])
        with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm)):
            agent = BaseAgent(spec=spec)

        agent.resource_manager.add_skill(skill_a)
        agent.resource_manager.add_skill(skill_b)
        # Re-apply spec filter to newly loaded skills
        agent.resource_manager.select_skills_by_names(["skill-a"])

        selected_skills = agent.resource_manager.get_selected_skills()
        assert len(selected_skills) == 1
        assert selected_skills[0].name == "skill-a"

    def test_no_spec_all_tools_selected(self):
        """Without a spec, all tools default to selected=True."""
        agent = _make_agent()
        all_tools = agent.resource_manager.collection.tools
        assert all(t.selected for t in all_tools)


# ===========================================================================
# TestBackwardsCompatibility
# ===========================================================================

class TestBackwardsCompatibility:
    def test_no_spec_run_interface_unchanged(self):
        """BaseAgent() without spec initialises without error."""
        agent = _make_agent()
        # Basic attributes that existing code depends on
        assert hasattr(agent, "system_prompt")
        assert hasattr(agent, "resource_manager")
        assert hasattr(agent, "llm")
        assert hasattr(agent, "node_executor")

    def test_default_role_description_constant_matches_prompt(self):
        """The _DEFAULT_ROLE_DESCRIPTION constant appears verbatim in the no-spec prompt."""
        agent = _make_agent()
        assert f"You are {_DEFAULT_ROLE_DESCRIPTION}." in agent.system_prompt

    def test_spec_none_equivalent_to_no_spec(self):
        """Explicitly passing spec=None is identical to omitting spec."""
        agent_no_spec = _make_agent()
        agent_none_spec = _make_agent(spec=None)
        assert agent_no_spec.name == agent_none_spec.name
        assert agent_no_spec.system_prompt == agent_none_spec.system_prompt
