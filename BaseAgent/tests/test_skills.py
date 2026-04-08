"""
Tests for the Agent Skills system.

Covers:
- Skill Pydantic model
- ResourceManager skill CRUD and selection
- BaseAgent._parse_skill_file()
- BaseAgent.add_skill() / load_skills()
- Skill injection in _generate_system_prompt()
- Retriever integration with skills
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from BaseAgent.resources import ResourceCollection, Skill
from BaseAgent.resource_manager import ResourceManager


# ==============================================================================
# Helpers
# ==============================================================================

def make_skill(**kwargs) -> Skill:
    defaults = dict(
        name="test-skill",
        description="A test skill for unit tests",
        instructions="## Instructions\nDo the thing.",
    )
    defaults.update(kwargs)
    return Skill(**defaults)


def write_skill_file(path: Path, frontmatter: str, body: str = "## Instructions\nDo the thing.") -> Path:
    path.write_text(f"---\n{frontmatter}\n---\n\n{body}")
    return path


# ==============================================================================
# TestSkillModel
# ==============================================================================

class TestSkillModel:
    def test_required_fields(self):
        skill = Skill(name="my-skill", description="does stuff")
        assert skill.name == "my-skill"
        assert skill.description == "does stuff"

    def test_defaults(self):
        skill = make_skill()
        assert skill.trigger == "auto"
        assert skill.tools == []
        assert skill.instructions == "## Instructions\nDo the thing."
        assert skill.source_path is None
        assert skill.selected is True

    def test_manual_trigger(self):
        skill = make_skill(trigger="manual")
        assert skill.trigger == "manual"

    def test_tools_list(self):
        skill = make_skill(tools=["run_python_repl", "read_function_source_code"])
        assert "run_python_repl" in skill.tools

    def test_missing_name_raises(self):
        with pytest.raises(Exception):
            Skill(description="no name")

    def test_missing_description_raises(self):
        with pytest.raises(Exception):
            Skill(name="no-desc")

    def test_selected_flag(self):
        skill = make_skill(selected=False)
        assert skill.selected is False

    def test_resource_collection_has_skills(self):
        rc = ResourceCollection()
        assert hasattr(rc, "skills")
        assert rc.skills == []


# ==============================================================================
# TestResourceManagerSkills
# ==============================================================================

class TestResourceManagerSkills:
    def setup_method(self):
        self.rm = ResourceManager()

    def test_add_skill(self):
        skill = make_skill(name="alpha")
        self.rm.add_skill(skill)
        assert len(self.rm.get_all_skills()) == 1
        assert self.rm.get_all_skills()[0].name == "alpha"

    def test_get_all_skills_empty(self):
        assert self.rm.get_all_skills() == []

    def test_get_skill_by_name(self):
        skill = make_skill(name="beta")
        self.rm.add_skill(skill)
        found = self.rm.get_skill_by_name("beta")
        assert found is not None
        assert found.name == "beta"

    def test_get_skill_by_name_missing(self):
        assert self.rm.get_skill_by_name("nope") is None

    def test_remove_skill_by_name(self):
        self.rm.add_skill(make_skill(name="gamma"))
        assert self.rm.remove_skill_by_name("gamma") is True
        assert self.rm.get_skill_by_name("gamma") is None

    def test_remove_skill_by_name_missing(self):
        assert self.rm.remove_skill_by_name("nope") is False

    def test_get_selected_skills(self):
        self.rm.add_skill(make_skill(name="s1", selected=True))
        self.rm.add_skill(make_skill(name="s2", selected=False))
        selected = self.rm.get_selected_skills()
        assert len(selected) == 1
        assert selected[0].name == "s1"

    def test_select_skills_by_names(self):
        self.rm.add_skill(make_skill(name="s1"))
        self.rm.add_skill(make_skill(name="s2"))
        self.rm.add_skill(make_skill(name="s3"))
        self.rm.select_skills_by_names(["s1", "s3"])
        selected_names = {s.name for s in self.rm.get_selected_skills()}
        assert selected_names == {"s1", "s3"}

    def test_select_skills_by_names_manual_not_deselected(self):
        self.rm.add_skill(make_skill(name="manual-skill", trigger="manual", selected=True))
        self.rm.add_skill(make_skill(name="auto-skill", trigger="auto", selected=True))
        # select only auto-skill; manual-skill should keep its selected state
        self.rm.select_skills_by_names(["auto-skill"])
        manual = self.rm.get_skill_by_name("manual-skill")
        assert manual.selected is True

    def test_select_all_resources_includes_skills(self):
        self.rm.add_skill(make_skill(name="s1", selected=False))
        self.rm.select_all_resources()
        assert self.rm.get_skill_by_name("s1").selected is True

    def test_deselect_all_resources_includes_skills(self):
        self.rm.add_skill(make_skill(name="s1", selected=True))
        self.rm.deselect_all_resources()
        assert self.rm.get_skill_by_name("s1").selected is False

    def test_get_summary_includes_skills(self):
        self.rm.add_skill(make_skill(name="s1"))
        summary = self.rm.get_summary()
        assert "skills" in summary
        assert summary["skills"]["total"] == 1
        assert summary["skills"]["selected"] == 1


# ==============================================================================
# TestParseSkillFile
# ==============================================================================

class TestParseSkillFile:
    def test_valid_skill_file(self, tmp_path):
        from BaseAgent.base_agent import BaseAgent
        skill_file = write_skill_file(
            tmp_path / "SKILL.md",
            frontmatter='name: my-skill\ndescription: "Does stuff"\ntrigger: auto\ntools:\n  - run_python_repl',
            body="## Workflow\n1. Do A\n2. Do B",
        )
        skill = BaseAgent._parse_skill_file(skill_file)
        assert skill.name == "my-skill"
        assert skill.description == "Does stuff"
        assert skill.trigger == "auto"
        assert skill.tools == ["run_python_repl"]
        assert "Do A" in skill.instructions
        assert skill.source_path == str(skill_file)

    def test_no_frontmatter_raises(self, tmp_path):
        from BaseAgent.base_agent import BaseAgent
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("# No frontmatter here")
        with pytest.raises(ValueError, match="must start with"):
            BaseAgent._parse_skill_file(skill_file)

    def test_unclosed_frontmatter_raises(self, tmp_path):
        from BaseAgent.base_agent import BaseAgent
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("---\nname: x\ndescription: y\n")
        with pytest.raises(ValueError, match="unclosed"):
            BaseAgent._parse_skill_file(skill_file)

    def test_missing_name_raises(self, tmp_path):
        from BaseAgent.base_agent import BaseAgent
        skill_file = write_skill_file(tmp_path / "SKILL.md", frontmatter="description: x")
        with pytest.raises(ValueError, match="name"):
            BaseAgent._parse_skill_file(skill_file)

    def test_missing_description_raises(self, tmp_path):
        from BaseAgent.base_agent import BaseAgent
        skill_file = write_skill_file(tmp_path / "SKILL.md", frontmatter="name: x")
        with pytest.raises(ValueError, match="description"):
            BaseAgent._parse_skill_file(skill_file)

    def test_empty_body(self, tmp_path):
        from BaseAgent.base_agent import BaseAgent
        skill_file = write_skill_file(
            tmp_path / "SKILL.md",
            frontmatter='name: empty-skill\ndescription: "No body"',
            body="",
        )
        skill = BaseAgent._parse_skill_file(skill_file)
        assert skill.instructions == ""

    def test_extra_frontmatter_fields_ignored(self, tmp_path):
        from BaseAgent.base_agent import BaseAgent
        skill_file = write_skill_file(
            tmp_path / "SKILL.md",
            frontmatter='name: x\ndescription: y\nunknown_field: foo',
        )
        # should not raise — unknown_field is silently dropped
        skill = BaseAgent._parse_skill_file(skill_file)
        assert skill.name == "x"


# ==============================================================================
# TestSkillPromptInjection
# ==============================================================================

class TestSkillPromptInjection:
    def _make_agent(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="")
        with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm)):
            from BaseAgent.base_agent import BaseAgent
            return BaseAgent()

    def test_selected_skill_appears_in_prompt(self):
        agent = self._make_agent()
        skill = make_skill(name="demo", description="Demo skill", instructions="Use method X.")
        agent.resource_manager.add_skill(skill)
        prompt = agent._generate_system_prompt()
        assert "Use method X." in prompt
        assert "demo" in prompt

    def test_deselected_skill_not_in_prompt(self):
        agent = self._make_agent()
        skill = make_skill(name="hidden", description="Hidden", instructions="Secret instructions.", selected=False)
        agent.resource_manager.add_skill(skill)
        prompt = agent._generate_system_prompt()
        assert "Secret instructions." not in prompt

    def test_no_skills_no_skills_section(self):
        agent = self._make_agent()
        prompt = agent._generate_system_prompt()
        assert "AGENT SKILLS" not in prompt

    def test_multiple_skills_all_appear(self):
        agent = self._make_agent()
        agent.resource_manager.add_skill(make_skill(name="skill-a", description="A", instructions="Instructions A."))
        agent.resource_manager.add_skill(make_skill(name="skill-b", description="B", instructions="Instructions B."))
        prompt = agent._generate_system_prompt()
        assert "Instructions A." in prompt
        assert "Instructions B." in prompt


# ==============================================================================
# TestBaseAgentAddSkill (integration, mock LLM)
# ==============================================================================

class TestBaseAgentAddSkill:
    def _make_agent(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="")
        with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm)):
            from BaseAgent.base_agent import BaseAgent
            return BaseAgent()

    def test_add_skill_object(self):
        agent = self._make_agent()
        skill = make_skill(name="my-skill", instructions="Do Y.")
        returned = agent.add_skill(skill)
        assert returned.name == "my-skill"
        assert agent.resource_manager.get_skill_by_name("my-skill") is not None
        assert "Do Y." in agent.system_prompt

    def test_add_skill_from_file(self, tmp_path):
        agent = self._make_agent()
        skill_file = write_skill_file(
            tmp_path / "SKILL.md",
            frontmatter='name: file-skill\ndescription: "From file"',
            body="## Steps\nFollow these steps.",
        )
        skill = agent.add_skill(skill_file)
        assert skill.name == "file-skill"
        assert "Follow these steps." in agent.system_prompt

    def test_add_skill_regenerates_prompt(self):
        agent = self._make_agent()
        old_prompt = agent.system_prompt
        agent.add_skill(make_skill(name="new-skill", instructions="Brand new."))
        assert agent.system_prompt != old_prompt
        assert "Brand new." in agent.system_prompt


class TestLoadSkills:
    def _make_agent(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="")
        with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm)):
            from BaseAgent.base_agent import BaseAgent
            return BaseAgent()

    def test_load_skills_empty_directory(self, tmp_path):
        agent = self._make_agent()
        skills = agent.load_skills(tmp_path)
        assert skills == []

    def test_load_skills_nonexistent_directory(self):
        agent = self._make_agent()
        skills = agent.load_skills("/nonexistent/path/to/skills")
        assert skills == []

    def test_skills_directory_config(self, tmp_path):
        write_skill_file(
            tmp_path / "SKILL.md",
            frontmatter='name: config-skill\ndescription: "Via config"',
            body="Config instructions.",
        )
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="")
        with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm)):
            from BaseAgent.base_agent import BaseAgent
            agent = BaseAgent(skills_directory=str(tmp_path))
        assert agent.resource_manager.get_skill_by_name("config-skill") is not None
        assert "Config instructions." in agent.system_prompt


# ==============================================================================
# TestSkillRetrieval
# ==============================================================================

class TestSkillRetrieval:
    def test_retriever_includes_auto_skills(self):
        from BaseAgent.retriever import ToolRetriever

        retriever = ToolRetriever()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="TOOLS: []\nDATA: []\nLIBRARIES: []\nSKILLS: [0]"
        )

        resources = {
            "all_tools": [],
            "all_data": [],
            "all_libraries": [],
            "all_skills": [{"name": "protein-skill", "description": "Protein analysis"}],
        }
        result = retriever.prompt_based_retrieval("protein structure", resources, llm=mock_llm)
        assert len(result["selected_skills"]) == 1
        assert result["selected_skills"][0]["name"] == "protein-skill"

    def test_retriever_empty_skills(self):
        from BaseAgent.retriever import ToolRetriever

        retriever = ToolRetriever()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="TOOLS: []\nDATA: []\nLIBRARIES: []\nSKILLS: []"
        )

        resources = {
            "all_tools": [],
            "all_data": [],
            "all_libraries": [],
            "all_skills": [],
        }
        result = retriever.prompt_based_retrieval("anything", resources, llm=mock_llm)
        assert result["selected_skills"] == []

    def test_retriever_no_skills_key(self):
        from BaseAgent.retriever import ToolRetriever

        retriever = ToolRetriever()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="TOOLS: []\nDATA: []\nLIBRARIES: []"
        )

        # No all_skills key — backward compat
        resources = {"all_tools": [], "all_data": [], "all_libraries": []}
        result = retriever.prompt_based_retrieval("anything", resources, llm=mock_llm)
        assert result["selected_skills"] == []
