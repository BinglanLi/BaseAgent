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
        assert skill.tools == []
        assert skill.instructions == "## Instructions\nDo the thing."
        assert skill.source_path is None
        assert skill.source_dir is None
        assert skill.selected is True

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

    def test_source_dir_default(self):
        skill = make_skill()
        assert skill.source_dir is None

    def test_has_bundled_resources_no_source_dir(self):
        skill = make_skill()
        assert skill.has_bundled_resources is False

    def test_has_bundled_resources_with_subdirs(self, tmp_path):
        (tmp_path / "references").mkdir()
        skill = make_skill(source_dir=str(tmp_path))
        assert skill.has_bundled_resources is True

    def test_has_bundled_resources_no_subdirs(self, tmp_path):
        skill = make_skill(source_dir=str(tmp_path))
        assert skill.has_bundled_resources is False

    def test_bundled_resource_manifest(self, tmp_path):
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "template.py").write_text("# template")
        skill = make_skill(source_dir=str(tmp_path))
        manifest = skill.bundled_resource_manifest
        assert "scripts" in manifest
        assert "template.py" in manifest["scripts"]


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

    def test_select_skills_by_names_deselects_others(self):
        self.rm.add_skill(make_skill(name="skill-x", selected=True))
        self.rm.add_skill(make_skill(name="skill-y", selected=True))
        self.rm.select_skills_by_names(["skill-y"])
        assert self.rm.get_skill_by_name("skill-x").selected is False
        assert self.rm.get_skill_by_name("skill-y").selected is True

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
            frontmatter='name: my-skill\ndescription: "Does stuff"\ntools:\n  - run_python_repl',
            body="## Workflow\n1. Do A\n2. Do B",
        )
        skill = BaseAgent._parse_skill_file(skill_file)
        assert skill.name == "my-skill"
        assert skill.description == "Does stuff"
        assert skill.tools == ["run_python_repl"]
        assert "Do A" in skill.instructions
        assert skill.source_path == str(skill_file)
        assert skill.source_dir == str(tmp_path)

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

    def test_trigger_field_silently_dropped(self, tmp_path):
        from BaseAgent.base_agent import BaseAgent
        skill_file = write_skill_file(
            tmp_path / "SKILL.md",
            frontmatter='name: x\ndescription: y\ntrigger: manual',
        )
        # Old files with trigger field should parse without error
        skill = BaseAgent._parse_skill_file(skill_file)
        assert skill.name == "x"
        assert not hasattr(skill, "trigger")


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

    def test_multiple_skills_catalog_mode(self):
        agent = self._make_agent()
        agent.resource_manager.add_skill(make_skill(name="skill-a", description="A", instructions="Instructions A."))
        agent.resource_manager.add_skill(make_skill(name="skill-b", description="B", instructions="Instructions B."))
        prompt = agent._generate_system_prompt()
        # Catalog mode: names/descriptions visible, full bodies not injected
        assert "skill-a" in prompt
        assert "skill-b" in prompt
        assert "Instructions A." not in prompt
        assert "Instructions B." not in prompt
        assert "AVAILABLE SKILLS" in prompt


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

    def test_load_skills_includes_skill_md_extension(self, tmp_path):
        skill_file = tmp_path / "test.skill.md"
        skill_file.write_text("---\nname: flat-skill\ndescription: Flat file skill\n---\nFlat instructions.")
        agent = self._make_agent()
        skills = agent.load_skills(tmp_path)
        assert len(skills) == 1
        assert skills[0].name == "flat-skill"


# ==============================================================================
# TestResolveSkillPath
# ==============================================================================

class TestResolveSkillPath:
    def _make_agent(self, skills_dir=None):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="")
        with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm)):
            from BaseAgent.base_agent import BaseAgent
            return BaseAgent(skills_directory=str(skills_dir) if skills_dir else None)

    def test_resolve_existing_skill(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        write_skill_file(skill_dir / "SKILL.md", frontmatter='name: my-skill\ndescription: "A skill"')
        agent = self._make_agent(tmp_path)
        result = agent._resolve_skill_path("my-skill")
        assert result == skill_dir / "SKILL.md"

    def test_resolve_missing_skill(self, tmp_path):
        agent = self._make_agent(tmp_path)
        result = agent._resolve_skill_path("nonexistent")
        assert result is None

    def test_resolve_no_skills_directory(self):
        agent = self._make_agent()
        result = agent._resolve_skill_path("any-skill")
        assert result is None


# ==============================================================================
# TestTargetedSkillLoading
# ==============================================================================

class TestTargetedSkillLoading:
    def _make_agent_with_spec(self, skills_dir, skill_names):
        from BaseAgent.agent_spec import AgentSpec
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="")
        with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm)):
            from BaseAgent.base_agent import BaseAgent
            return BaseAgent(
                skills_directory=str(skills_dir),
                spec=AgentSpec(name="test", role="tester", skill_names=skill_names),
            )

    def test_targeted_loading_only_loads_named_skills(self, tmp_path):
        # Create two skill subdirectories
        for name in ("skill-a", "skill-b"):
            d = tmp_path / name
            d.mkdir()
            write_skill_file(d / "SKILL.md", frontmatter=f'name: {name}\ndescription: "{name}"',
                             body=f"Instructions for {name}.")
        agent = self._make_agent_with_spec(tmp_path, ["skill-a"])
        assert agent.resource_manager.get_skill_by_name("skill-a") is not None
        assert agent.resource_manager.get_skill_by_name("skill-b") is None

    def test_targeted_loading_missing_skill_warns(self, tmp_path, capsys):
        agent = self._make_agent_with_spec(tmp_path, ["nonexistent"])
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "nonexistent" in captured.out

    def test_legacy_glob_fallback_without_spec(self, tmp_path):
        for name in ("skill-a", "skill-b"):
            d = tmp_path / name
            d.mkdir()
            write_skill_file(d / "SKILL.md", frontmatter=f'name: {name}\ndescription: "{name}"',
                             body=f"Instructions for {name}.")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="")
        with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm)):
            from BaseAgent.base_agent import BaseAgent
            agent = BaseAgent(skills_directory=str(tmp_path))
        # Both skills loaded via glob
        assert agent.resource_manager.get_skill_by_name("skill-a") is not None
        assert agent.resource_manager.get_skill_by_name("skill-b") is not None


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


# ==============================================================================
# TestBundledResourceAccess
# ==============================================================================

class TestBundledResourceAccess:
    def test_read_skill_resource_success(self, tmp_path):
        from BaseAgent.tools.support_tools import read_skill_resource
        from BaseAgent.resource_manager import ResourceManager

        refs = tmp_path / "references"
        refs.mkdir()
        (refs / "doc.md").write_text("# Reference doc")

        rm = ResourceManager()
        rm.add_skill(make_skill(name="my-skill", source_dir=str(tmp_path)))
        result = read_skill_resource("my-skill", "references/doc.md", _resource_manager=rm)
        assert "Reference doc" in result

    def test_read_skill_resource_missing_skill(self, tmp_path):
        from BaseAgent.tools.support_tools import read_skill_resource
        from BaseAgent.resource_manager import ResourceManager

        rm = ResourceManager()
        with pytest.raises(FileNotFoundError, match="not found"):
            read_skill_resource("nonexistent", "references/doc.md", _resource_manager=rm)

    def test_read_skill_resource_missing_file(self, tmp_path):
        from BaseAgent.tools.support_tools import read_skill_resource
        from BaseAgent.resource_manager import ResourceManager

        rm = ResourceManager()
        rm.add_skill(make_skill(name="my-skill", source_dir=str(tmp_path)))
        with pytest.raises(FileNotFoundError):
            read_skill_resource("my-skill", "references/nope.md", _resource_manager=rm)

    def test_read_skill_resource_path_traversal_blocked(self, tmp_path):
        from BaseAgent.tools.support_tools import read_skill_resource
        from BaseAgent.resource_manager import ResourceManager

        rm = ResourceManager()
        rm.add_skill(make_skill(name="my-skill", source_dir=str(tmp_path)))
        with pytest.raises(FileNotFoundError):
            read_skill_resource("my-skill", "../../etc/passwd", _resource_manager=rm)

    def test_read_skill_resource_no_agent_context(self):
        from BaseAgent.tools.support_tools import read_skill_resource
        with pytest.raises(RuntimeError, match="requires an agent context"):
            read_skill_resource("any", "references/doc.md")

    def test_inject_read_skill_resource_when_bundled_resources_exist(self, tmp_path):
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "template.py").write_text("# template")

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="")
        with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm)):
            from BaseAgent.base_agent import BaseAgent
            agent = BaseAgent()

        agent.resource_manager.add_skill(make_skill(name="my-skill", source_dir=str(tmp_path)))
        agent._inject_custom_functions_to_repl()
        assert "read_skill_resource" in agent._repl_namespace

    def test_no_inject_without_bundled_resources(self, tmp_path):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="")
        with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm)):
            from BaseAgent.base_agent import BaseAgent
            agent = BaseAgent()

        # Skill with source_dir but no subdirectories
        agent.resource_manager.add_skill(make_skill(name="bare-skill", source_dir=str(tmp_path)))
        agent._inject_custom_functions_to_repl()
        assert "read_skill_resource" not in agent._repl_namespace


# ==============================================================================
# TestProgressiveDisclosure
# ==============================================================================

class TestProgressiveDisclosure:
    def _make_agent(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="SKILLS: []")
        with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm)):
            from BaseAgent.base_agent import BaseAgent
            return BaseAgent()

    def test_single_skill_full_injection(self):
        agent = self._make_agent()
        agent.resource_manager.add_skill(make_skill(
            name="solo", description="Solo skill", instructions="Do this important thing."))
        prompt = agent._generate_system_prompt(is_retrieval=False)
        # Single skill → full body injected immediately
        assert "Do this important thing." in prompt
        assert "AGENT SKILLS" in prompt

    def test_multiple_skills_catalog_only_initial(self):
        agent = self._make_agent()
        agent.resource_manager.add_skill(make_skill(name="s1", description="Skill one", instructions="Body one."))
        agent.resource_manager.add_skill(make_skill(name="s2", description="Skill two", instructions="Body two."))
        prompt = agent._generate_system_prompt(is_retrieval=False)
        # Catalog mode: metadata visible, bodies hidden
        assert "s1" in prompt
        assert "s2" in prompt
        assert "Body one." not in prompt
        assert "Body two." not in prompt
        assert "AVAILABLE SKILLS" in prompt

    def test_multiple_skills_full_body_after_retrieval(self):
        agent = self._make_agent()
        agent.resource_manager.add_skill(make_skill(name="s1", description="Skill one", instructions="Body one."))
        agent.resource_manager.add_skill(make_skill(name="s2", description="Skill two", instructions="Body two."))
        # Simulate retriever selecting s1
        agent.resource_manager.select_skills_by_names(["s1"])
        prompt = agent._generate_system_prompt(is_retrieval=True)
        assert "Body one." in prompt
        assert "Body two." not in prompt

    def test_select_skills_for_prompt_selects_by_index(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="SKILLS: [0]")
        with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm)):
            from BaseAgent.base_agent import BaseAgent
            agent = BaseAgent()
        agent.resource_manager.add_skill(make_skill(name="s1", description="First skill"))
        agent.resource_manager.add_skill(make_skill(name="s2", description="Second skill"))
        agent._select_skills_for_prompt("test task")
        selected = agent.resource_manager.get_selected_skills()
        assert len(selected) == 1
        assert selected[0].name == "s1"

    def test_select_skills_for_prompt_no_match(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="SKILLS: []")
        with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm)):
            from BaseAgent.base_agent import BaseAgent
            agent = BaseAgent()
        agent.resource_manager.add_skill(make_skill(name="s1", description="First skill"))
        agent.resource_manager.add_skill(make_skill(name="s2", description="Second skill"))
        agent._select_skills_for_prompt("unrelated task")
        assert len(agent.resource_manager.get_selected_skills()) == 0

    def test_retrieve_node_skips_skill_selection_with_one_skill(self):
        from BaseAgent.nodes import NodeExecutor
        from BaseAgent.state import AgentState
        from langchain_core.messages import HumanMessage as HM

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="SKILLS: []")
        with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm)):
            from BaseAgent.base_agent import BaseAgent
            agent = BaseAgent()

        agent.resource_manager.add_skill(make_skill(name="solo", description="Only skill"))
        node = NodeExecutor(agent)
        state = AgentState(input=[HM(content="task")], next_step=None,
                           pending_code=None, pending_language=None)
        invoke_count_before = mock_llm.invoke.call_count
        node.retrieve(state)
        # No extra LLM call for skill selection with a single skill
        assert mock_llm.invoke.call_count == invoke_count_before


# ==============================================================================
# TestFunctionalToolsField
# ==============================================================================

class TestFunctionalToolsField:
    def _make_agent(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="SKILLS: []")
        with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm)):
            from BaseAgent.base_agent import BaseAgent
            return BaseAgent()

    def test_configure_warns_missing_tool(self, tmp_path, capsys):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        write_skill_file(
            skill_dir / "SKILL.md",
            frontmatter='name: my-skill\ndescription: "A skill"\ntools:\n  - nonexistent_tool',
        )
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="SKILLS: []")
        from BaseAgent.agent_spec import AgentSpec
        with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm)):
            from BaseAgent.base_agent import BaseAgent
            BaseAgent(
                skills_directory=str(tmp_path),
                spec=AgentSpec(name="t", role="r", skill_names=["my-skill"]),
            )
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "nonexistent_tool" in captured.out

    def test_configure_no_warning_for_valid_tool(self, tmp_path, capsys):
        skill_dir = tmp_path / "repl-skill"
        skill_dir.mkdir()
        write_skill_file(
            skill_dir / "SKILL.md",
            frontmatter='name: repl-skill\ndescription: "Uses REPL"\ntools:\n  - run_python_repl',
        )
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="SKILLS: []")
        from BaseAgent.agent_spec import AgentSpec
        with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm)):
            from BaseAgent.base_agent import BaseAgent
            BaseAgent(
                skills_directory=str(tmp_path),
                spec=AgentSpec(name="t", role="r", skill_names=["repl-skill"]),
            )
        captured = capsys.readouterr()
        assert "run_python_repl" not in captured.out or "Warning" not in captured.out

    def test_skill_selection_auto_selects_referenced_tools(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="SKILLS: [0]")
        with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm)):
            from BaseAgent.base_agent import BaseAgent
            agent = BaseAgent()

        # Add a skill referencing run_python_repl
        agent.resource_manager.add_skill(make_skill(
            name="repl-skill", description="Uses REPL", tools=["run_python_repl"]))
        # Deselect run_python_repl
        agent.resource_manager.select_tools_by_names([])
        assert agent.resource_manager.get_tool_by_name("run_python_repl") is not None

        agent._select_skills_for_prompt("run some code")
        # run_python_repl should be re-selected because the chosen skill requires it
        selected_tool_names = {t.name for t in agent.resource_manager.get_selected_tools()}
        assert "run_python_repl" in selected_tool_names

    def test_skill_selection_preserves_existing_selected_tools(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="SKILLS: [0]")
        with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm)):
            from BaseAgent.base_agent import BaseAgent
            agent = BaseAgent()

        agent.resource_manager.add_skill(make_skill(
            name="s1", description="Skill one", tools=["run_python_repl"]))
        # Pre-select read_function_source_code
        agent.resource_manager.select_tools_by_names(["read_function_source_code"])
        agent._select_skills_for_prompt("run code and read sources")
        selected_tool_names = {t.name for t in agent.resource_manager.get_selected_tools()}
        # Both the previously selected tool and the skill-required tool should be present
        assert "read_function_source_code" in selected_tool_names
        assert "run_python_repl" in selected_tool_names
