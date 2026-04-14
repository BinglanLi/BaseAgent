"""
Skills System Example
=====================

This example shows how to use the skills system to give agents
specialized domain knowledge via SKILL.md files.

Skills directory convention:
  skills/
    skill-name/
      SKILL.md            # required: YAML frontmatter + markdown body
      references/         # optional: bundled reference documents
      scripts/            # optional: bundled template scripts

Single skill:  full body always injected into the system prompt.
Multiple skills: catalog mode (name + description only) shown initially;
  the retrieve node selects relevant skills per task and injects full bodies.
"""

from BaseAgent import BaseAgent, Skill
from BaseAgent.agent_spec import AgentSpec

SKILLS_DIR = "examples/skills"

# --------------------------
# Single agent, single skill
# --------------------------
# With one skill, the full body is always injected — no catalog mode.
agent = BaseAgent()
agent.add_skill(f"{SKILLS_DIR}/data-analyst/SKILL.md")

# --------------------------
# Ad-hoc skill from object
# --------------------------
# Build a Skill inline without a SKILL.md file.
agent2 = BaseAgent()
agent2.add_skill(Skill(
    name="citation-formatter",
    description="Formats literature references in APA or Vancouver style",
    tools=[],
    instructions="## Citation rules\n1. Vancouver: number citations in order of appearance.\n2. APA: Author, Year, Title, Journal, DOI.",
))

# --------------------------
# Multi-agent, glob mode (all skills)
# --------------------------
# load_skills() globs all SKILL.md files under SKILLS_DIR.
# With 3 skills progressive disclosure applies: the system prompt initially
# shows a catalog (name + description only); the retrieve node injects full
# skill bodies relevant to each task.
agent3 = BaseAgent(skills_directory=SKILLS_DIR)

# --------------------------
# Multi-agent, targeted loading
# --------------------------
# Each agent loads ONLY the skills it needs:
#   spec.skill_names → resolves {skills_directory}/{name}/SKILL.md directly.
# Skill names must exactly match the subdirectory name under SKILLS_DIR.

analyst_agent = BaseAgent(
    skills_directory=SKILLS_DIR,
    spec=AgentSpec(
        name="analyst-agent",
        role="A data analyst that profiles and summarises tabular biomedical datasets",
        skill_names=["data-analyst"],
    ),
)

mapper_agent = BaseAgent(
    skills_directory=SKILLS_DIR,
    spec=AgentSpec(
        name="mapper-agent",
        role="An ontology alignment agent that maps tabular columns to OWL terms",
        skill_names=["ontology-mapper"],
    ),
)

# --------------------------
# Bundled resources
# --------------------------
# The ontology-mapper skill has a references/ subdirectory.
# When any loaded skill has bundled resources, BaseAgent injects
# read_skill_resource(skill_name, path) into the REPL namespace so the
# agent can read those files at runtime:
#
#   content = read_skill_resource("ontology-mapper", "references/owl_primer.md")
#
# Path traversal outside the skill directory is blocked.
