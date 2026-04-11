"""
Skills System Example
=====================

This example shows how to use the skills system to give agents
specialized domain knowledge via SKILL.md files.

Skills directory convention:
  skills/
    ontology-design/
      SKILL.md            # required: YAML frontmatter + markdown body
      references/         # optional: bundled reference documents
        owl_spec.md
      scripts/            # optional: bundled template scripts
        template.py

Single agent: full skill body is always injected.
Multiple skills: catalog mode (metadata only) shown initially;
the retrieve node selects relevant skills per task and injects their full bodies.
"""

from BaseAgent import BaseAgent
from BaseAgent.agent_spec import AgentSpec

SKILLS_DIR = "examples/skills"

# --------------------------
# Single agent, all skills (legacy glob mode — no AgentSpec)
# --------------------------
agent = BaseAgent(skills_directory=SKILLS_DIR)
# All SKILL.md files under SKILLS_DIR are loaded.
# With multiple skills, the system prompt shows a catalog; the retrieve node
# selects relevant skill bodies on each run.

# --------------------------
# Single agent, ad-hoc skill from object
# --------------------------
from BaseAgent.resources import Skill

agent2 = BaseAgent()
agent2.add_skill(Skill(
    name="cypher-export",
    description="Best practices for exporting data to Memgraph via Cypher",
    tools=["run_python_repl"],
    instructions="## Cypher Export\n1. Batch nodes in chunks of 1000\n2. Use MERGE to avoid duplicates",
))

# --------------------------
# Multi-agent setup (spec-driven targeted loading)
# --------------------------
# Each agent loads ONLY the skills it needs:
#   spec.skill_names → loads {skills_directory}/{name}/SKILL.md directly (no glob)

oncology_agent = BaseAgent(
    skills_directory=SKILLS_DIR,
    spec=AgentSpec(
        name="oncology_agent",
        role="A disease domain expert that defines disease ontology schemas",
        skill_names=["oncology_agent_protocol"],
    ),
)

mapping_agent = BaseAgent(
    skills_directory=SKILLS_DIR,
    spec=AgentSpec(
        name="mapping_agent",
        role="An ontology alignment agent that maps extracted data to OWL terms",
        skill_names=["mapping_agent_protocol"],
    ),
)
