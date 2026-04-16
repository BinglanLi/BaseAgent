"""
Agent Identity (AgentSpec)
==========================

``AgentSpec`` gives each agent a distinct name, role, tool/skill subset, and
optional model override.  This is the foundation for multi-agent systems where
different agents need different personas and capabilities.

Key behaviours demonstrated:
1. Custom role injected into the system prompt.
2. Tool subset — agent only sees the tools it needs.
3. Model override per agent.
4. Full system-prompt override when you need complete control.
5. Backwards compatibility — BaseAgent() without spec works unchanged.
"""

from BaseAgent import BaseAgent, AgentSpec

# ---------------------------------------------------------------------------
# 1. Basic role injection
# ---------------------------------------------------------------------------
ontology_analyst = BaseAgent(
    spec=AgentSpec(
        name="ontology_analyst",
        role=(
            "a biomedical ontology analyst that reads OWL reference ontologies "
            "and proposes a disease-specific schema with entity types and "
            "relationship types"
        ),
    )
)
print(f"Agent name : {ontology_analyst.name}")
# The system prompt now opens with:
# "You are a biomedical ontology analyst that reads OWL …"
assert "ontology analyst" in ontology_analyst.system_prompt

# ---------------------------------------------------------------------------
# 2. Tool subset — agent only exposes the tools it needs
# ---------------------------------------------------------------------------
parser_developer = BaseAgent(
    spec=AgentSpec(
        name="parser_developer",
        role="a software engineer that develops database-specific parser scripts",
        tool_names=["run_python_repl"],   # only the REPL, no other tools
    )
)
selected_tools = [
    t.name for t in
    parser_developer.resource_manager.collection.tools
    if t.selected
]
print(f"parser_developer selected tools: {selected_tools}")

# ---------------------------------------------------------------------------
# 3. Model override per agent
# ---------------------------------------------------------------------------
fast_agent = BaseAgent(
    spec=AgentSpec(
        name="fast_agent",
        role="a lightweight agent for quick lookups",
        llm="claude-haiku-4-5-20251001",   # cheaper model for simple tasks
        temperature=0.2,
    )
)
print(f"fast_agent model: {fast_agent.llm_model_name}")

# ---------------------------------------------------------------------------
# 4. Full system-prompt override
# ---------------------------------------------------------------------------
custom_agent = BaseAgent(
    spec=AgentSpec(
        name="custom_agent",
        role="ignored when override is set",
        system_prompt_override=(
            "You are a highly specialised Cypher query generator. "
            "Respond only with valid Cypher statements."
        ),
    )
)
assert custom_agent.system_prompt == (
    "You are a highly specialised Cypher query generator. "
    "Respond only with valid Cypher statements."
)
print("system_prompt_override applied correctly.")

# ---------------------------------------------------------------------------
# 5. Backwards compatibility — spec=None is identical to omitting spec
# ---------------------------------------------------------------------------
plain_agent = BaseAgent()
assert plain_agent.name == "agent"
assert plain_agent.spec is None
print("Backwards compatible: BaseAgent() without spec works unchanged.")
