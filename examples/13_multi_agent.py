"""Example 13: AgentTeam — supervisor-routed multi-agent orchestration.

AgentTeam coordinates multiple BaseAgent instances via a supervisor LLM that
decides which specialist agent to call next based on the task and accumulated
results.

Run this script from the repo root::

    python examples/13_multi_agent.py
"""

from BaseAgent import BaseAgent, AgentTeam
from BaseAgent.agent_spec import AgentSpec


def two_agent_pipeline():
    """Analyst → writer pipeline: analyse a topic then write a report."""
    team = AgentTeam(
        agents=[
            BaseAgent(
                spec=AgentSpec(
                    name="scientist",
                    role="A biomedical scientist with expertise in Alzheimer's disease.",
                    llm="azure-claude-haiku-4-5",
                ),
                require_approval="never",
            ),
            BaseAgent(
                spec=AgentSpec(
                    name="writer",
                    role="A report writer that turns analysis results into a clear, readable summary.",
                    llm="azure-claude-haiku-4-5",
                ),
                require_approval="never",
            ),
        ],
        supervisor_llm="azure-claude-sonnet-4-5",
        max_rounds=10,
    )

    try:
        log, result = team.run_sync(
            "Analyse the key risk factors for Alzheimer's disease and write a one-paragraph summary."
        )
        print(result)
    finally:
        team.close()


def three_agent_pipeline():
    """Researcher → analyst → writer: gather facts, analyse, then report."""
    team = AgentTeam(
        agents=[
            BaseAgent(
                spec=AgentSpec(
                    name="researcher",
                    role="A biomedical researcher that retrieves factual background on a given topic.",
                    llm="azure-claude-haiku-4-5",
                ),
                require_approval="never",
            ),
            BaseAgent(
                spec=AgentSpec(
                    name="analyst",
                    role="A data analyst that interprets research findings and identifies patterns.",
                    llm="azure-claude-haiku-4-5",
                ),
                require_approval="never",
            ),
            BaseAgent(
                spec=AgentSpec(
                    name="writer",
                    role="A technical writer that composes a structured report from analysis results.",
                    llm="azure-claude-haiku-4-5",
                ),
                require_approval="never",
            ),
        ],
        supervisor_llm="azure-claude-sonnet-4-5",
        max_rounds=15,
    )

    try:
        log, result = team.run_sync(
            "Research the role of tau protein in neurodegeneration, analyse its significance, "
            "and write a structured two-paragraph report."
        )
        print(result)
    finally:
        team.close()


if __name__ == "__main__":
    print("=== Two-agent pipeline (analyst → writer) ===")
    two_agent_pipeline()

    # print("\n=== Three-agent pipeline (researcher → analyst → writer) ===")
    # three_agent_pipeline()
