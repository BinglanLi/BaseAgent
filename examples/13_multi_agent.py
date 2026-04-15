"""Example 13: AgentTeam — supervisor-routed multi-agent orchestration.

AgentTeam coordinates multiple BaseAgent instances via a supervisor LLM that
decides which specialist agent to call next based on the task and results.

This example uses mock agents that run without API keys. The commented-out
section at the bottom shows real usage with actual LLM API keys.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from BaseAgent import AgentTeam
from BaseAgent.agent_spec import AgentSpec
from BaseAgent.multi_agent import SupervisorDecision


# ---------------------------------------------------------------------------
# Mock demo — runs without API keys
# ---------------------------------------------------------------------------

def make_demo_agent(name: str, role: str, result: str):
    """Create a mock BaseAgent that returns a fixed result."""
    mock_llm = MagicMock()
    mock_llm.model_name = "mock-model"
    with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm)):
        from BaseAgent.base_agent import BaseAgent
        agent = BaseAgent(
            spec=AgentSpec(name=name, role=role),
            require_approval="never",
        )
    agent.arun = AsyncMock(return_value=([], result))
    return agent


def run_mock_demo():
    """Demonstrate AgentTeam routing with mock agents."""
    print("=== AgentTeam Mock Demo ===\n")

    analyst = make_demo_agent(
        name="analyst",
        role="Data analyst that examines datasets and produces statistics",
        result="Dataset has 1,000 rows. Mean age: 34.5. Top disease: Alzheimer's (42%).",
    )
    writer = make_demo_agent(
        name="writer",
        role="Report writer that summarises analysis results into a readable report",
        result="Report: Analysis of 1,000 patient records reveals Alzheimer's as the leading diagnosis (42%), with a mean patient age of 34.5 years.",
    )

    # Supervisor call sequence: analyst → writer → FINISH
    call_count = 0
    def supervisor_side_effect(messages):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return SupervisorDecision(next_agent="analyst", sub_task="Analyse the patient dataset")
        elif call_count == 2:
            return SupervisorDecision(next_agent="writer", sub_task="Write a report from the analysis")
        return SupervisorDecision(next_agent="FINISH", sub_task="")

    mock_supervisor = MagicMock()
    mock_supervisor.with_structured_output.return_value.invoke.side_effect = supervisor_side_effect

    with patch("BaseAgent.multi_agent.orchestrator.get_llm", return_value=("Anthropic", mock_supervisor)):
        team = AgentTeam(
            agents=[analyst, writer],
            supervisor_llm="mock-model",
            max_rounds=5,
        )

    log, result = team.run_sync("Analyse the patient dataset and write a summary report")
    team.close()

    print(f"Final result:\n{result}\n")
    print(f"Supervisor calls: {call_count}")


# ---------------------------------------------------------------------------
# Real usage (requires API keys)
# ---------------------------------------------------------------------------

# Uncomment to run with actual LLM API keys:
#
# from BaseAgent import BaseAgent, AgentTeam
# from BaseAgent.agent_spec import AgentSpec
#
# SKILLS_DIR = "skills"
#
# team = AgentTeam(
#     agents=[
#         BaseAgent(
#             spec=AgentSpec(
#                 name="analyst",
#                 role="A data analyst that examines datasets and produces statistics.",
#             ),
#             require_approval="never",
#         ),
#         BaseAgent(
#             spec=AgentSpec(
#                 name="writer",
#                 role="A report writer that summarises analysis results into readable reports.",
#             ),
#             require_approval="never",
#         ),
#     ],
#     supervisor_llm="claude-sonnet-4-20250514",
#     max_rounds=10,
# )
#
# log, result = team.run_sync("Analyse the dataset and write a summary report")
# print(result)
# team.close()


if __name__ == "__main__":
    run_mock_demo()
