"""AgentTeam: supervisor-routed multi-agent orchestrator.

A supervisor LLM coordinates multiple BaseAgent instances by routing between
them dynamically. Each agent runs to completion before the supervisor decides
the next step.

Usage::

    from BaseAgent import BaseAgent, AgentTeam
    from BaseAgent.agent_spec import AgentSpec

    team = AgentTeam(
        agents=[
            BaseAgent(spec=AgentSpec(name="analyst", role="Data analyst"), require_approval="never"),
            BaseAgent(spec=AgentSpec(name="writer", role="Report writer"), require_approval="never"),
        ],
        supervisor_llm="claude-sonnet-4-20250514",
    )
    log, result = team.run_sync("Analyse the dataset and write a summary report")
    team.close()
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from BaseAgent.config import default_config
from BaseAgent.errors import BaseAgentError, MaxRoundsExceededError
from BaseAgent.llm import get_llm
from BaseAgent.multi_agent.state import MultiAgentState
from BaseAgent.prompts import _SUPERVISOR_PROMPT
from BaseAgent.utils.formatting import extract_agent_result

if TYPE_CHECKING:
    from BaseAgent.base_agent import BaseAgent

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    _HAS_SQLITE_SAVER = True
except ImportError:
    _HAS_SQLITE_SAVER = False

logger = logging.getLogger(__name__)


class SupervisorDecision(BaseModel):
    """Structured output schema for supervisor routing decisions."""

    next_agent: str = Field(
        description="Exact name of the next agent to call (as listed in the prompt), or the string FINISH."
    )
    sub_task: str = Field(
        description="Task instruction for the chosen agent. Empty string when next_agent is FINISH."
    )


class AgentTeam:
    """Supervisor-routed multi-agent orchestrator.

    A supervisor LLM decides which specialist agent to call next based on the
    task and accumulated results. Agents run to completion sequentially.

    Args:
        agents: List of BaseAgent instances. Each must have a unique ``spec.name``
            and must be constructed with ``require_approval="never"`` to prevent
            sub-agent interrupts that the orchestrator cannot handle.
        supervisor_llm: Model name for the supervisor. Defaults to
            ``default_config.llm``. Provider is auto-detected from the model name.
        max_rounds: Maximum number of supervisor routing calls. Raises
            ``MaxRoundsExceededError`` if exceeded before reaching FINISH.
    """

    def __init__(
        self,
        agents: list,
        supervisor_llm: str | None = None,
        max_rounds: int = 10,
    ):
        # --- Validate agents ---
        names = [a.spec.name for a in agents]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(
                f"All agents in an AgentTeam must have unique names. "
                f"Duplicate name(s): {sorted(duplicates)}"
            )

        for agent in agents:
            if agent.require_approval != "never":
                raise ValueError(
                    f"Agent '{agent.spec.name}' has require_approval="
                    f"'{agent.require_approval}'. All agents in an AgentTeam must "
                    f"be constructed with require_approval='never' to prevent "
                    f"sub-agent interrupts that the orchestrator cannot handle."
                )

        self._agent_map: dict[str, BaseAgent] = {a.spec.name: a for a in agents}
        self.max_rounds = max_rounds

        # --- Supervisor LLM (no stop sequences — uses structured output) ---
        llm_name = supervisor_llm or default_config.llm
        _, self._supervisor_llm = get_llm(llm_name)

        # --- Build and compile the LangGraph StateGraph ---
        self._checkpointer = self._create_checkpointer()
        graph = StateGraph(MultiAgentState)
        graph.add_node("supervisor", self._supervisor_node)
        for name, agent in self._agent_map.items():
            graph.add_node(name, self._make_agent_node(agent))
            graph.add_edge(name, "supervisor")
        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            self._route,
            {name: name for name in self._agent_map} | {"__end__": END},
        )
        self.app = graph.compile(checkpointer=self._checkpointer)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self, task: str, thread_id: str | None = None
    ) -> tuple[list, str]:
        """Run the agent team on a task.

        Returns:
            (log, result) where log is [] (streaming not yet implemented) and
            result is the output string of the last agent that ran.
        """
        import uuid

        thread_id = thread_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        inputs: MultiAgentState = {
            "messages": [HumanMessage(content=task)],
            "next_agent": "FINISH",
            "sub_task": "",
            "task": task,
            "results": {},
            "round": 0,
        }
        last_state = None
        async for state in self.app.astream(inputs, config=config, stream_mode="values"):
            last_state = state
        results = (last_state or {}).get("results", {})
        final_result = list(results.values())[-1] if results else ""
        return [], final_result

    def run_sync(
        self, task: str, thread_id: str | None = None
    ) -> tuple[list, str]:
        """Synchronous wrapper for run().

        Applies nest_asyncio before asyncio.run() for Jupyter notebook
        compatibility (same pattern as BaseAgent.add_mcp).
        """
        import nest_asyncio

        nest_asyncio.apply()
        return asyncio.run(self.run(task, thread_id))

    def close(self):
        """Close all sub-agent checkpointers and the orchestrator's own checkpointer."""
        for agent in self._agent_map.values():
            agent.close()
        if hasattr(self._checkpointer, "conn"):
            try:
                self._checkpointer.conn.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal graph nodes
    # ------------------------------------------------------------------

    def _supervisor_node(self, state: MultiAgentState) -> dict:
        """Supervisor node: decide which agent to call next."""
        new_round = state["round"] + 1
        if new_round > self.max_rounds:
            raise MaxRoundsExceededError(
                f"AgentTeam reached max_rounds={self.max_rounds} without FINISH.",
                max_rounds=self.max_rounds,
            )

        agent_roster = "".join(
            f"- {name}: {agent.spec.role}\n"
            for name, agent in self._agent_map.items()
        )
        results = state.get("results", {})
        if results:
            results_summary = "".join(
                f"- {name}: {extract_agent_result(result)}\n"
                for name, result in results.items()
            )
        else:
            results_summary = "None"

        prompt_text = _SUPERVISOR_PROMPT.format(
            task=state["task"],
            agent_roster=agent_roster,
            results_summary=results_summary,
        )

        decision: SupervisorDecision = (
            self._supervisor_llm
            .with_structured_output(SupervisorDecision)
            .invoke([HumanMessage(content=prompt_text)])
        )
        return {"next_agent": decision.next_agent, "sub_task": decision.sub_task, "round": new_round}

    def _make_agent_node(self, agent: BaseAgent):
        """Return a coroutine function that runs the given agent as a graph node."""

        async def _agent_node(state: MultiAgentState) -> dict:
            sub_task = state["sub_task"]
            try:
                _log, result = await agent.arun(sub_task)
            except BaseAgentError as e:
                # Re-raise: the supervisor cannot fix infrastructure errors by retrying.
                # Fatal failures (LLMError, budget exceeded) should surface to the caller.
                raise

            updated_results = dict(state.get("results", {}))
            updated_results[agent.spec.name] = result
            logger.info(
                "[AgentTeam] %s | task: %.120s | result: %.200s",
                agent.spec.name,
                sub_task,
                extract_agent_result(result),
            )
            return {
                "results": updated_results,
                "messages": [AIMessage(name=agent.spec.name, content=result)],
            }

        return _agent_node

    def _route(self, state: MultiAgentState) -> str:
        """Conditional edge: route to the next agent node or END."""
        next_agent = state["next_agent"]
        if next_agent == "FINISH":
            return END
        if next_agent not in self._agent_map:
            logger.warning(
                "Supervisor returned unknown agent %r; routing to END. Known agents: %s",
                next_agent,
                list(self._agent_map),
            )
            return END
        return next_agent

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_checkpointer(self):
        from langgraph.checkpoint.memory import MemorySaver

        db_path = default_config.checkpoint_db_path
        if db_path == ":memory:":
            return MemorySaver()
        if _HAS_SQLITE_SAVER:
            import sqlite3
            conn = sqlite3.connect(db_path, check_same_thread=False)
            return SqliteSaver(conn)
        warnings.warn(
            "langgraph-checkpoint-sqlite is not installed; falling back to "
            "in-memory checkpointer for AgentTeam.",
            stacklevel=3,
        )
        return MemorySaver()
