"""Graph node implementations for BaseAgent.

Each node is a method of NodeExecutor, which holds a reference to the parent
BaseAgent. This design makes nodes independently testable: construct a
NodeExecutor with a mock agent and call node methods directly.
"""

import re
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from BaseAgent.llm import extract_usage_metrics
from BaseAgent.prompts import get_feedback_prompt
from BaseAgent.state import AgentState
from BaseAgent.tools.support_tools import run_python_repl
from BaseAgent.utils.execution import detect_code_language, run_bash_script, run_r_code, run_with_timeout, strip_code_markers

if TYPE_CHECKING:
    from BaseAgent.base_agent import BaseAgent


class NodeExecutor:
    """Encapsulates graph node logic, independently testable."""

    def __init__(self, agent: "BaseAgent"):
        self.agent = agent

    # ------------------------------------------------------------------
    # Context window management
    # ------------------------------------------------------------------

    def _truncate_messages(self, messages: list) -> list:
        """Sliding window: keep first message + last (N-1) messages.

        The first message in state["input"] is the initial user HumanMessage
        (the task). It is always preserved so the LLM retains its objective.
        The system prompt is prepended separately in generate() and is not
        part of state["input"].

        When max_context_messages is None (default), all messages are passed
        through unchanged.
        """
        if not messages:
            return messages
        max_msgs = self.agent.max_context_messages
        if max_msgs is None or len(messages) <= max_msgs:
            return messages
        return [messages[0]] + messages[-(max_msgs - 1):]

    # ------------------------------------------------------------------
    # Core nodes
    # ------------------------------------------------------------------

    def generate(self, state: "AgentState") -> "AgentState":
        """LLM invocation + response parsing."""
        agent = self.agent

        # Create a system message with the system prompt
        system_message = SystemMessage(content=agent.system_prompt)
        # Enable prompt caching for Claude 3+ models
        if agent.source == "Anthropic":
            system_message.additional_kwargs = {
                "cache_control": {"type": "ephemeral"}
            }

        # Add the system prompt to the input to LLM; apply sliding window if configured
        input = [system_message] + self._truncate_messages(state["input"])
        output = agent.llm.invoke(input)

        usage_metrics = extract_usage_metrics(agent.source, output, model=getattr(agent.llm, "model_name", None))
        if usage_metrics is not None:
            agent._record_usage(usage_metrics)

        # Parse the response
        resp = str(output.content)

        # Check for incomplete tags and fix them
        if "<execute>" in resp and "</execute>" not in resp:
            resp += "</execute>"
        if "<solution>" in resp and "</solution>" not in resp:
            resp += "</solution>"
        if "<think>" in resp and "</think>" not in resp:
            resp += "</think>"

        # Parse the response
        think_match = re.search(r"<think>(.*?)</think>", resp, re.DOTALL)
        execute_match = re.search(r"<execute>(.*?)</execute>", resp, re.DOTALL)
        answer_match = re.search(r"<solution>(.*?)</solution>", resp, re.DOTALL)

        # Add the message to the state before checking for errors
        state["input"].append(AIMessage(content=resp.strip()))

        if answer_match:
            state["next_step"] = "end"
            state["pending_code"] = None
            state["pending_language"] = None
        elif execute_match:
            code = execute_match.group(1).strip()
            language, _ = detect_code_language(code)
            state["pending_code"] = code
            state["pending_language"] = language
            state["next_step"] = "execute"
        elif think_match:
            state["next_step"] = "generate"
            state["pending_code"] = None
            state["pending_language"] = None
        else:
            # Response doesn't contain required tags, will retry
            print("parsing error. Below is the last response from the LLM:")
            print("--------------------------------")
            print(resp)
            print("--------------------------------")

            error_count = sum(
                1 for _ in state["input"] if isinstance(_, AIMessage) and "There are no tags" in _.content
            )

            state["pending_code"] = None
            state["pending_language"] = None
            if error_count >= 2:
                # If we've already tried to correct the model twice, just end the conversation
                print("Detected repeated parsing errors, ending conversation")
                state["next_step"] = "end"
                # Add a final message explaining the termination
                state["input"].append(
                    AIMessage(
                        content="Execution terminated due to repeated parsing errors. Please check your input prompt and try again."
                    )
                )
            else:
                # Try to correct it
                state["input"].append(
                    HumanMessage(
                        content="Each response must include thinking process followed by either <execute> or <solution> tag. But there are no tags in the current response. Please follow the instruction, fix and regenerate the response again."
                    )
                )
                state["next_step"] = "generate"
        return state

    def execute(self, state: "AgentState") -> "AgentState":
        """Code dispatch + execution."""
        agent = self.agent
        last_resp = state["input"][-1].content
        # Only add the closing tag if it's not already there
        if "<execute>" in last_resp and "</execute>" not in last_resp:
            last_resp += "</execute>"

        execute_match = re.search(r"<execute>(.*?)</execute>", last_resp, re.DOTALL)
        if execute_match:
            code = execute_match.group(1)

            # Set timeout duration (10 minutes = 600 seconds)
            timeout = agent.timeout_seconds

            language, _ = detect_code_language(code.strip())
            stripped_code = strip_code_markers(code.strip(), language)

            if language == "r":
                result = run_with_timeout(run_r_code, [stripped_code], timeout=timeout)
            elif language == "bash":
                # CLI commands are run as single-line bash; bash scripts keep newlines
                if code.strip().startswith("#!CLI"):
                    stripped_code = stripped_code.replace("\n", " ")
                result = run_with_timeout(run_bash_script, [stripped_code], timeout=timeout)
            else:
                # Python: clear plots, activate per-instance patches, inject custom tools
                agent._plot_capture.clear()
                agent._plot_capture.apply_patches()
                agent._inject_custom_functions_to_repl()
                result = run_with_timeout(
                    run_python_repl,
                    [code, agent._repl_namespace],
                    timeout=timeout,
                )

            if len(result) > 10000:
                result = (
                    "The output is too long to be added to context. Here are the first 10K characters...\n"
                    + result[:10000]
                )

            # Store the execution result with the triggering message
            if not hasattr(agent, "_execution_results"):
                agent._execution_results = []

            # Get any plots captured during this execution (per-instance buffer)
            execution_plots = []
            try:
                execution_plots = agent._plot_capture.get_plots()
            except Exception as e:
                print(f"Warning: Could not capture plots from execution: {e}")
                execution_plots = []

            # Store the execution result with metadata
            execution_entry = {
                "triggering_message": last_resp,  # The AI message that contained <execute>
                "images": execution_plots,  # Base64 encoded images from this execution
                "timestamp": datetime.now().isoformat(),
            }
            agent._execution_results.append(execution_entry)

            observation = f"\n<observation>{result}</observation>"
            state["input"].append(AIMessage(content=observation.strip()))

        state["pending_code"] = None
        state["pending_language"] = None
        return state

    def execute_self_critic(self, state: "AgentState", test_time_scale_round: int) -> "AgentState":
        """LLM feedback generation for self-critic mode."""
        agent = self.agent
        if agent.critic_count < test_time_scale_round:
            # Generate feedback based on message history; apply sliding window if configured.
            # Note: no system message is prepended here — first element of state["input"]
            # is the user HumanMessage (the task).
            input = self._truncate_messages(state["input"])
            feedback_prompt = get_feedback_prompt(agent.user_task)
            feedback = agent.llm.invoke(input + [HumanMessage(content=feedback_prompt)])

            usage_metrics = extract_usage_metrics(agent.source, feedback, model=getattr(agent.llm, "model_name", None))
            if usage_metrics is not None:
                agent._record_usage(usage_metrics)

            # Add feedback as a new message
            state["input"].append(
                HumanMessage(
                    content=f"Wait... this is not enough to solve the task. Here are some feedbacks for improvement:\n{feedback.content}"
                )
            )
            agent.critic_count += 1
            state["next_step"] = "generate"
        else:
            state["next_step"] = "end"

        return state

    # ------------------------------------------------------------------
    # Routing functions (conditional edges)
    # ------------------------------------------------------------------

    def approval_gate(self, state: "AgentState") -> "AgentState":
        """Pause for human approval before code execution.

        Calls ``interrupt()`` with the pending code info so the caller can
        inspect it via ``result["__interrupt__"]``.  On resume:

        - ``Command(resume=True)`` — approved; state is returned unchanged so
          routing proceeds to ``execute``.
        - ``Command(resume={"approved": False, "feedback": "..."})`` — rejected;
          the feedback is injected as a :class:`HumanMessage` and
          ``next_step`` is set to ``"generate"`` so the agent tries again.

        **Important**: this node re-executes from the beginning on every
        resume.  There are no side effects before ``interrupt()``, so
        re-execution is safe.
        """
        code = state.get("pending_code") or ""
        language = state.get("pending_language") or "python"

        decision = interrupt({
            "code": code,
            "language": language,
            "message": f"Review this {language} code block before execution",
        })

        if isinstance(decision, dict):
            approved = decision.get("approved", True)
            feedback = decision.get("feedback", "")
        else:
            approved = bool(decision)
            feedback = ""

        if approved:
            return state

        # Rejected: inject feedback and route back to generate
        feedback_msg = feedback or "User rejected this code. Try a different approach."
        state["input"].append(HumanMessage(content=feedback_msg))
        state["next_step"] = "generate"
        state["pending_code"] = None
        state["pending_language"] = None
        return state

    def routing_function(self, state: "AgentState") -> Literal["execute", "generate", "end"]:
        """Conditional edge: maps next_step to node name."""
        next_step = state.get("next_step")
        if next_step == "execute":
            return "execute"
        elif next_step == "generate":
            return "generate"
        elif next_step == "end":
            return "end"
        else:
            raise ValueError(f"Unexpected next_step: {next_step}")

    def routing_function_self_critic(self, state: "AgentState") -> Literal["generate", "end"]:
        """Conditional edge for self-critic branch."""
        next_step = state.get("next_step")
        if next_step == "generate":
            return "generate"
        elif next_step == "end":
            return "end"
        else:
            raise ValueError(f"Unexpected next_step: {next_step}")

    # ------------------------------------------------------------------
    # Retrieve node (Problem 3: moves retriever inside the graph)
    # ------------------------------------------------------------------

    def retrieve(self, state: "AgentState") -> "AgentState":
        """Resource selection via LLM retriever.

        Skill retrieval always runs when 2+ skills are loaded (progressive
        disclosure — selects relevant skill bodies for the current task).
        Tool/data/library retrieval runs only when ``use_tool_retriever=True``.
        """
        agent = self.agent
        prompt = state["input"][-1].content
        updated = False

        # Skill retrieval: when 2+ skills loaded, select relevant ones for this task
        if len(agent.resource_manager.get_all_skills()) > 1:
            agent._select_skills_for_prompt(prompt)
            agent.system_prompt = agent._generate_system_prompt(
                self_critic=agent.self_critic,
                is_retrieval=True,
            )
            updated = True

        # Tool/data/library retrieval: only when use_tool_retriever=True
        if agent.use_tool_retriever:
            agent._select_resources_for_prompt(prompt)
            agent.system_prompt = agent._generate_system_prompt(
                self_critic=agent.self_critic,
                is_retrieval=True,
            )
            updated = True

        if updated:
            # Sync system message already in state
            if state["input"] and isinstance(state["input"][0], SystemMessage):
                state["input"][0] = SystemMessage(content=agent.system_prompt)

        return state
