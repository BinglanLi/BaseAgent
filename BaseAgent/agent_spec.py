"""
AgentSpec: Identity and persona configuration for a BaseAgent instance.

Used by the multi-agent system to give each agent a distinct name, role,
tool subset, skill subset, and optional model override.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from BaseAgent.llm import SourceType


@dataclass
class AgentSpec:
    """Identity and configuration for a single agent.

    When passed to ``BaseAgent(spec=...)``, the spec:

    * Sets ``agent.name`` (used for attribution in multi-agent logs).
    * Injects ``role`` into the system prompt header.
    * Optionally restricts which tools and skills are visible to this agent.
    * Optionally overrides the LLM model, provider, and temperature.

    All fields except ``name`` and ``role`` are optional.  When a field is
    ``None`` the corresponding ``BaseAgentConfig`` default is used unchanged,
    so omitting a field never changes existing behaviour.

    Args:
        name: Unique agent identifier, e.g. ``"ontology_analyst"``.
        role: One-line description injected into the system prompt, e.g.
            ``"a biomedical ontology analyst that …"``.
        system_prompt_override: If provided, replaces the entire generated
            system prompt.  ``name`` and ``role`` are still stored but
            ``role`` is not injected.
        tool_names: If provided, only the named tools are enabled for this
            agent.  ``None`` means all loaded tools are enabled.
        skill_names: If provided, only the named skills are enabled.
            ``None`` means all loaded skills are enabled.
        llm: Model name override, e.g. ``"claude-sonnet-4-20250514"``.
        source: Provider override, e.g. ``"Anthropic"``.
        temperature: Sampling temperature override.
    """

    name: str
    role: str
    system_prompt_override: str | None = None
    tool_names: list[str] | None = None
    skill_names: list[str] | None = None
    llm: str | None = None
    source: "SourceType | None" = None
    temperature: float | None = None
