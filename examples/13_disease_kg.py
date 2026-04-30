"""Example 13: Disease knowledge graph — AgentTeam bootstraps a disease-specific KG.

Copies alzkb-updater as a template, then uses a 6-agent AgentTeam to adapt
and extend it for a target disease. A supervisor coordinates the agents in
pipeline order: ontology → database → engineer → mapping → memgraph → evaluator.

Agents:
- ontology_agent  — updates config/project.yaml with disease-specific identifiers
- database_agent  — enables relevant sources in config/databases.yaml
- engineer_agent  — writes parsers in src/parsers/ for newly enabled sources
- mapping_agent   — adds config/ontology_mappings.yaml entries for new parsers
- memgraph_agent  — runs the full pipeline and validates graph export
- evaluator_agent — runs the three-stage eval suite and reports KG quality
- hitl_agent      — pauses for user review after ontology and database steps

Human-in-the-loop: the user reviews the template config files before agents
start, then reviews ontology and database decisions mid-pipeline via hitl_agent.

Note: disease identifiers (UMLS CUIs, DOID IDs, MeSH IDs) are filled by the LLM
and should be manually verified before running the production pipeline.

Run from the repo root::

    python examples/13_disease_kg.py
"""

import os
import shutil
import sys

from BaseAgent import BaseAgent, AgentTeam, MaxRoundsExceededError
from BaseAgent.agent_spec import AgentSpec

MCP_CONFIG = "examples/mcp_config.yaml"
SKILLS_DIR = "skills"


def copy_template(disease: str, template_src: str) -> str:
    """Copy the template repo into ./<disease_slug>-kg/ and return the path."""
    slug = disease.lower().replace(" ", "_").replace("'", "").replace("-", "_")
    dest = os.path.abspath(f"./{slug}-kg")

    if os.path.exists(dest):
        print(f"Directory {dest} already exists — skipping copy.")
        return dest

    if not os.path.isdir(template_src):
        print(f"Template source not found: {template_src}", file=sys.stderr)
        sys.exit(1)

    shutil.copytree(
        template_src,
        dest,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
    )
    print(f"Template copied to {dest}")
    return dest


def ask_user(message: str) -> str:
    """Present a summary to the human user and collect approval or feedback."""
    print(f"\n{'─' * 60}")
    print(message)
    print('─' * 60)
    response = input("Press Enter to approve, or type feedback: ").strip()
    return response if response else "approved"


def _make_agent(name: str, role: str, llm: str, skill_name: str) -> BaseAgent:
    """Create a BaseAgent with MCP access and a targeted skill loaded."""
    agent = BaseAgent(
        spec=AgentSpec(name=name, role=role, llm=llm, skill_names=[skill_name]),
        skills_directory=SKILLS_DIR,
        require_approval="never",
    )
    agent.add_mcp(MCP_CONFIG)
    return agent


def build_disease_kg(disease: str, template_src: str):
    """Bootstrap a disease KG repo using a 6-agent AgentTeam."""
    repo_path = copy_template(disease, template_src)

    print(f"\nReview the template config files before agents modify them:")
    print(f"  {repo_path}/config/project.yaml")
    print(f"  {repo_path}/config/databases.yaml")
    input("\nPress Enter to start the AgentTeam, or Ctrl+C to abort.\n")

    ontology_agent = _make_agent(
        name="ontology_agent",
        role=(
            "A biomedical ontology engineer managing the OWL schema and project configuration. "
            "You own data/ontology/alzkb_v2.rdf and config/project.yaml: update disease_scope "
            "(primary_terms, umls_cuis, doid_ids, mesh_ids) and keep node_types/edge_types in "
            "sync with the RDF. Only modify the RDF on explicit request. Never edit Python source files."
        ),
        llm="azure-claude-haiku-4-5",
        skill_name="ontology-protocol",
    )

    database_agent = _make_agent(
        name="database_agent",
        role=(
            "A bioinformatics data engineer managing config/databases.yaml. "
            "You evaluate biomedical data sources and set enabled flags for the target disease. "
            "You do not write parsers or ontology mappings."
        ),
        llm="azure-claude-haiku-4-5",
        skill_name="database-protocol",
    )

    engineer_agent = _make_agent(
        name="engineer_agent",
        role=(
            "A Python software engineer writing parsers under src/parsers/. "
            "Each parser inherits from BaseParser and downloads data from one biomedical source, "
            "returning clean pandas DataFrames. Follow the full registration checklist: "
            "src/parsers/__init__.py, src/main.py PARSERS dict, test/eval_parser.py PARSER_CLASS_MAP. "
            "Run `python src/main.py --source <name>` to verify each parser produces TSVs. "
            "You do not modify OWL files or ontology_mappings.yaml."
        ),
        llm="azure-claude-sonnet-4-6",
        skill_name="parser-protocol",
    )

    mapping_agent = _make_agent(
        name="mapping_agent",
        role=(
            "A knowledge graph mapping specialist owning config/ontology_mappings.yaml. "
            "You map parser TSV columns to OWL node types and relationship types. "
            "Other useful information such as gene descriptions, annotations, structures, and cross-references is included as properties. "
            "Always place node entries before relationship entries. "
            "Verify all OWL names against config/project.yaml node_types/edge_types before writing. "
            "Never edit Python source files."
        ),
        llm="azure-claude-haiku-4-5",
        skill_name="mapping-protocol",
    )

    memgraph_agent = _make_agent(
        name="memgraph_agent",
        role=(
            "A graph database engineer who runs the full pipeline and validates graph export. "
            "Run `python src/main.py` inside the repo to produce data/output/alzkb_v2_populated.rdf, "
            "then run MemgraphExporter. Validate import.cypher (INDEX statements, LOAD CSV paths, "
            "globally unique node IDs). Provide docker run import instructions."
        ),
        llm="azure-claude-haiku-4-5",
        skill_name="memgraph-protocol",
    )

    evaluator_agent = _make_agent(
        name="evaluator_agent",
        role=(
            "A KG quality evaluator who runs eval_after_parser.py, eval_after_ontology.py, "
            "and eval_after_memgraph.py in sequence. Report tier-1 blocking failures "
            "(zero node/edge counts, OWL conformance < 1.0) and overall KG quality. "
            "Flag any blocking failures that must be resolved before the KG is used."
        ),
        llm="azure-claude-haiku-4-5",
        skill_name="evaluation-protocol",
    )

    hitl_agent = BaseAgent(
        spec=AgentSpec(
            name="hitl_agent",
            role=(
                "A human review coordinator. Summarize the previous agent's output in 3-5 bullet "
                "points, then call ask_user once with that summary and a clear yes/no question. "
                "If the user presses Enter or types 'approved', return 'approved'. "
                "Otherwise relay the user's feedback verbatim so the supervisor can act on it."
            ),
            llm="azure-claude-haiku-4-5",
        ),
        require_approval="never",
    )
    hitl_agent.add_tool(ask_user)

    agents = [
        ontology_agent, database_agent, engineer_agent,
        mapping_agent, memgraph_agent, evaluator_agent,
        hitl_agent,
    ]

    team = AgentTeam(
        agents=agents,
        supervisor_llm="azure-claude-sonnet-4-6",
        max_rounds=20,
    )

    task = (
        f"Build a disease knowledge graph for '{disease}' in the repo at {repo_path}.\n\n"
        "Goal: produce a validated, Memgraph-importable KG with no tier-1 blocking failures.\n\n"
        "Available agents and their responsibilities:\n"
        "- ontology_agent: Update config/project.yaml — set project.name, display_name, and "
        "all disease_scope fields (primary_terms, umls_cuis, doid_ids, mesh_ids). "
        "Keep ontology paths, node_types, edge_types, and graph_indexes unchanged.\n"
        "- database_agent: Update config/databases.yaml — enable sources relevant to the "
        "disease and disable irrelevant ones.\n"
        "- engineer_agent: For each newly enabled source that has no parser in src/parsers/, "
        "write a BaseParser subclass and complete the full registration checklist. "
        "Verify with `python src/main.py --source <name>` run inside the repo.\n"
        "- mapping_agent: For each new parser, add ontology_mappings.yaml entries — "
        "node entries first, then relationship entries. Confirm all OWL names exist in "
        "project.yaml node_types/edge_types.\n"
        "- memgraph_agent: Run `python src/main.py` inside the repo, then run "
        "MemgraphExporter to produce import.cypher. Validate the Cypher script and "
        "provide docker import instructions.\n"
        "- evaluator_agent: Run all three eval scripts and report the quality summary. "
        "List any tier-1 blocking failures explicitly.\n"
        "- hitl_agent: Pause for user review at key decision points (after ontology changes, "
        "after database selection). Relay any feedback back to the responsible agent.\n\n"
        "Hard dependencies (respect these regardless of agent order):\n"
        "- ontology_agent must complete and be approved before database_agent starts.\n"
        "- database_agent must complete and be approved before engineer_agent starts.\n"
        "- All parsers must be verified before mapping_agent runs.\n"
        "- mapping_agent must complete before memgraph_agent runs.\n"
        "- evaluator_agent runs last, after memgraph_agent succeeds.\n\n"
        "Pipeline contracts — violations fail silently:\n"
        "- The databases.yaml key, PARSERS key, PARSER_CLASS_MAP key, ontology_mappings.yaml "
        "prefix, and data/processed/ subdirectory name must all be identical strings.\n"
        "- In ontology_mappings.yaml, all node entries must precede all relationship entries.\n"
        "- OWL names in ontology_mappings.yaml must be active (uncommented) in project.yaml.\n"
        "- Credentials use the _env suffix in databases.yaml args; the parser constructor "
        "must accept the stripped parameter name.\n"
    )

    try:
        _log, result = team.run_sync(task)
        print(f"\n=== Result ===\n{result}")
        print(f"\nKG repo ready at: {repo_path}")
        print("Next: review the eval report, address any blocking failures, then import with docker.")
    except MaxRoundsExceededError as e:
        print(f"Team hit round limit before finishing: {e}", file=sys.stderr)
    finally:
        _print_token_summary(agents)
        team.close()


def _print_token_summary(agents: list):
    """Print per-agent and total token counts from accumulated usage_metrics."""
    print("\n=== Token usage ===")
    totals = {"input": 0, "output": 0, "total": 0}
    for agent in agents:
        metrics = agent.usage_metrics
        input_tokens = sum(m.input_tokens or 0 for m in metrics)
        output_tokens = sum(m.output_tokens or 0 for m in metrics)
        cache_creation = sum(m.cache_creation_tokens or 0 for m in metrics)
        cache_read = sum(m.cache_read_tokens or 0 for m in metrics)
        total_tokens = sum(m.total_tokens or 0 for m in metrics)
        cost = sum(m.cost or 0.0 for m in metrics)
        cost_str = f"  ${cost:.4f}" if cost else ""
        print(f"  {agent.spec.name}: {input_tokens} in / {output_tokens} out / {cache_creation} cache creation / {cache_read} cache read / {total_tokens} total{cost_str}")
        totals["input"] += input_tokens
        totals["output"] += output_tokens
        totals["cache_creation"] += cache_creation
        totals["cache_read"] += cache_read
        totals["total"] += total_tokens
    print(f"  {'─' * 40}")
    print(f"  all agents:  {totals['input']} in / {totals['output']} out / {totals['cache_creation']} cache creation / {totals['cache_read']} cache read / {totals['total']} total")


if __name__ == "__main__":
    # Parkinson's disease (IDs manually verified):
    # UMLS: C0030567, DOID: DOID:14330, MeSH: D010300
    TEMPLATE_SRC = os.path.expanduser("~/GitHub/alzkb-updater")
    build_disease_kg("Parkinson's disease", TEMPLATE_SRC)
