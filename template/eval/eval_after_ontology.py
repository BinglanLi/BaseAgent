"""
eval_after_ontology.py — Tier 1 metrics computed from the populated RDF.

Metrics implemented (see docs/eval_metrics.md):
  Tier 1: Number of node types, List of node types,
          Number of edge types, List of edge types,
          Ontology mapping activation rate,
          OWL class conformance rate,
          OWL ObjectProperty conformance rate

Output JSON schema (one object per metric):
  name         — metric name from eval_metrics.md
  data_type    — integer | list[str] | float
  tier         — 1
  result       — the computed value

Usage:
    python eval/eval_after_ontology.py
    python eval/eval_after_ontology.py --output report.json
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"

RDF_NS  = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
OWL_NS  = "http://www.w3.org/2002/07/owl#"
RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"
NAMED_INDIVIDUAL = f"{{{OWL_NS}}}NamedIndividual"
RDF_TYPE         = f"{{{RDF_NS}}}type"
RDF_RESOURCE     = f"{{{RDF_NS}}}resource"
RDF_ABOUT        = f"{{{RDF_NS}}}about"


def load_configs() -> tuple[dict, dict]:
    project = yaml.safe_load((CONFIG_DIR / "project.yaml").read_text())["project"]
    mappings_raw = yaml.safe_load((CONFIG_DIR / "ontology_mappings.yaml").read_text())
    mappings = mappings_raw.get("mappings", mappings_raw)
    mappings = {k: v for k, v in mappings.items() if v is not None}
    return project, mappings


def _local_name(iri: str) -> str:
    """Extract the local fragment from an IRI or fragment-only reference."""
    if "#" in iri:
        return iri.split("#")[-1]
    return iri.lstrip("#")


def parse_base_ontology(rdf_path: Path) -> tuple[set[str], set[str], dict[str, dict]]:
    """
    Return (owl_classes, owl_object_props, domain_range) from the base ontology.
    domain_range maps property local name → {"domain": str | None, "range": str | None}.
    """
    tree = ET.parse(rdf_path)
    root = tree.getroot()

    owl_classes: set[str] = set()
    owl_object_props: set[str] = set()
    domain_range: dict[str, dict] = {}

    for child in root:
        about = child.get(RDF_ABOUT, "")
        local = _local_name(about)
        if not local:
            continue

        if child.tag == f"{{{OWL_NS}}}Class":
            owl_classes.add(local)

        elif child.tag == f"{{{OWL_NS}}}ObjectProperty":
            owl_object_props.add(local)
            domain = None
            range_ = None
            for sub in child:
                if sub.tag == f"{{{RDFS_NS}}}domain":
                    domain = _local_name(sub.get(RDF_RESOURCE, ""))
                elif sub.tag == f"{{{RDFS_NS}}}range":
                    range_ = _local_name(sub.get(RDF_RESOURCE, ""))
            domain_range[local] = {"domain": domain, "range": range_}

    return owl_classes, owl_object_props, domain_range


def parse_populated_rdf(rdf_path: Path) -> tuple[set[str], set[str]]:
    """
    Stream-parse the populated RDF and return (node_types, edge_types) found.
    node_types: set of OWL class local names used as rdf:type on NamedIndividuals.
    edge_types: set of ObjectProperty local names with at least one assertion.
    Uses iterparse with start+end events for memory-efficient streaming.
    """
    project, _ = load_configs()
    ns_prefix = project["ontology"]["namespace"].rstrip("/").rstrip("#") + "#"
    ont_ns_tag = f"{{{ns_prefix}}}"  # Clark notation prefix for ontology properties
    node_types: set[str] = set()
    edge_types: set[str] = set()

    # Track whether we're inside a NamedIndividual element so we can process
    # children at their "end" events without waiting for the parent.
    in_individual = False

    for event, elem in ET.iterparse(rdf_path, events=("start", "end")):
        if event == "start":
            if elem.tag == NAMED_INDIVIDUAL:
                in_individual = True
        else:  # event == "end"
            if elem.tag == NAMED_INDIVIDUAL:
                in_individual = False
                elem.clear()
            elif in_individual:
                # Direct child of a NamedIndividual — process it now.
                if elem.tag == RDF_TYPE:
                    resource = elem.get(RDF_RESOURCE, "")
                    if resource.startswith(ns_prefix) or resource.startswith("#"):
                        node_types.add(_local_name(resource))
                elif elem.tag.startswith(ont_ns_tag):
                    if elem.get(RDF_RESOURCE) is not None:
                        edge_types.add(elem.tag.split("}")[-1])
            else:
                # Non-individual element (ObjectProperty/Class declarations, etc.)
                elem.clear()

    return node_types, edge_types


def declared_types(mappings: dict) -> tuple[set[str], set[str]]:
    """Return (declared_node_types, declared_edge_types) from ontology_mappings.yaml."""
    node_types: set[str] = set()
    edge_types: set[str] = set()
    for cfg in mappings.values():
        if cfg.get("data_type") == "node":
            nt = cfg.get("node_type")
            if nt:
                node_types.add(nt)
        else:
            rt = cfg.get("relationship_type")
            if rt:
                edge_types.add(rt)
            inv = cfg.get("inverse_relationship_type")
            if inv:
                edge_types.add(inv)
    return node_types, edge_types


def _metric(name: str, data_type: str, result, tier: int, **kwargs) -> dict:
    entry = {"name": name, "data_type": data_type, "tier": tier, "result": result}
    entry.update({k: v for k, v in kwargs.items() if v is not None})
    return entry


def compute_metrics(
    populated_node_types: set[str],
    populated_edge_types: set[str],
    declared_node_types: set[str],
    declared_edge_types: set[str],
    owl_classes: set[str],
    owl_object_props: set[str],
) -> list[dict]:
    metrics: list[dict] = []

    # --- Number of node types ---
    metrics.append(_metric(
        "Number of node types", "integer", len(populated_node_types), tier=1,
    ))

    # --- List of node types ---
    metrics.append(_metric(
        "List of node types", "list[str]", sorted(populated_node_types), tier=1,
    ))

    # --- Number of edge types ---
    metrics.append(_metric(
        "Number of edge types", "integer", len(populated_edge_types), tier=1,
    ))

    # --- List of edge types ---
    metrics.append(_metric(
        "List of edge types", "list[str]", sorted(populated_edge_types), tier=1,
    ))

    # --- Ontology mapping activation rate ---
    # Fraction of declared types that have at least one individual/triple in the RDF.
    all_declared = declared_node_types | declared_edge_types
    all_populated = populated_node_types | populated_edge_types
    if all_declared:
        activated = all_declared & all_populated
        activation_rate = round(len(activated) / len(all_declared), 4)
    else:
        activation_rate = None
    metrics.append(_metric(
        "Ontology mapping activation rate", "float", activation_rate, tier=1,
        activated=sorted(all_declared & all_populated),
        inactive=sorted(all_declared - all_populated),
    ))

    # --- OWL class conformance rate ---
    # Fraction of declared node_types that exist as OWL classes in the base ontology.
    if declared_node_types and owl_classes:
        valid_nodes = declared_node_types & owl_classes
        class_conf_rate = round(len(valid_nodes) / len(declared_node_types), 4)
    else:
        class_conf_rate = None
    metrics.append(_metric(
        "OWL class conformance rate", "float", class_conf_rate, tier=1,
        non_conformant_node_types=sorted(declared_node_types - owl_classes) or None,
    ))

    # --- OWL ObjectProperty conformance rate ---
    if declared_edge_types and owl_object_props:
        valid_edges = declared_edge_types & owl_object_props
        prop_conf_rate = round(len(valid_edges) / len(declared_edge_types), 4)
    else:
        prop_conf_rate = None
    metrics.append(_metric(
        "OWL ObjectProperty conformance rate", "float", prop_conf_rate, tier=1,
        non_conformant_edge_types=sorted(declared_edge_types - owl_object_props) or None,
    ))

    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute after-ontology-population metrics from the populated RDF."
    )
    ap.add_argument("--output", metavar="FILE", help="Write JSON to FILE (default: stdout)")
    args = ap.parse_args()

    project, mappings = load_configs()

    populated_rdf = ROOT / project["ontology"]["populated_output"]
    base_rdf = ROOT / project["ontology"]["base_file"]

    if not populated_rdf.exists():
        print(f"ERROR: populated RDF not found: {populated_rdf}", flush=True)
        sys.exit(1)
    if not base_rdf.exists():
        print(f"ERROR: base ontology not found: {base_rdf}", flush=True)
        sys.exit(1)

    print(f"Parsing base ontology: {base_rdf}", flush=True)
    owl_classes, owl_object_props, _ = parse_base_ontology(base_rdf)

    print(f"Streaming populated RDF: {populated_rdf}", flush=True)
    populated_node_types, populated_edge_types = parse_populated_rdf(populated_rdf)

    decl_node_types, decl_edge_types = declared_types(mappings)

    metrics = compute_metrics(
        populated_node_types, populated_edge_types,
        decl_node_types, decl_edge_types,
        owl_classes, owl_object_props,
    )

    report = {
        "run_timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "metrics": metrics,
    }
    output = json.dumps(report, indent=2, default=str)

    if args.output:
        Path(args.output).write_text(output)
        print(f"Report written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
