"""
Agent tool: inspect an OWL RDF ontology.

Import and call these functions directly — each returns a dict with:
  name, description, results

Or run as a CLI script:
  python3 scripts/inspect_ontology.py <rdf_path> [options]

Options:
  --type {classes,object_properties,datatype_properties,all}
                        Which declaration type to list (default: all)
  --crossref <yaml>     Cross-reference against a project.yaml
  -h, --help            Show this help message and exit

Examples:
  python3 scripts/inspect_ontology.py data/ontology/ontology.rdf --type classes
  python3 scripts/inspect_ontology.py data/ontology/ontology.rdf --crossref config/project.yaml

Exit codes: 0 success, 1 error.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _ontology_utils import _extract_names, _read_rdf


# ---------------------------------------------------------------------------
# project.yaml helpers
# ---------------------------------------------------------------------------

def _parse_yaml_list_all(yaml_text: str, key: str) -> list[str]:
    """Return all items from a YAML list under key, including commented-out entries."""
    items: list[str] = []
    in_section = False
    for line in yaml_text.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{key}:"):
            in_section = True
            continue
        if in_section:
            if stripped.startswith("- "):
                items.append(stripped[2:].split("#")[0].strip())
            elif stripped.startswith("# - "):
                items.append(stripped[4:].split("#")[0].strip())
            elif stripped and not stripped.startswith("#") and ":" in stripped:
                break
    return items


# ---------------------------------------------------------------------------
# Public agent functions
# ---------------------------------------------------------------------------

def get_classes(ontology_path: str | Path) -> dict:
    """Return all owl:Class local names defined in the ontology."""
    return {
        "name": "get_classes",
        "description": "All OWL classes (node types) defined in the ontology.",
        "results": _extract_names(_read_rdf(ontology_path), "owl:Class"),
    }


def get_object_properties(ontology_path: str | Path) -> dict:
    """Return all owl:ObjectProperty local names defined in the ontology."""
    return {
        "name": "get_object_properties",
        "description": "All OWL object properties (edge types) defined in the ontology.",
        "results": _extract_names(_read_rdf(ontology_path), "owl:ObjectProperty"),
    }


def get_datatype_properties(ontology_path: str | Path) -> dict:
    """Return all owl:DatatypeProperty local names defined in the ontology."""
    return {
        "name": "get_datatype_properties",
        "description": "All OWL datatype properties (node and edge scalar attributes) defined in the ontology.",
        "results": _extract_names(_read_rdf(ontology_path), "owl:DatatypeProperty"),
    }


def crossref_project_yaml(ontology_path: str | Path, project_yaml_path: str | Path) -> dict:
    """
    Cross-reference the ontology against project.yaml (active + inactive entries).

    Returns four lists:
      - classes_not_in_yaml:    RDF classes absent from node_types
      - yaml_nodes_not_in_rdf:  node_types entries absent from the RDF
      - properties_not_in_yaml: RDF object properties absent from edge_types
      - yaml_edges_not_in_rdf:  edge_types entries absent from the RDF
    """
    rdf_text = _read_rdf(ontology_path)
    rdf_classes = set(_extract_names(rdf_text, "owl:Class"))
    rdf_edges = set(_extract_names(rdf_text, "owl:ObjectProperty"))

    yaml_text = Path(project_yaml_path).read_text()
    yaml_nodes = set(_parse_yaml_list_all(yaml_text, "node_types"))
    yaml_edges = set(_parse_yaml_list_all(yaml_text, "edge_types"))

    return {
        "name": "crossref_project_yaml",
        "description": (
            "Cross-reference between ontology.rdf and project.yaml (all entries, including inactive). "
            "Gaps indicate names that need to be added or removed."
        ),
        "results": {
            "classes_not_in_yaml": sorted(rdf_classes - yaml_nodes),
            "yaml_nodes_not_in_rdf": sorted(yaml_nodes - rdf_classes),
            "properties_not_in_yaml": sorted(rdf_edges - yaml_edges),
            "yaml_edges_not_in_rdf": sorted(yaml_edges - rdf_edges),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect an OWL RDF ontology. Outputs JSON to stdout.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 inspect_ontology.py data/ontology/ontology.rdf --type classes\n"
            "  python3 inspect_ontology.py data/ontology/ontology.rdf --crossref config/project.yaml"
        ),
    )
    parser.add_argument("rdf", type=Path, help="Path to ontology.rdf")
    parser.add_argument(
        "--type",
        choices=["classes", "object_properties", "datatype_properties", "all"],
        default="all",
        help="Which declaration type to list (default: all)",
    )
    parser.add_argument("--crossref", type=Path, default=None, metavar="YAML",
                        help="Cross-reference against a project.yaml")
    args = parser.parse_args()

    if not args.rdf.exists():
        print(f"Error: RDF file not found: {args.rdf}", file=sys.stderr)
        sys.exit(1)

    if args.crossref and not args.crossref.exists():
        print(f"Error: project.yaml not found: {args.crossref}", file=sys.stderr)
        sys.exit(1)

    results = {}

    if args.crossref:
        results = crossref_project_yaml(args.rdf, args.crossref)
    elif args.type == "all":
        results = {
            "classes": get_classes(args.rdf)["results"],
            "object_properties": get_object_properties(args.rdf)["results"],
            "datatype_properties": get_datatype_properties(args.rdf)["results"],
        }
    elif args.type == "classes":
        results = get_classes(args.rdf)
    elif args.type == "object_properties":
        results = get_object_properties(args.rdf)
    elif args.type == "datatype_properties":
        results = get_datatype_properties(args.rdf)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
