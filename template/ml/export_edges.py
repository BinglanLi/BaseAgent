"""
Export edge list from Memgraph for Node2Vec training.

Exports all edges as (source_id, target_id, rel_type) triples,
mapping each node to a unique integer ID for Node2Vec input.
Also exports node metadata (id, label, name) for interpreting results.
"""

import os
import sys
import csv
import json
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

MEMGRAPH_URI = os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")
MEMGRAPH_USER = os.getenv("MEMGRAPH_USERNAME", "")
MEMGRAPH_PASS = os.getenv("MEMGRAPH_PASSWORD", "")

OUTPUT_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)


def get_driver():
    return GraphDatabase.driver(MEMGRAPH_URI, auth=(MEMGRAPH_USER, MEMGRAPH_PASS))


def export_edges(driver):
    """Export all edges with node integer IDs."""
    node_map = {}
    next_id = [0]

    def get_id(node_id):
        if node_id not in node_map:
            node_map[node_id] = next_id[0]
            next_id[0] += 1
        return node_map[node_id]

    edge_path = OUTPUT_DIR / "edges.tsv"
    rel_type_counts = defaultdict(int)

    print("Exporting edges from Memgraph...")
    with driver.session() as session:
        result = session.run(
            "MATCH (a)-[r]->(b) "
            "RETURN id(a) AS src, id(b) AS dst, type(r) AS rel_type"
        )

        with open(edge_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["src", "dst", "rel_type"])
            count = 0
            for record in result:
                src = get_id(record["src"])
                dst = get_id(record["dst"])
                rel = record["rel_type"]
                writer.writerow([src, dst, rel])
                rel_type_counts[rel] += 1
                count += 1
                if count % 500_000 == 0:
                    print(f"  ...{count:,} edges exported")

    print(f"Exported {count:,} edges to {edge_path}")
    print(f"Unique nodes: {len(node_map):,}")
    print(f"\nEdge type counts:")
    for rel, c in sorted(rel_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {rel}: {c:,}")

    return node_map, rel_type_counts


def export_node_metadata(driver, node_map):
    """Export node metadata: integer ID, label, display name."""
    meta_path = OUTPUT_DIR / "nodes.tsv"
    id_to_int = {v: k for k, v in node_map.items()}
    int_to_memgraph_id = {v: k for k, v in node_map.items()}

    print("\nExporting node metadata...")
    node_rows = []

    with driver.session() as session:
        result = session.run(
            "MATCH (n) "
            "RETURN id(n) AS nid, labels(n) AS labels, "
            "coalesce(n.commonName, n.geneSymbol, n.diseaseName, "
            "n.trialId, n.pathwayName, n.variantId, n.phenotypeName, "
            "n.bodyPartName, n.sideEffectName, n.symptomName, "
            "n.processName, n.functionName, n.componentName, "
            "n.familyName, n.className, n.labelName, n.tfSymbol, "
            "'unknown') AS name"
        )
        for record in result:
            nid = record["nid"]
            if nid in node_map:
                labels = record["labels"]
                label = labels[0] if labels else "Unknown"
                node_rows.append({
                    "int_id": node_map[nid],
                    "label": label,
                    "name": record["name"],
                    "memgraph_id": nid,
                })

    node_rows.sort(key=lambda x: x["int_id"])
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["int_id", "label", "name", "memgraph_id"], delimiter="\t")
        writer.writeheader()
        writer.writerows(node_rows)

    print(f"Exported {len(node_rows):,} node metadata rows to {meta_path}")

    label_counts = defaultdict(int)
    for row in node_rows:
        label_counts[row["label"]] += 1
    print(f"\nNode label counts:")
    for label, c in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label}: {c:,}")

    return node_rows


def export_edgelist_plain(node_map):
    """Write plain (src dst) edgelist for Node2Vec input (no header, no rel_type)."""
    edgelist_path = OUTPUT_DIR / "edgelist.txt"
    edges_path = OUTPUT_DIR / "edges.tsv"

    print(f"\nWriting plain edgelist to {edgelist_path}...")
    count = 0
    with open(edges_path) as fin, open(edgelist_path, "w") as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        for row in reader:
            fout.write(f"{row['src']}\t{row['dst']}\n")
            count += 1

    print(f"Wrote {count:,} edges to {edgelist_path}")


def save_summary(node_map, rel_type_counts):
    """Save export summary as JSON for downstream scripts."""
    summary = {
        "num_nodes": len(node_map),
        "num_edges": sum(rel_type_counts.values()),
        "rel_type_counts": dict(rel_type_counts),
    }
    summary_path = OUTPUT_DIR / "export_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


def main():
    driver = get_driver()
    try:
        node_map, rel_type_counts = export_edges(driver)
        export_node_metadata(driver, node_map)
        export_edgelist_plain(node_map)
        save_summary(node_map, rel_type_counts)
        print("\nEdge list export complete.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
