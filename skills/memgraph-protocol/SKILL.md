---
name: memgraph-protocol
description: Use when running the graph export step, inspecting or validating CSV and Cypher outputs, importing the knowledge graph into Memgraph via Docker, or extending MemgraphExporter. Covers running the exporter in isolation, output file formats (nodes_*.csv, edges_*.csv, import.cypher), Docker volume mount, Cypher script validation, and known constraints (string-only values, global node ID uniqueness, graph_indexes is informational).
---

## Prerequisite

Confirm `data/output/ontology_populated.rdf` exists before proceeding. The exporter reads this file; if it is absent, run the full pipeline first (`python src/main.py`) or the populate step in isolation.

---

## Running the Exporter

**Isolated run** (populated RDF already exists):

```python
from src.export.memgraph_exporter import MemgraphExporter

exporter = MemgraphExporter(
    rdf_files=["data/output/ontology_populated.rdf"],
    output_dir="data/output",
)
result = exporter.export()
# result["cypher_script"] is a path string to import.cypher, not the file content
```

`rdf_files` is a list — pass multiple RDF files if the graph spans more than one populated ontology file.

**Full pipeline** (runs populate then export):
```bash
python src/main.py
```

---

## Outputs

All files are written to `data/output/`:

| File | Content |
|------|---------|
| `nodes_{NodeType}.csv` | One file per OWL class; columns: `id`, data properties in alphabetical order, then `uri` as the final column. `uri` is archival and is excluded from the Cypher `CREATE` statement. |
| `edges_{RelType}.csv` | One file per OWL object property; columns: `start_id`, `end_id`, `start_uri`, `end_uri` |
| `import.cypher` | Cypher LOAD CSV script for Memgraph |

**All CSV values are strings.** Use `ToInteger()` / `ToFloat()` in Cypher for numeric comparisons.

---

## Importing into Memgraph

Mount `data/output/` to `/import-data/` inside the container:

```bash
docker run -v /abs/path/to/data/output:/import-data memgraph/memgraph-platform
```

Then paste or load `import.cypher` in Memgraph Lab. The `/import-data/` prefix in `LOAD CSV` paths is hardcoded by the exporter and must match this mount point.

---

## Validating import.cypher

After export, verify `import.cypher` before running in Memgraph:

- Each node type has both `CREATE INDEX ON :NodeType;` and `CREATE INDEX ON :NodeType(id);`.
- `LOAD CSV` paths use the `/import-data/` prefix.
- Node `CREATE` statements include all expected data properties. `uri` is present in the CSV but intentionally absent from the Cypher `CREATE` — do not add it.
- Edge `MATCH` clauses are label-agnostic: `MATCH (a {id: row.start_id})` — correct only when node IDs are globally unique across types (see below).

---

## Known Constraints

**Global node ID uniqueness** — The Cypher script uses label-agnostic `MATCH` to resolve edge endpoints. This relies on every individual's local name (the IRI fragment, e.g., `gene_7157`) being unique across all node types. Individual IRIs are assigned by ista at populate time using type-specific prefixes (`gene_*`, `drug_*`, `disease_*`), which ensures uniqueness in practice. If a new node type could produce IRI collisions with an existing type, establish a distinct prefix in the ontology before running export.

**Multi-type individuals** — An individual that belongs to multiple OWL classes appears in every corresponding node CSV. Row counts across node files are not mutually exclusive; do not use them to infer total unique individual counts.

**Stale file overwrite** — The exporter silently overwrites all existing CSV files in `output_dir` with no warning. After re-export, every CSV reflects the current run only — no append mode, no stale-file detection.

**`graph_indexes` is informational** — `project.yaml` `graph_indexes` lists properties intended for indexing per node type, but `MemgraphExporter` does not read it. The exporter generates only a label index and an `id` index per node type. To add property indexes, append `CREATE INDEX ON :NodeType(prop);` statements to `import.cypher` manually, or extend `_write_cypher_script()` in `src/export/memgraph_exporter.py`.

---

## Extending the Exporter

`MemgraphExporter` is self-contained in `src/export/memgraph_exporter.py`. Key methods:

| Method | Purpose |
|--------|---------|
| `_extract_nodes(ontology_ns)` | Groups `owl:NamedIndividual` triples by OWL class; collects literal data properties |
| `_extract_edges(ontology_ns)` | Groups object property assertions between individuals by property local name |
| `_write_node_csv(filepath, nodes, node_type)` | Writes node CSV; returns property column list (used by `_write_cypher_script`) |
| `_write_edge_csv(filepath, edges, rel_type)` | Writes edge CSV |
| `_write_cypher_script(node_columns, rel_types)` | Generates `import.cypher` |

The exporter uses `rdflib` for RDF parsing — no OWL-specific library. Namespace detection reads the `owl:Ontology` IRI from the RDF, falling back to the most common individual namespace if absent.
