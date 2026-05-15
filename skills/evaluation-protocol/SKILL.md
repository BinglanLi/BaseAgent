---
name: evaluation-protocol
description: Use when running, interpreting, or extending the KG build evaluation suite. Covers the three pipeline-stage eval scripts (eval_after_parser.py, eval_after_ontology.py, eval_after_memgraph.py), output JSON format, the three-tier metric system, blocking vs. monitoring thresholds, and adding new metrics. Use when asked to evaluate pipeline output, diagnose zero-count or low-resolution failures, check ontology conformance, run benchmarks, or interpret eval reports.
---

## Eval Scripts by Pipeline Stage

Run each script after its prerequisite pipeline step completes.

| Script | Prerequisite | Input files |
|--------|-------------|-------------|
| `eval/eval_after_parser.py` | `python src/main.py` (all enabled sources) | `data/processed/<source>/<name>.tsv` |
| `eval/eval_after_ontology.py` | populate step | `data/output/ontology_populated.rdf`, `data/ontology/ontology.rdf` |
| `eval/eval_after_memgraph.py` | export step | `data/output/nodes_*.csv`, `data/output/edges_*.csv` |

All scripts auto-load config from `config/project.yaml`, `config/databases.yaml`, and `config/ontology_mappings.yaml`.

---

## CLI

```bash
# After parser step — all sources
python eval/eval_after_parser.py
python eval/eval_after_parser.py --output report.json

# After ontology population
python eval/eval_after_ontology.py
python eval/eval_after_ontology.py --output report.json

# After Memgraph export — full suite
python eval/eval_after_memgraph.py
python eval/eval_after_memgraph.py --output report.json

# After Memgraph export — with optional inputs
python eval/eval_after_memgraph.py --baseline prev_report.json   # enables run-to-run delta (Tier 2)
python eval/eval_after_memgraph.py --omim-genemap genemap2.txt   # enables disease-gene recall (Tier 3)
python eval/eval_after_memgraph.py --drugbank-tsv approved.tsv   # enables drug-target coverage (Tier 3)
```

---

## Output JSON Schema

All `eval_after_*.py` scripts write a JSON object:

```json
{
  "run_timestamp": "2025-04-27T12:00:00+00:00",
  "metrics": [ ... ]
}
```

`eval_after_memgraph.py` also includes `"entity_counts": {"Gene": 123, ...}` at the top level (used by `--baseline` comparison).

Each element of `metrics`:

```json
{
  "name": "metric name matching eval_metrics.json",
  "data_type": "binary | integer | float | date | object | list[str]",
  "tier": 1,
  "result": <computed value>,
  "source": "disgenet",
  "mapping": "disgenet.gene_disease_associations",
  "note": "optional explanation"
}
```

- `source` and `mapping` are absent from `eval_after_ontology.py` output — that script operates on the whole ontology, not per-source.
- Additional keys vary by metric (e.g., `column`, `ontology_property`, `source_rows`, `resolved_rows`, `total_omim_entries`, `recalled`, `duplicate_edge_count`).

---

## Tier System

| Tier | Label | Action |
|------|-------|--------|
| 1 | Block Release | Zero Tier 1 node/edge counts or failed OWL conformance block release |
| 2 | Monitor Trends | Track across runs; investigate regressions |
| 3 | Periodic Audit | Scheduled checks; some require optional external files |

**Blocking failures (Tier 1)** — convention only, no script enforces a non-zero exit code:
- Any node or edge count = 0 for an expected type
- OWL class or ObjectProperty conformance rate < 1.0
- Domain/range constraint violations > 0

**Zero is not always bad**: `result = 0` for filter pass rate means the filter dropped all rows — likely a schema change upstream, not a correct result.

---

## Interpretation Rules

**Resolution rate** (`eval_after_memgraph.py`):
- = rows where both subject AND object identifiers matched a graph node / filtered TSV rows (bounded 0–1)
- `0.0` → silent join failure: check source name consistency and node-before-relationship ordering in `ontology_mappings.yaml`
- `result: null` for merge match rate → merge property column absent from output CSV; indicates a mapping or export configuration error

**Run-to-run delta** (Tier 2):
- Requires `--baseline prev_report.json` pointing to a previous `eval_after_memgraph.py` JSON output
- Non-zero delta on identical source versions = non-determinism

**Tier 3 biological benchmarks**:
- Disease-gene recall requires `genemap2.txt` from OMIM; a pair is recalled only when both the Entrez Gene ID (column 9) and the OMIM phenotype MIM number (column 12, type-3 entries) resolve to nodes in the graph
- Drug-target coverage requires a DrugBank TSV with a `drugbank_id` column (approved drugs only scope must be specified)
- Pin benchmark file versions for reproducibility

---

## References

- [references/eval_metrics.md](references/eval_metrics.md) — full metric specs organized by pipeline stage and tier (narrative); authoritative
- [references/eval_metrics.json](references/eval_metrics.json) — same catalog in machine-readable form; use when implementing a report parser or adding new metrics
