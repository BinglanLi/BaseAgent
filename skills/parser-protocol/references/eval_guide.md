# Evaluator Guide

## Running

```bash
python test/eval_parser.py --parser <source_name>   # single source
python test/eval_parser.py --all                     # all sources
python test/eval_parser.py --all --format json --output eval.json
```

## Status Values

| Status | Meaning |
|--------|---------|
| `PASS` | Check passed |
| `FAIL ⚠` | Check failed; inspect the detail message |
| `N/A` | Check not applicable (output not yet generated, or key absent from `get_schema()`) |

---

## Check Categories

### Extraction Validity

| Check | What it detects |
|-------|----------------|
| TSV exists | File was created by `export_tsv()` |
| Record count | Non-zero rows after any filter; below 100 triggers a warning |
| Bad rows | Rows dropped due to field-count mismatch (embedded tab characters) |
| Header integrity | `\r` characters in column names (causes silent lookup failure in the populator) |

### Data Format & Schema Validity

| Check | What it detects |
|-------|----------------|
| Required columns present | IRI column, data property columns, and match columns named in the mapping config all exist in the TSV |
| IRI non-null rate | Fraction of rows where the IRI column is non-null; below 1.0 = missing identifiers → unnamed individuals |
| IRI duplicate count | Duplicate IRI values in the filtered row set |
| Schema integrity | TSV columns vs. `get_schema()` — extra columns = stale schema; missing columns = `parse_data()` dropped a field |

**Schema integrity FAIL**: update `get_schema()` to exactly match the current output of `parse_data()`.

### Ontology Adherence

| Check | What it detects |
|-------|----------------|
| Node type valid | `node_type` in the mapping exists in `project.yaml` `node_types` list |
| Edge type valid | `relationship_type` in the mapping exists in `project.yaml` `edge_types` list |

Failures indicate a type name not yet added to `project.yaml` `node_types` / `edge_types` lists.

### Mapping Integrity (requires full pipeline run)

| Check | What it detects |
|-------|----------------|
| Output node count | Nodes in `nodes_{Type}.csv`; `N/A` if pipeline hasn't run |
| Merge eligible/match rate | For `merge: true` entries, fraction of rows with a non-null merge key and fraction matched |
| Output edge count | Edges in `edges_{Rel}.csv` |
| Resolution rate | `output_edge_count / source_row_count` |
| Unresolved count | Rows that produced no edge |

**Resolution rate < 1.0**: subject or object nodes not found. Most common cause: the relationship entry was processed before its nodes in `ontology_mappings.yaml` (all node entries must precede relationship entries).

**Resolution rate > 1.0**: the populator expanded rows into multiple edges (one-to-many join in the parser). Normal and expected.

---

## Registering a New Source

Add to `PARSER_CLASS_MAP` in `test/eval_parser.py`:

```python
PARSER_CLASS_MAP: dict[str, tuple[str, str]] = {
    ...
    "mysource": ("parsers.mysource_parser", "MySourceParser"),
}
```

The key must equal the `databases.yaml` key for the source, which also determines the `data/processed/` subdirectory name. It is independent of any `self.source_name` override — that override only affects `data/raw/`.

---

## Orphan Outputs

TSV files under `data/processed/<source>/` with no matching entry in `ontology_mappings.yaml` are flagged as "Orphan Outputs." They may indicate extra parser output not yet mapped, a stale TSV from a previous parser version, or an accidentally deleted mapping entry.
