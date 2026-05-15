---
name: supervisor-protocol
description: Use when coordinating across pipeline modules — understanding which config files own which concerns, what contracts must hold between parsers, mappings, ontology, and graph export, what causes silent failures, or how to integrate a new data source end-to-end. Covers cross-module constraints, source name consistency rules, the new-source checklist, and known failure modes that produce no error but wrong output.
---

## Pipeline Data Flow

```
databases.yaml + .env
       ↓
   BaseParser (download + parse) → data/raw/<source>/
       ↓
   DataFrame dict → export_tsv → data/processed/<source>/<name>.tsv
       ↓
   OntologyPopulator (ontology_mappings.yaml + project.yaml) → data/output/ontology_populated.rdf
       ↓
   MemgraphExporter → data/output/nodes_*.csv, edges_*.csv, import.cypher
```

---

## Config File Ownership

| File | Owns |
|------|------|
| `config/databases.yaml` | Which parsers run; constructor args and credentials |
| `config/project.yaml` | Published OWL name table (active node/edge types); disease scope; ontology paths |
| `config/ontology_mappings.yaml` | TSV column → OWL property mappings; node/relationship entry order |

Changing a `databases.yaml` key breaks: `data/processed/` subdirectory, `ontology_mappings.yaml` prefix, and `PARSERS` dict in `src/main.py` — all must be updated together.

---

## Cross-Module Contracts

These six rules must hold for the pipeline to produce correct output. Violations fail silently.

**1. Source name consistency** — `databases.yaml` key = `PARSERS` key (main.py) = `ontology_mappings.yaml` entry prefix = `data/processed/<source>/` subdirectory name. All four must be identical strings. Note: `BaseParser.source_name` is derived from the class name (`ClassName.replace('Parser','').lower()`), which controls `data/raw/<classname>/` and may differ from the databases.yaml key — this is an expected split (e.g., `MEDLINECooccurrenceParser` → raw dir `medlinecooccurrence`, processed dir `medline_cooccurrence`).

**2. TSV filename stems** — Keys in `parse_data()` return dict become TSV filename stems (e.g., key `"gene_disease"` → `gene_disease.tsv`). Each `source_filename` in `ontology_mappings.yaml` must exactly match one of these stems.

**3. Column name agreement** — Every column name referenced in `ontology_mappings.yaml` (`iri_column_name`, `subject_column_name`, `object_column_name`, `data_property_map` keys, `merge_column.source_column_name`, `filter_column`) must appear as a column in the corresponding TSV. `get_schema()` must exactly match what `parse_data()` produces.

**4. Node-before-relationship ordering** — In `ontology_mappings.yaml`, all node entries must precede all relationship entries. The populate step resolves relationships by matching against already-loaded individuals; a relationship entry processed before its subject or object type exists produces zero edges with no error.

**5. OWL name validity** — `node_type` and `relationship_type` values in `ontology_mappings.yaml` must be OWL classes/properties that exist in `data/ontology/ontology.rdf` AND appear as active (uncommented) entries in `project.yaml` `node_types`/`edge_types`. Data property names (in `data_property_map`, `*_match_property`, `merge_column`) are not in `project.yaml` — verify them against the ontology or existing mapping entries for the same class.

**6. `_env` credential injection** — `_resolve_env_vars()` in `main.py` strips `_env` suffix and replaces value with `os.environ.get(VAR)`. If the env var is unset, the value becomes `None` and a WARNING is logged — no hard error. The parser constructor must declare the stripped parameter name. Only add `_env` keys when the env var is guaranteed present.

---

## Known Silent Failures

| Symptom | Root cause |
|---------|------------|
| Zero edges for a source | Relationship entry precedes node entry in `ontology_mappings.yaml` |
| Populate loads 0 rows | Source name mismatch across configs (contract 1) |
| TSV column not found at populate | Column name mismatch between parser and mappings (contract 3) |
| Credential silently None | `_env` key added but env var not set in `.env` |
| OWL property not resolved | Name in mappings doesn't match active entry in `project.yaml` |
| Stale graph output | `MemgraphExporter` silently overwrites without warning |

`validate_config()` on `OntologyPopulator` checks contracts 3 and 5 partially but does NOT check `merge_column.data_property`. It is never called automatically — must be run manually before a full pipeline run.

---

## New Source Integration Checklist

Complete all steps in order before running the full pipeline:

1. Create parser class extending `BaseParser` in `src/parsers/<source>_parser.py` — implement `download_data()`, `parse_data()`, `get_schema()`
2. Add import and add to `__all__` in `src/parsers/__init__.py`
3. Register in `PARSERS` in `src/main.py`: `"<databases_yaml_key>": ClassName`
4. Add entry to `config/databases.yaml` with `enabled: true` and any required args
5. Add credentials to `.env` if the parser uses `_env` keys
6. Add `ontology_mappings.yaml` entries — node entries first, then relationship entries
7. Activate any new OWL class or property names in `config/project.yaml` `node_types` / `edge_types`

After step 4, run `python src/main.py --source <key>` to produce TSVs.

---

## References

- [references/module_apis.md](references/module_apis.md) — BaseParser, OntologyPopulator, MemgraphExporter method signatures and return contracts
- [references/cli_and_outputs.md](references/cli_and_outputs.md) — CLI usage, output file formats, running steps in isolation
