---
name: mapping-protocol
description: Use when adding, modifying, or fixing entries in config/ontology_mappings.yaml. Maps parser TSV output columns to OWL node types, data properties, and relationship types. Covers config key format, node and relationship entry schemas, the node-before-relationship ordering constraint, merge semantics, filter patterns, skip flag, and pre-flight validation. Requires reading config/project.yaml to confirm valid OWL names before writing entries. Does not modify the OWL RDF file or Python source files.
---

You own `config/ontology_mappings.yaml`. You map parsed TSV columns to OWL types and properties so the populate step can create individuals and relationships in the knowledge graph.

**Strict constraints**:
- Only use OWL class names that appear in `project.yaml` `node_types` and OWL object property names that appear in `project.yaml` `edge_types`. Data properties (used in `data_property_map`, `merge_column`, and `*_match_property`) are not enumerated in `project.yaml` — verify them by inspecting existing entries that reference the same ontology class.
- If `merge` is `true`, there must be a `merge_column` with a valid `data_property` that exists in the ontology and is used by other entries for the same class.
- If a required class or object property is absent from `project.yaml`, stop and report the missing name. Do not propose changes to the RDF.
- Never edit Python source files.

---

## Pre-Editing Checklist

All string names in `ontology_mappings.yaml` are resolved to live OWL objects at populate time via `getattr(ontology, name)`. A wrong name silently produces no output — no runtime error.

Before writing any entry:
1. Read `config/project.yaml` `node_types` to confirm the node type is valid.
2. Read `config/project.yaml` `edge_types` to confirm the relationship type is valid.
3. Read `data/processed/<source>/<output>.tsv` (or the parser's `get_schema()`) to confirm TSV column names.
4. Verify data property names by checking existing entries that reference the same ontology class — there is no enumeration of data properties in `project.yaml`; they are resolved directly from the ontology.

---

## Config Key Format

Keys use `{source_name}.{output_name}`:
- `source_name` must match the `databases.yaml` key and the `data/processed/<source_name>/` subdirectory.
- `output_name` must match the TSV filename stem (without `.tsv`).

Example: `disgenet.gene_disease_associations` → `data/processed/disgenet/gene_disease_associations.tsv`.

---

## Critical Ordering Rule

**All node entries must precede all relationship entries.**

The populate step resolves relationships by matching against already-loaded individuals. A relationship entry processed before its subject or object node type exists produces zero edges with no error. Always place new node entries in the `NODE POPULATIONS` section and new relationship entries in the `RELATIONSHIP POPULATIONS` section.

---

## Skip Flag

Use `skip: true` for planned but unimplemented entries:
```yaml
sider.side_effect_nodes:
  data_type: node
  skip: true
  ...rest of entry...
```
Skipped entries are reported as `N/A` rather than failing. This documents future mappings without executing them.

A skipped entry may reference a `node_type` or `relationship_type` that is currently commented out in `project.yaml` (i.e., inactive). Do not remove `skip: true` until the corresponding type is activated in `project.yaml`.

---

## Pre-flight Validation

Run after any edit before a full pipeline run:

```python
from src.ontology.populator import OntologyPopulator
from src.main import load_config

project, databases, mappings = load_config()
populator = OntologyPopulator(
    ontology_path="data/ontology/ontology.rdf",
    data_dir="data/processed",
    ontology_mappings=mappings,
)
errors = populator.validate_config("source.output_name", mappings["source.output_name"])
for e in errors: print(e)
```

`validate_config()` checks `node_type`, `relationship_type`, `*_node_type`, `*_match_property`, and `data_property_map` values. It does **not** check `merge_column.data_property` — verify that field manually. The pipeline does not call `validate_config()` automatically.

---

## Entry Schemas

See [references/entry_format.md](references/entry_format.md) for annotated node and relationship entry schemas, merge semantics, filter patterns, compound_fields, and inverse relationships.
