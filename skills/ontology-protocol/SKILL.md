---
name: ontology-protocol
description: Use when enforcing OWL ontology terms, adding or modifying OWL classes and object/data properties in the RDF, updating node_types or edge_types in project.yaml, or managing disease scope. This skill owns data/ontology/ontology.rdf and config/project.yaml. It does not manage ontology_mappings.yaml — that is mapping-protocol's scope. Does not require knowledge of parser internals or Memgraph.
---

You manage the OWL schema and the project configuration:

| File | Your responsibility |
|------|-------------------|
| `data/ontology/ontology.rdf` | OWL schema: classes, object properties, data properties |
| `config/project.yaml` | `node_types`, `edge_types`, `disease_scope`, ontology paths |

**Strict constraints**:
- Only modify `data/ontology/ontology.rdf` on explicit user request.
- Never edit Python source files.

---

## project.yaml as the Published OWL Name Table

`project.yaml` `node_types` and `edge_types` are the single source of truth for valid OWL names available to the rest of the pipeline:
- `mapping-protocol` reads them to confirm a class or property name is valid before writing a mapping entry.

Keeping these lists in sync with the RDF is the primary ongoing responsibility of this skill. A name present in the RDF but absent from `project.yaml` is invisible to the rest of the pipeline.

Active entries are uncommented. Inactive entries (defined in ontology but no data source produces them yet) are commented out — preserve them.

---

## Adding a New OWL Class or Property

1. Edit `data/ontology/ontology.rdf`: add the OWL class or object/data property.
2. Add the name to `project.yaml` under `node_types` (for classes) or `edge_types` (for object properties).

OWL names are case-sensitive. The name in `project.yaml` must exactly match the local name in the RDF (the fragment after `#` in the IRI). Data properties do not appear in `node_types`/`edge_types` — they are referenced directly from `ontology_mappings.yaml` and resolved at populate time.

---

## Activating an Existing OWL Class or Property

Uncomment the entry in `node_types` or `edge_types` in `project.yaml`. No RDF change needed — the class or property already exists in the ontology.

---

## Disease Scope

See [references/project_yaml_format.md](references/project_yaml_format.md) for `disease_scope` field semantics and which parsers consume each field.
