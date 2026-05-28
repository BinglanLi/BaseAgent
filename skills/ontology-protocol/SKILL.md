---
name: ontology-protocol
tools:
  - inspect_ontology.py
  - edit_ontology.py
description: Use when enforcing OWL ontology terms, adding or modifying OWL classes and object/data properties in the RDF, updating node_types or edge_types in project.yaml, or managing disease scope. Provides scripts/inspect_ontology.py to verify valid names and scripts/edit_ontology.py to add or remove declarations while keeping project.yaml in sync. Owns data/ontology/ontology.rdf and config/project.yaml. Does not manage ontology_mappings.yaml (mapping-protocol's scope) or Python source files.
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

## Activating an Existing OWL Class or Property

Uncomment the entry in `node_types` or `edge_types` in `project.yaml`. No RDF change needed — the class or property already exists in the ontology.

---


## Reference Tools

### Checking valid names — `scripts/inspect_ontology.py`

Before writing any mapping or parser entry, confirm the class or property name exists in the ontology:

```python
from scripts.inspect_ontology import get_classes, get_object_properties, get_datatype_properties, crossref_project_yaml

crossref_project_yaml("data/ontology/ontology.rdf", "config/project.yaml")
```

Run `crossref_project_yaml` after any ontology or `project.yaml` edit to verify they are in sync. Use `get_classes`, `get_object_properties`, and `get_datatype_properties` to list valid names by type.

### Adding or removing declarations — `scripts/edit_ontology.py`

Use when a required class or property does not yet exist in the RDF. Each function writes the RDF and, when `project_yaml_path` is provided, updates `project.yaml` in the same call. Pass `dry_run=True` to preview without writing.

```python
from scripts.edit_ontology import add_class, add_object_property, add_datatype_property

add_class("data/ontology/ontology.rdf", "MyNode", parent="GeneticEntity",
          project_yaml_path="config/project.yaml")

add_object_property("data/ontology/ontology.rdf", "myRelatesTo",
                    domain="Gene", range_="Disease",
                    project_yaml_path="config/project.yaml", active=False)
```

Declarations are inserted alphabetically. `active=False` adds a commented-out inactive entry; the default is active. Datatype properties are not listed in `project.yaml` — only the RDF is updated. Run `python3 scripts/edit_ontology.py --help` for full CLI usage.

---

## Disease Scope

See [references/project_yaml_format.md](references/project_yaml_format.md) for `disease_scope` field semantics and which parsers consume each field.
