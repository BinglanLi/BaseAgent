# project.yaml Format Reference

## Disease Scope

The `disease_scope` block is injected by the pipeline into any parser whose constructor declares a `disease_scope` parameter. Each parser uses whichever fields it needs.

```yaml
project:
  disease_scope:
    primary_terms:        # Free-text search strings (DisGeNET API)
      - "alzheimer"
    umls_cuis:            # UMLS concept IDs (DisGeNET filtering)
      - "C0002395"
    doid_ids:             # Disease Ontology IDs (DiseaseOntologyParser filtering)
      - "DOID:10652"
    mesh_ids:             # MeSH descriptor IDs (MeSHParser, MEDLINECooccurrenceParser)
      - "D000544"
    drug_names:           # Known drugs for this disease (DrugBankParser post-filter)
      - "donepezil"
```

| Field | Format | Consuming parsers |
|-------|--------|------------------|
| `primary_terms` | lowercase strings | DisGeNET (free-text API search) |
| `umls_cuis` | `C` + 7 digits | DisGeNET (identifier filter) |
| `doid_ids` | `DOID:` + digits | DiseaseOntologyParser |
| `mesh_ids` | `D` + digits | MeSHParser, MEDLINECooccurrenceParser |
| `drug_names` | lowercase strings | DrugBankParser (commented out by default) |

Parsers that do not declare `disease_scope` in their constructor receive nothing — the pipeline detects the parameter via `inspect.signature()`.

---

## Ontology Paths

```yaml
  ontology:
    base_file: "data/ontology/alzkb_v2.rdf"       # Unmodified base OWL schema
    namespace: "http://jdr.bio/ontologies/alzkb.owl"
    populated_output: "data/output/alzkb_v2_populated.rdf"
```

`base_file` is loaded by `OntologyPopulator` at populate time. `populated_output` is where the pipeline saves the ontology after population. Do not change `namespace` unless the RDF IRI changes.

---

## node_types and edge_types

```yaml
  node_types:
    - Gene          # Active: a parser currently produces Gene nodes
    # - CellType    # Inactive: defined in ontology, no parser produces it yet
  edge_types:
    - geneAssociatesWithDisease
    # - diseaseUpregulatesGene
```

- **Active** entries are uncommented. `eval_parser.py` validates mapping entries against this list.
- **Inactive** entries are commented out. Preserve them — they document classes/properties that exist in the RDF but have no data source yet.
- Only object properties (edge types) appear in `edge_types`. Data properties used in `data_property_map` are resolved directly from the ontology and do not need to be listed here.

---

## graph_indexes

```yaml
  graph_indexes:
    Gene: ["geneSymbol", "xrefNcbiGene"]
```

Informational only — used by `MemgraphExporter` to generate `CREATE INDEX` statements in `import.cypher`. Add an entry when a new node type needs indexed properties for fast Cypher lookups.
