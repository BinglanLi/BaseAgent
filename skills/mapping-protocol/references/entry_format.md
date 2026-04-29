# Entry Format Reference

## Contents
- [Node Entry](#node-entry)
- [Relationship Entry](#relationship-entry)
- [merge Semantics](#merge-semantics)
- [filter_column / filter_value](#filter_column--filter_value)
- [compound_fields](#compound_fields)
- [inverse_relationship_type](#inverse_relationship_type)

---

## Node Entry

```yaml
<source>.<output_name>:
  data_type: node
  node_type: <OWL class name>          # must be in project.yaml node_types
  source_filename: <output_name>.tsv
  merge: false                          # see merge semantics below
  skip: false
  parse_config:
    headers: true
    iri_column_name: <col>             # column whose value becomes the individual IRI
    filter_column: <col>               # optional: restrict rows before loading
    filter_value: <val>
    data_property_map:
      <tsv_col>: <OWL data property>   # must exist in the ontology (not listed in project.yaml)
    merge_column:                       # required when merge: true
      source_column_name: <col>
      data_property: <OWL data property>
      # optional: additional tsv_col → data_property pairs may follow
      <tsv_col>: <OWL data property>
```

**Real example** — disease node with filter and merge (DisGeNET):

```yaml
disgenet.disease_mappings:
  data_type: node
  node_type: Disease
  source_filename: disease_mappings.tsv
  merge: true
  skip: false
  parse_config:
    headers: true
    iri_column_name: diseaseId
    filter_column: DO
    filter_value: "0"
    merge_column:
      source_column_name: diseaseId
      data_property: xrefUmlsCUI
    data_property_map:
      DO: xrefDiseaseOntology
```

---

## Relationship Entry

```yaml
<source>.<output_name>:
  data_type: relationship
  relationship_type: <OWL object property>           # must be in project.yaml edge_types
  inverse_relationship_type: <OWL object property>   # optional; must also be in edge_types
  source_filename: <output_name>.tsv
  merge: false
  skip: false
  parse_config:
    headers: true
    subject_node_type: <OWL class>
    subject_column_name: <col>
    subject_match_property: <OWL data property>      # look up subject by this property value
    object_node_type: <OWL class>
    object_column_name: <col>
    object_match_property: <OWL data property>       # look up object by this property value
    filter_column: <col>                             # optional
    filter_value: <val>
```

**Real example** — gene→pathway with inverse (AOP-DB):

```yaml
aopdb.gene_pathway_relationships:
  data_type: relationship
  relationship_type: geneInPathway
  inverse_relationship_type: pathwayContainsGene
  source_filename: gene_pathway_relationships.tsv
  merge: false
  skip: false
  parse_config:
    headers: true
    subject_node_type: Gene
    subject_column_name: entrez
    subject_match_property: xrefNcbiGene
    object_node_type: Pathway
    object_column_name: path_name
    object_match_property: pathwayName
```

---

## merge Semantics

When `merge: true`, the populate step finds an existing individual by matching the value in `merge_column.source_column_name` against the OWL data property `merge_column.data_property`. If a match is found, new data properties are added to the existing individual. If no match is found, a new individual is created.

Use `merge: true` when multiple data sources contribute properties to the same node type (e.g., Drug nodes populated by both DrugBank and AOP-DB). The `output_node_count` in `eval_parser.py` includes all nodes from all sources, so merge match rates may appear biased.

---

## filter_column / filter_value

Restricts rows processed from the TSV before IRI generation or relationship matching. Comparison is string-based.

```yaml
filter_column: diseaseType
filter_value: disease   # only rows where diseaseType == "disease" are loaded
```

Use when a single TSV mixes entity subtypes that should be loaded into separate mapping entries, or to exclude invalid/placeholder rows.

Both sides of the comparison are cast to `str` before matching. Always quote `filter_value` in YAML (e.g., `filter_value: "0"` not `filter_value: 0`) to avoid YAML type coercion changing the value before comparison.

---

## compound_fields

Splits a multi-value column into multiple data property assertions. Used when a TSV column contains pipe-delimited values with prefix-separated key-value pairs (e.g., NCBI Gene's `dbXrefs` column).

```yaml
compound_fields:
  dbXrefs:
    delimiter: "|"
    field_split_prefix: ":"
```

A value like `MIM:123456|HGNC:7890` is split on `|`, then each part is split on `:` to extract the prefix as a lookup key. The prefix maps to a data property via `data_property_map` entries whose keys match that prefix.

The parent column (`dbXrefs`) must **not** appear in `data_property_map` — the split sub-values are mapped through their prefixes only. See `ncbigene.genes` for a working example.

---

## inverse_relationship_type

When present, the populate step creates both the forward edge (`relationship_type`) and the inverse edge (`inverse_relationship_type`) from the same TSV row. Both types must appear in `project.yaml` `edge_types`.

Inverse edges are written to a separate `edges_{inverse_rel}.csv` file and counted independently by `eval_parser.py`. A resolution rate above 1.0 for the forward edge does not indicate an error — it reflects inverse expansion.
