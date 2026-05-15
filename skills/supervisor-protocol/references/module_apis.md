# Module API Contracts

## Contents
- [BaseParser](#baseparser)
- [OntologyPopulator](#ontologypopulator)
- [MemgraphExporter](#memgraphexporter)

---

## BaseParser

`src/parsers/base_parser.py`

### Abstract methods (must implement)

```python
def download_data(self) -> bool:
    # Download source data to data/raw/<source_name>/
    # Returns True on success, False on failure
    # Must be idempotent — skip download if files exist unless force_download=True

def parse_data(self) -> dict[str, pd.DataFrame]:
    # Returns dict mapping output_name → DataFrame
    # Keys become TSV filename stems: "gene_disease" → gene_disease.tsv
    # Must match get_schema() column names exactly

def get_schema(self) -> dict[str, dict[str, str]]:
    # Returns dict mapping output_name → {column_name: description}
    # Must exactly match what parse_data() produces — drift causes eval failures
```

### Constructor

```python
BaseParser.__init__(self, data_dir: Optional[str] = None)
```

- `data_dir` is passed from `main.py` via `args["data_dir"] = str(raw_dir)` before parser instantiation
- `source_name` is derived automatically: `self.__class__.__name__.replace('Parser', '').lower()` — NOT from databases.yaml
- `source_dir = data_dir / source_name` is fixed at construction time and controls `data/raw/<classname>/`
- Subclasses may declare additional kwargs (e.g. `pg_config`, `disease_scope`) that receive values from databases.yaml `args` after `_env` resolution

### Attributes and helper methods

```python
self.data_dir: Path                          # root data directory (raw_dir passed from main.py)
self.source_name: str                        # derived from class name (may differ from databases.yaml key)
self.source_dir: Path                        # data_dir / source_name — where raw files are stored
self.force: bool                             # set by caller after construction (parser.force = force_download)
self.download_file(url, filename) -> Path    # downloads to source_dir/
self.extract_gzip(gz_path) -> Path           # extracts .gz in place
self.read_tsv(path, **kwargs) -> DataFrame   # wraps pd.read_csv with tab separator
```

---

## OntologyPopulator

`src/ontology/populator.py`

### Constructor

```python
OntologyPopulator(
    ontology_path: str,                          # data/ontology/ontology.rdf
    data_dir: str,                               # data/processed/
    mysql_config: Optional[Dict[str, str]] = None,  # required when using parser_type="mysql" (AOP-DB)
    ontology_mappings: Optional[Dict] = None,    # if None, loads from config/ontology_mappings.yaml
)
```

### Key methods

```python
def populate_from_config(
    self,
    config_key: str,             # e.g. "disgenet.gene_disease_associations"
    fmt: str = "tsv",
    parser_type: str = "flat",
) -> tuple[bool | None, str | None]:
    # Returns (None, None) if config_key not found in ontology_mappings
    # Returns (False, None) if source_filename missing from config
    # Returns (True, 'node') or (True, 'relationship') for skip: true entries
    # Returns (True/False, 'node'/'relationship') based on populate outcome

def validate_config(
    self,
    config_key: str,
    config: dict,
) -> list[str]:
    # Checks: node_type, relationship_type, *_node_type, *_match_property, data_property_map
    # Does NOT check: merge_column.data_property
    # Never called automatically — must be invoked manually before pipeline run
    # Returns list of error strings (empty = valid)

def save_ontology(self, output_path: str) -> str:
    # Serializes populated ontology to RDF
    # Returns path written

def print_stats(self):
    # Logs aggregate ontology statistics: total class count, individual count,
    # object property count, data property count — not per-class breakdowns
```

### Population resolution

- Node entry: creates individual with IRI `<namespace>#<iri_column_value>`; attaches data properties from `data_property_map`
- Relationship entry: looks up subject by `subject_match_property = subject_column_value`, looks up object by `object_match_property = object_column_value`; creates object property assertion
- `merge: true`: finds existing individual by matching `merge_column.data_property = merge_column.source_column_name value` before creating new

---

## MemgraphExporter

`src/export/memgraph_exporter.py`

### Constructor

```python
MemgraphExporter(
    rdf_files: list[str],        # list of populated RDF file paths
    output_dir: str,             # destination for CSVs and import.cypher
)
```

### export()

```python
def export(self) -> dict:
    # Returns:
    # {
    #   "nodes_count": int,
    #   "edges_count": int,
    #   "output_files": list[str],
    #   "cypher_script": str,    # PATH to import.cypher — NOT the file content
    # }
    # Silently overwrites all existing CSV files in output_dir — no append, no warning
```

### Output file formats

**Node CSV** (`nodes_{NodeType}.csv`): columns = `id`, data properties in alphabetical order, `uri` (last)
- `uri` is present in CSV but excluded from Cypher CREATE statement
- An individual belonging to multiple OWL classes appears in every corresponding node CSV

**Edge CSV** (`edges_{RelType}.csv`): columns = `start_id`, `end_id`, `start_uri`, `end_uri`

**`import.cypher`**: LOAD CSV script for Memgraph
- Uses label-agnostic `MATCH (a {id: row.start_id})` for edge endpoints — relies on globally unique node IDs across all types
- Node IRI prefixes (`gene_*`, `drug_*`, `disease_*`) are assigned by ista at populate time, ensuring uniqueness in practice
- Each node type gets a label index and an `id` index; `graph_indexes` in `project.yaml` is informational only — not read by the exporter

### Key internal methods

| Method | Purpose |
|--------|---------|
| `_extract_nodes(ontology_ns)` | Groups `owl:NamedIndividual` triples by class; collects data properties |
| `_extract_edges(ontology_ns)` | Groups object property assertions by property local name |
| `_write_node_csv(filepath, nodes, node_type)` | Writes node CSV; returns property column list |
| `_write_edge_csv(filepath, edges, rel_type)` | Writes edge CSV |
| `_write_cypher_script(node_columns, rel_types)` | Generates import.cypher |
