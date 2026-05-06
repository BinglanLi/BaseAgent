# CLI Usage and Output Files

## Pipeline Commands

```bash
# Full pipeline (all enabled sources)
python src/main.py

# Single source — runs extract + export_tsv only, then stops
python src/main.py --source <databases_yaml_key>

# Force re-download of source files
python src/main.py --force-download

# Verbose logging
python src/main.py --log-level DEBUG
```

**`--source` note**: runs only extract and export_tsv steps. Populate and graph export do NOT run.

---

## Running Steps in Isolation

### Populate only (populated RDF already exists prerequisite: TSVs in data/processed/)

```python
from src.ontology.populator import OntologyPopulator
from src.main import load_config

project, databases, mappings = load_config()
populator = OntologyPopulator(
    ontology_path="data/ontology/alzkb_v2.rdf",
    data_dir="data/processed",
    ontology_mappings=mappings,
)
# Validate before running (not automatic)
errors = populator.validate_config("source.output_name", mappings["source.output_name"])
for e in errors: print(e)
```

### Graph export only (prerequisite: data/output/alzkb_v2_populated.rdf exists)

```python
from src.export.memgraph_exporter import MemgraphExporter

exporter = MemgraphExporter(
    rdf_files=["data/output/alzkb_v2_populated.rdf"],
    output_dir="data/output",
)
result = exporter.export()
# result["cypher_script"] is a path string, not file content
```

---

## Output Files

| Path | Produced by | Content |
|------|-------------|---------|
| `data/raw/<source>/` | parser `download_data()` | downloaded source files |
| `data/processed/<source>/<name>.tsv` | `export_tsv()` | parsed DataFrames as TSV |
| `data/output/alzkb_v2_populated.rdf` | `OntologyPopulator.save_ontology()` | populated OWL ontology |
| `data/output/nodes_{NodeType}.csv` | `MemgraphExporter.export()` | one file per OWL class |
| `data/output/edges_{RelType}.csv` | `MemgraphExporter.export()` | one file per OWL object property |
| `data/output/import.cypher` | `MemgraphExporter.export()` | Cypher LOAD CSV script for Memgraph |

---

## Importing into Memgraph

Mount `data/output/` to `/import-data/` inside the container:

```bash
docker run -v /abs/path/to/data/output:/import-data memgraph/memgraph-platform
```

The `/import-data/` prefix in `LOAD CSV` paths is hardcoded in `import.cypher` and must match this mount point.

---

## Environment Variables

Set in `.env` (loaded by `python-dotenv` at startup). Variables are injected into parser constructors via the `_env` convention in `databases.yaml`.

See `docs/reference.md` for the full environment variable table. Common variables:

| Variable | Used by |
|----------|---------|
| `DISGENET_API_KEY` | DisGeNETParser |
| `MYSQL_USERNAME`, `MYSQL_PASSWORD`, `MYSQL_HOST`, `MYSQL_PORT` | AOPDBParser |

**Never commit `.env`.** Add credentials via `<VAR>_env: <VAR_NAME>` keys in `databases.yaml args`.


