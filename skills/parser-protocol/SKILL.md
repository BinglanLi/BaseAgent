---
name: parser-protocol
description: Use when creating a new parser or fixing/updating an existing parser under src/parsers/*.py. Covers the BaseParser contract, constructor patterns (credentials, disease scope), loading project config via config_loader.py, minimal template, registration steps (including eval_parser.py), and running the evaluator. Parsers download biomedical source data and return clean pandas DataFrames; this skill does not require knowledge of OWL, ista, or Memgraph.
---

You write, improve, and maintain parsers under `src/parsers/*.py`. A parser produces pandas DataFrames from one biomedical source. You do not need to understand OWL, ista, or Memgraph.

## What a Parser Does

1. Downloads source data into `data/raw/<source_name>/`.
2. Returns one or more named DataFrames from those files.
3. Declares the exact column schema of those DataFrames.

The pipeline writes each DataFrame to `data/processed/<source_name>/<output_name>.tsv`, where `<source_name>` is the `databases.yaml` key for this source. The `ontology_mappings.yaml` config references both the TSV stems and the column names — choose them carefully and keep `get_schema()` in sync.

---

## The `BaseParser` Contract

Inherit from `src/parsers/base_parser.py`. Implement three abstract methods:

### `download_data() -> bool`
Download and cache source files. Return `True` if files are ready; `False` on failure (pipeline logs a warning and continues with any existing cached files).

### `parse_data() -> dict[str, pd.DataFrame]`
Return `{output_name: df}`. Dict keys become TSV filename stems and must match the `source_filename` values in `ontology_mappings.yaml`.

### `get_schema() -> dict[str, dict[str, str]]`
Return `{output_name: {col_name: description}}` matching every column in `parse_data()` output. Keep in sync whenever columns change; `eval_parser.py` uses this for schema drift detection.

---

## BaseParser Helpers

| Method | Use for |
|--------|---------|
| `self.download_file(url, filename)` | Download a file; skips if cached (respects `self.force`) |
| `self.extract_gzip(gz_path)` | Decompress `.gz`; skips if already extracted |
| `self.read_tsv(filepath, **kwargs)` | `pd.read_csv` with `sep="\t"` |
| `self.read_csv(filepath, **kwargs)` | `pd.read_csv` with default separator |
| `self.validate_data(df, required_columns)` | Check required columns are present |
| `self.get_file_path(filename)` | Absolute path under `self.source_dir` |

`BaseParser.__init__` sets: `self.data_dir`, `self.source_name` (class name minus "Parser", lowercased), `self.source_dir` (`data_dir/<source_name>/`), `self.force = False`.

Note: `self.force` is set externally by the pipeline's `--force-download` flag — do not set it inside a parser.

---

## Constructor Patterns

```python
class MySourceParser(BaseParser):
    def __init__(self, data_dir: str, my_param: str = None):
        super().__init__(data_dir)
        self.my_param = my_param
```

**Disease scope** — for parsers that query an API by disease terms:
```python
def __init__(self, data_dir: str, disease_scope: dict = None):
    super().__init__(data_dir)
    self.disease_terms = (disease_scope or {}).get("primary_terms", [])
```
The pipeline auto-injects `disease_scope` from `config/project.yaml` when it detects this parameter via `inspect.signature()`.

---

## Loading Project Configuration Directly

Use `src/config_loader.py` when a parser needs project config values **outside `__init__`** (e.g., in `parse_data()`) or needs fields other than `disease_scope` (e.g., `node_types`, `edge_types`, `ontology`).

```python
from src.config_loader import load_project_config, get_disease_scope
```

| Function | Returns |
|----------|---------|
| `load_project_config()` | Full `project` dict from `config/project.yaml` (cached) |
| `get_disease_scope()` | The `disease_scope` sub-dict, or `{}` if absent |

**Example — filtering rows by disease terms at parse time:**
```python
from src.config_loader import get_disease_scope

def parse_data(self) -> dict[str, pd.DataFrame]:
    disease_terms = get_disease_scope().get("primary_terms", [])
    df = self.read_tsv(self.get_file_path(self.SOURCE_FILE))
    if disease_terms:
        df = df[df["disease"].isin(disease_terms)]
    return {OUTPUT_NAME: df}
```

Prefer the auto-injection pattern in `__init__` for `disease_scope`; use `load_project_config()` only when you need config in a method or need a field that isn't auto-injected.

**Credentials** — never hard-code; declare as a parameter:
```python
def __init__(self, data_dir: str, api_key: str = None):
```
In `config/databases.yaml`, the `_env` suffix resolves the value from the environment at startup. The suffix is stripped when injecting: `api_key_env: MY_API_KEY` → constructor receives `api_key=<value>`. Name the constructor parameter to match the stripped key.
```yaml
mysource:
  args:
    api_key_env: MYSOURCE_API_KEY
```

---

## Minimal Parser Template

```python
from .base_parser import BaseParser
import pandas as pd
import logging

logger = logging.getLogger(__name__)

OUTPUT_NAME = "my_entities"   # TSV stem; must match source_filename in ontology_mappings.yaml

class MySourceParser(BaseParser):

    SOURCE_URL = "https://example.org/data.tsv.gz"
    SOURCE_FILE = "data.tsv"

    def download_data(self) -> bool:
        gz = self.download_file(self.SOURCE_URL, "data.tsv.gz")
        if not gz:
            return False
        return self.extract_gzip(gz) is not None

    def parse_data(self) -> dict[str, pd.DataFrame]:
        df = self.read_tsv(self.get_file_path(self.SOURCE_FILE))
        if df is None:
            return {}
        df = df.rename(columns={"raw_col": "standardized_col"})
        df["source_database"] = "MySource"
        return {OUTPUT_NAME: df}

    def get_schema(self) -> dict[str, dict[str, str]]:
        return {
            OUTPUT_NAME: {
                "standardized_col": "Description of this column",
                "source_database": "Source name string",
            }
        }
```

---

## Registration Checklist

1. **`src/parsers/__init__.py`** — add the import and add the class name to `__all__`:
   ```python
   from .mysource_parser import MySourceParser
   # and in __all__:
   'MySourceParser',
   ```

2. **`src/main.py`** `PARSERS` dict — add `"mysource": MySourceParser`

3. **`config/databases.yaml`** — add the source entry under `databases:`:
   ```yaml
   mysource:
     enabled: true
     args:
       api_key_env: MYSOURCE_API_KEY   # remove if no credentials needed
     notes: "One-line description of what this source provides."
   ```
   The key (`mysource`) controls the `data/processed/` subdirectory name and must match the prefix used in `ontology_mappings.yaml`.

