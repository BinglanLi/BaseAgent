# AOP-DB Operational Reference

---

## Setup

**Download and load:**
```bash
# Download from EPA FTP (~7.2 GB compressed)
# https://gaftp.epa.gov/EPADataCommons/ORD/AOP-DB/  →  AOP-DB_v2.zip
# Unzip and import the SQL dump into MySQL
mysql -u root aopdb < AOP-DB_v2.sql
```

**Python dependency:** `mysql-connector-python` must be installed.

**databases.yaml entry:**
```yaml
aopdb:
  enabled: true
  args:
    mysql_config:
      host: "localhost"
      user_env: MYSQL_USERNAME
      password_env: MYSQL_PASSWORD
      database_env: MYSQL_DB_NAME
```

`database_env: MYSQL_DB_NAME` means the database name is read from the env var — set `MYSQL_DB_NAME=aopdb` (or whatever the imported database is named).

---

## Available Tables

### `pathway_gene` — gene–pathway memberships

| Column | Description |
|---|---|
| `path_id` | Pathway identifier |
| `path_name` | Pathway name — may contain HTML tags (`<sub>`, `<i>`) and the suffix ` - Homo sapiens (human)` |
| `ext_source` | External source (e.g. KEGG, Reactome) |
| `tax_id` | Taxonomy ID — human records have `9606` |
| `entrez` | Entrez Gene ID |

Multiple `path_id` values can share the same `path_name`. Pathway identity should be treated as name-based, not ID-based. Records must be filtered to `tax_id = 9606` to get human-only data.

### `chemical_info` — chemical/drug records

Columns are dump-version dependent (`SELECT *`). Confirmed present: `DTX_id` (DSSTox identifier), `ChemicalID` (MeSH chemical ID). Verify column names after importing a new dump.

---

## Known Gotchas

**HTML in `path_name`** — pathway names contain HTML tags and species suffixes that must be stripped before use as identifiers or display strings.

**`path_id` is not a stable unique identifier** — multiple pathway IDs can share one name; use `path_name` (after cleaning) as the canonical pathway identifier.

**`database_env` is unusual** — most parsers use a fixed database name, but AOP-DB reads it from `MYSQL_DB_NAME`. If that env var is unset, the value resolves to `None` and the connection fails silently (see `_env` convention in SKILL.md).
