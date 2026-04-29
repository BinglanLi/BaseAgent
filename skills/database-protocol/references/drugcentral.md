# DrugCentral Operational Reference

For evaluation metadata (node types, relationship types, update schedule, access method), see [drugcentral_eval.json](drugcentral_eval.json).

---

## Setup

**Download and load:**
```bash
curl -O https://unmtid-dbs.net/download/drugcentral.dump.11012023.sql.gz
createdb -O <owner> drugcentral
gunzip -c drugcentral.dump.11012023.sql.gz | psql drugcentral
```

**Python access**: the parser connects via psycopg2 to `host=localhost, port=5432, dbname=drugcentral` by default. Override by adding a `pg_config` block under `drugcentral.args` in `databases.yaml`. The parser sets `search_path=public` via psycopg2 options.

**databases.yaml entry** (no credentials needed for local connection with default pg settings):
```yaml
drugcentral:
  enabled: true
  args: {}
  notes: "Drug-disease treatment indications, off-label uses, and pharmacologic class data from PostgreSQL."
```

**Overriding connection parameters** — add a `pg_config` block under `args`. Accepted keys: `host`, `port`, `dbname`. Use the `_env` convention for any sensitive values:
```yaml
drugcentral:
  enabled: true
  args:
    pg_config:
      host: localhost
      port: 5432
      dbname: drugcentral
  notes: "..."
```

---

## Schema

### `omop_relationship` — Drug-disease links

| Column | Type | Description |
|--------|------|-------------|
| `id` | integer PK | |
| `struct_id` | integer | FK → `structures.id` |
| `concept_id` | integer | OMOP concept ID for disease |
| `relationship_name` | varchar(256) | `'indication'` or `'off-label use'` |
| `concept_name` | varchar(256) | disease name |
| `umls_cui` | char(8) | UMLS CUI |
| `cui_semantic_type` | char(4) | UMLS semantic type code |
| `snomed_full_name` | varchar(500) | SNOMED-CT term |
| `snomed_conceptid` | bigint | SNOMED-CT concept ID |

Use `WHERE relationship_name IN ('indication', 'off-label use')` to filter drug-disease edges.

### `pharma_class` — Pharmacologic classes + struct-to-class mapping

No separate junction table — class definition and drug membership are both here.

| Column | Type | Description |
|--------|------|-------------|
| `id` | integer PK | |
| `struct_id` | integer | FK → `structures.id` |
| `type` | varchar(20) | `'Mechanism of Action'`, `'Physiologic Effect'`, `'Chemical/Ingredient'`, `'Chemical Structure'` |
| `name` | varchar(1000) | class name |
| `class_code` | varchar(20) | class identifier (e.g. NDFRT code) |
| `source` | varchar(100) | e.g. `'FDA'` |

### `identifier` — External ID mappings

Maps `struct_id` to external database IDs. Use `WHERE id_type = 'DRUGBANK_ID'` to get DrugBank IDs for cross-source merging.

| Column | Type | Description |
|--------|------|-------------|
| `id` | integer PK | |
| `identifier` | varchar(50) | the external ID value |
| `id_type` | varchar(50) | e.g. `'DRUGBANK_ID'` |
| `struct_id` | integer | FK → `structures.id` |
| `parent_match` | boolean | |

---

## Inspect Queries

```bash
psql drugcentral -c "\dt"
psql drugcentral -c "SELECT relationship_name, count(*) FROM omop_relationship GROUP BY 1;"
psql drugcentral -c "SELECT type, count(*) FROM pharma_class GROUP BY 1;"
psql drugcentral -c "\d omop_relationship"
psql drugcentral -c "\d pharma_class"
psql drugcentral -c "\d identifier"
```

---

## Known Gotchas

**Do not add `user_env`/`password_env` to `databases.yaml` unless the env vars are set.** `_resolve_env_vars()` converts unset `_env` keys to `None`. psycopg2 treats `None` as an empty string and fails authentication silently. For local PostgreSQL with no password required, use `args: {}` (no credential keys at all).

**Search path**: the parser sets `search_path=public`. If tables are not found after loading the dump, verify the tables are in the `public` schema: `psql drugcentral -c "\dt public.*"`.
