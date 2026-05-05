# DrugCentral Operational Reference

For evaluation metadata (node types, relationship types, update schedule, access method), see [drugcentral_eval.json](drugcentral_eval.json).

---

## Setup

**Download and load (local):**
```bash
curl -O https://unmtid-dbs.net/download/drugcentral.dump.11012023.sql.gz
createdb -O <owner> drugcentral
gunzip -c drugcentral.dump.11012023.sql.gz | psql drugcentral
```

**Public read-only instance** (no setup; may be slow under load):
```
postgresql://drugman:dosage@unmtid-dbs.net:5433/drugcentral
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

### `structures` — Primary drug entity table

~4,995 entries. Fields: `id` (DrugCentral ID), `name` (INN), `cas_reg_no`, `smiles`, `inchi`, `inchikey`, `cd_formula`, `cd_molweight`, `clogp`, `tpsa`, `lipinski`, `status` (approved vs. other), physicochemical descriptors. `id` is the FK used by all other tables as `struct_id`.

### `omop_relationship` — Drug-disease links

~42,307 rows. Fields: `struct_id`, `concept_id` (OMOP), `relationship_name`, `concept_name`, `umls_cui`, `cui_semantic_type`, `snomed_full_name`, `snomed_conceptid`.

| `relationship_name` | Count |
|---|---|
| `contraindication` | 27,731 |
| `indication` | 12,047 |
| `off-label use` | 2,525 |
| `symptomatic treatment` | 2 |
| `reduce risk` | 1 |
| `diagnosis` | 1 |

**Disease vocabulary is SNOMED CT / UMLS CUI — not Disease Ontology.** Use `doid_xref` to map UMLS CUIs to DO/MeSH/ICD for KG use.

### `act_table_full` — Drug-target binding activity

~20,978 measurements; 2,754 drugs; 3,205 targets. Fields: `struct_id`, `target_id`, `target_name`, `target_class`, `accession` (UniProt), `gene`, `swissprot`, `act_value`, `act_unit`, `act_type` (Ki/IC50/Kd/EC50/AC50/Km/etc.), `act_comment`, `act_source`, `relation` (=/</>/etc.), `moa` (1 = mechanism of action), `action_type`, `first_in_class`, `tdl`, `organism`.

`moa=1` marks curated MoA interactions (2,866 records). Top action types for MoA: INHIBITOR, AGONIST, ANTAGONIST, BLOCKER. Sources: CHEMBL (12,581), WOMBAT-PK (2,845), DRUG MATRIX (2,255), IUPHAR (1,294).

### `pharma_class` — Pharmacologic classes

~25,687 rows. No separate junction table — class definition and drug membership are both here. Fields: `struct_id`, `type`, `name`, `class_code`, `source`.

| `source` / `type` | Count | Notes |
|---|---|---|
| FDA EPC (Established Pharmacological Class) | 1,654 | Most specific for approved-drug class |
| FDA MoA | 1,500 | |
| MeSH PA (Pharmacological Action) | 14,274 | Broadest coverage |
| CHEBI role | 7,385 | |

### `identifier` — External ID mappings

Maps `struct_id` to 20 external database IDs. Fields: `struct_id`, `identifier`, `id_type`, `parent_match`.

Available `id_type` values: `CHEBI`, `ChEMBL_ID`, `DRUGBANK_ID`, `INN_ID`, `IUPHAR_LIGAND_ID`, `KEGG_DRUG`, `MESH_DESCRIPTOR_UI`, `MESH_SUPPLEMENTAL_RECORD_UI`, `MMSL`, `NDDF`, `NUI`, `PDB_CHEM_ID`, `PUBCHEM_CID`, `RXNORM`, `SECONDARY_CAS_RN`, `SNOMEDCT_US`, `UMLSCUI`, `UNII`, `VANDF`, `VUID`.

Use `WHERE id_type = 'DRUGBANK_ID'` to get DrugBank IDs for cross-source merging.

### `faers` — FDA adverse event signals

~364,935 rows; 2,013 drugs; 13,725 MedDRA Preferred Terms. Fields: `struct_id`, `meddra_name`, `meddra_code`, `level` (always `PT`), `llr` (log-likelihood ratio), `llr_threshold`, `drug_ae`, `drug_no_ae`, `no_drug_ae`, `no_drug_no_ae`. Only statistically significant signals (`llr > llr_threshold`) are included — not raw FAERS reports.

### `doid` / `doid_xref` — Disease Ontology cross-references

10,040 DO terms. `doid_xref` maps each to external vocabularies: UMLS_CUI (6,851), OMIM (4,958), SNOMEDCT (4,839), NCI (4,683), ICD10CM (3,658), MESH (3,508), ICD9CM (2,276), GARD (1,866), ORDO (1,760), EFO (132), MEDDRA (33), others. Use this table to bridge `omop_relationship.umls_cui` → Disease Ontology.

### `atc` / `struct2atc` — ATC hierarchy

5,372 ATC codes (full 5-level hierarchy with DDD). `struct2atc` maps 3,282 drugs to ATC codes.

### `target_dictionary` / `target_component` / `td2tc` — Target metadata

`target_dictionary`: `id`, `name`, `target_class`, `protein_type`, `tdl` (IDG Target Development Level). `target_component`: `accession` (UniProt), `swissprot`, `organism`, `gene`, `geneid` (Entrez). `td2tc` links multi-component targets.

---

## Inspect Queries

```bash
psql drugcentral -c "\dt"
psql drugcentral -c "SELECT relationship_name, count(*) FROM omop_relationship GROUP BY 1 ORDER BY 2 DESC;"
psql drugcentral -c "SELECT type, source, count(*) FROM pharma_class GROUP BY 1,2 ORDER BY 3 DESC;"
psql drugcentral -c "SELECT id_type, count(*) FROM identifier GROUP BY 1 ORDER BY 2 DESC;"
psql drugcentral -c "SELECT act_type, count(*) FROM act_table_full WHERE moa=1 GROUP BY 1 ORDER BY 2 DESC;"
psql drugcentral -c "\d omop_relationship"
psql drugcentral -c "\d act_table_full"
psql drugcentral -c "\d identifier"
```

---

## Known Gotchas

**`omop_relationship` disease vocabulary is SNOMED CT / UMLS CUI, not Disease Ontology** — direct join to KG Disease nodes requires mapping via `doid_xref` (`umls_cui` → DO ID). The `contraindication` type (27,731 rows) outnumbers `indication` (12,047) — filter on `relationship_name` deliberately.

**Do not add `user_env`/`password_env` to `databases.yaml` unless the env vars are set.** `_resolve_env_vars()` converts unset `_env` keys to `None`. psycopg2 treats `None` as an empty string and fails authentication silently. For local PostgreSQL with no password required, use `args: {}` (no credential keys at all).

**Search path**: the parser sets `search_path=public`. If tables are not found after loading the dump, verify the tables are in the `public` schema: `psql drugcentral -c "\dt public.*"`.

**`act_table_full` is a curated subset, not a full binding database** — covers approved/clinical drugs only (~21K measurements). For broad chemical-biology binding data, BindingDB (~2.9M measurements) is necessary. For MoA-confirmed interactions, use `moa=1` filter (2,866 records).

**`faers` contains disproportionality signals, not raw reports** — all rows already pass `llr > llr_threshold`. There are no label-extracted side effects in DrugCentral; for those, SIDER v4.1 has coverage but is frozen at 2015.
