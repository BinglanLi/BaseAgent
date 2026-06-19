# HPO Operational Reference

---

## Setup

No API key or account required. All files are direct bulk downloads.

---

## Source Files

### `hp.obo`

Full Human Phenotype Ontology in OBO format. One node per phenotype term; hierarchy is encoded in `is_a:` fields.

**Download URL:**
```
http://purl.obolibrary.org/obo/hp.obo
```

**Relevant OBO fields parsed per term:**

| Field | Description |
|---|---|
| `id` | HP CURIE (e.g. `HP:0001250`) |
| `name` | Phenotype name |
| `def` | Text definition; OBO quotes and citation brackets are stripped |
| `synonym` | Synonym strings; extracted text is pipe-delimited in output |
| `is_a` | Parent HP IDs; pipe-delimited in output |
| `is_obsolete` | Obsolete flag; obsolete terms are excluded |

---

### `phenotype.hpoa`

HPOA annotation file. Maps diseases (OMIM, Orphanet) to HPO terms with evidence, frequency, onset, and sex modifier annotations. Tab-delimited; lines beginning with `#` are comment/header lines.

**Download URL:**
```
http://purl.obolibrary.org/obo/hp/hpoa/phenotype.hpoa
```

**Columns used:**

| Column | Description |
|---|---|
| `database_id` | Disease identifier (e.g. `OMIM:143100`, `ORPHA:14`) |
| `qualifier` | `NOT` for absent phenotype; empty for present |
| `hpo_id` | HPO term CURIE (e.g. `HP:0001250`) |
| `evidence` | Evidence code: `IEA` (electronic), `TAS` (traceable author), `PCS` (published clinical study) |
| `onset` | Age-of-onset HPO term (e.g. `HP:0003577`) |
| `frequency` | Frequency annotation — HPO term (e.g. `HP:0040281`) or ratio (e.g. `3/10`) |
| `sex` | Sex modifier: `MALE`, `FEMALE`, or empty |

---

### `MedGenIDMappings.txt.gz`

NCBI MedGen cross-reference file. Maps OMIM and Orphanet disease IDs to UMLS CUIs. Pipe-delimited; the first line is a `#`-prefixed header.

**Download URL:**
```
https://ftp.ncbi.nlm.nih.gov/pub/medgen/MedGenIDMappings.txt.gz
```

**Columns used:**

| Column | Description |
|---|---|
| `CUI_or_CN_id` (col 0) | UMLS CUI (e.g. `C0002395`) |
| `source_id` (col 2) | Source identifier — numeric for OMIM (e.g. `143100`), prefixed for Orphanet (e.g. `Orphanet_14`) |
| `source` (col 3) | Source name: `OMIM` or `Orphanet` |

Updated weekly. No credentials required.

---

## Processing

### Phenotype nodes (`hp.obo`)

1. **Prefix filter** — only terms with `id` starting with `HP:` are retained.
2. **Obsolete filter** — terms with `is_obsolete: true` are excluded.
3. **Output** — one file, `phenotype_nodes.tsv`:

| Column | Description |
|---|---|
| `hp_id` | HP CURIE (e.g. `HP:0001250`) — IRI column |
| `name` | Phenotype name |
| `definition` | Text definition (OBO quotes and citation brackets stripped) |
| `synonyms` | Pipe-delimited synonym strings |
| `is_a` | Pipe-delimited parent HP CURIEs via `is_a` relationships |
| `source_database` | Literal `"HPO"` |

---

### Disease-phenotype edges (`phenotype.hpoa` + `MedGenIDMappings.txt.gz`)

1. **CUI mapping** — `database_id` (OMIM/Orphanet CURIE) is mapped to a UMLS CUI using MedGenIDMappings. OMIM: `source="OMIM"`, `source_id="143100"` → key `"OMIM:143100"`. Orphanet: `source="Orphanet"`, `source_id="Orphanet_14"` → key `"ORPHA:14"`. Rows with no CUI mapping retain an empty `cui` column.

2. **ID filter** — rows missing `database_id` or `hpo_id`, or whose `hpo_id` does not start with `HP:`, are dropped.

3. **Qualifier split** — rows with `qualifier == "NOT"` go to the negative edges file; all others go to the positive edges file. This follows PrimeKG filter logic; no aspect filter is applied.

4. **Output** — two files:

`disease_phenotype_positive.tsv` — phenotype-present associations:

| Column | Description |
|---|---|
| `hp_id` | HPO term CURIE |
| `cui` | UMLS CUI of the disease (mapped from OMIM/ORPHA via MedGen) |
| `evidence` | Evidence code (`IEA`, `TAS`, `PCS`) |
| `onset` | Age-of-onset HPO term, or empty |
| `frequency` | Frequency annotation, or empty |
| `sex` | Sex modifier (`MALE`, `FEMALE`), or empty |
| `source_database` | Literal `"HPO"` |

`disease_phenotype_negative.tsv` — phenotype-absent associations (all `TAS` evidence by HPOA curation convention):

| Column | Description |
|---|---|
| `hp_id` | HPO term CURIE |
| `cui` | UMLS CUI of the disease |
| `evidence` | Evidence code (always `TAS`) |
| `onset` | Age-of-onset HPO term, or empty |
| `frequency` | Frequency annotation, or empty |
| `sex` | Sex modifier, or empty |
| `source_database` | Literal `"HPO"` |

---

## Cross-Reference Strategy

HPO produces one node type and two edge types. Edges match Disease nodes via `xrefUmlsCUI`.

| Property | Source column | Populated by |
|---|---|---|
| `xrefHPO` | `hp_id` | HPO |
| `parentHPO` | `is_a` | HPO |
| `xrefUmlsCUI` (on Disease) | `cui` | HPO (via MedGen), DisGeNET, MONDO, DrugCentral |

---

## Known Gotchas

**MedGen download failure is non-fatal** — if `MedGenIDMappings.txt.gz` fails to download, the parser logs a warning and continues. All edge rows will have an empty `cui` column, causing zero disease-phenotype edges to populate (the populator matches on `xrefUmlsCUI` and will find no Disease nodes).

**No aspect filter** — the parser follows PrimeKG convention and does not filter by HPOA `aspect` column. This includes inheritance mode (`I`), clinical modifier (`M`), and other non-phenotypic annotations alongside phenotypic ones (`P`). Filter downstream if only `aspect == P` annotations are needed.

**Negative edges are always `TAS`** — absent-phenotype annotations require explicit expert curation from a published source; they cannot be computationally inferred. All `qualifier == NOT` rows carry `TAS` evidence by definition.

**Frequency and onset are sparse** — many HPOA rows leave `frequency` and `onset` blank. These columns are retained but will be empty for the majority of edges.

**`hp.obo` and `phenotype.hpoa` are released together** — both are updated with each HPO release (approximately monthly). Using mismatched versions may result in edges referencing HP IDs not present in the node file.
