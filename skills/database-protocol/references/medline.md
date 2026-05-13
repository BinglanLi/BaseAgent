# MEDLINE Operational Reference

---

## Setup

No account required for basic access. An NCBI API key increases the rate limit from 3 to 10 requests/second and is strongly recommended given the volume of EDirect calls.

**databases.yaml entry:**
```yaml
medline:
  enabled: true
  args:
    api_key_env: NCBI_EUTILS_API_KEY
```

Set `NCBI_EUTILS_API_KEY` in `.env`. Without it, the key resolves to `None` and EDirect uses the 3 req/s unauthenticated limit.

**Requires:**
- **EDirect CLI** (`esearch` + `efetch`) installed locally under `./edirect/` in the project root (preferred) or on the system `PATH`. Install with:
  ```bash
  HOME=$(pwd) sh -c "$(curl -fsSL https://ftp.ncbi.nlm.nih.gov/entrez/entrezdirect/install-edirect.sh)"
  ```
- **Python packages:** `lxml`, `scipy`
- **Processed outputs from prior parsers** (see Prerequisites below)

---

## Data Source

MEDLINE citations are queried via the **NCBI E-utilities** (PubMed) using EDirect CLI. There is no direct file download. All network I/O happens during `parse_data()`; `download_data()` is a no-op that returns `True` immediately.

**Query pattern per MeSH entity:**
```bash
esearch -db pubmed -query '"<mesh_name>"[MeSH Terms]' | efetch -format uid
```

EDirect handles PubMed's internal pagination, bypassing the 9,999-record REST API limit. Each call returns a list of PMIDs (one per line).

**PMID cache:** Results are written to `data/raw/medline/pmids/{mesh_id}.txt.gz` (one PMID per line, gzip-compressed). The cache is always force-refreshed on each run (`force=True`); there is no incremental reuse of prior PMID files.

---

## Prerequisites

The parser consumes processed outputs from three other parsers. All must have run successfully before invoking the MEDLINE parser:

| Required file | Source parser | Used for |
|---|---|---|
| `data/processed/disease_ontology/slim_terms.tsv` | `disease_ontology` | Disease DOID list |
| `data/raw/diseaseontology/doid.obo` | `disease_ontology` | Disease → MeSH xref lookup |
| `data/raw/mesh/desc{year}.xml` | `mesh` | MeSH ID → preferred name lookup |
| `data/processed/mesh/symptom_nodes.tsv` | `mesh` | Symptom MeSH term list |
| `data/processed/uberon/uberon_nodes.tsv` | `uberon` | Anatomy UBERON → MeSH mapping |

---

## Two-Phase Algorithm

### Phase 1 — PMID Fetch (~1,200 EDirect calls)

One EDirect call per unique MeSH entity across all three entity classes (diseases, symptoms, anatomy). Diseases are mapped from DOID → MeSH xref via `doid.obo`; anatomy terms require a `mesh_id` column from `uberon_nodes.tsv`. Entities without a valid MeSH ID are excluded entirely.

Two MeSH IDs are unconditionally excluded (flagged as erroneous DO cross-references): `D003327`, `D017202`.

### Phase 2 — In-Memory Co-occurrence Statistics

For each pair (S, T) with corpus C = union of all PMID sets:

| Variable | Formula |
|---|---|
| `cooccurrence` | `a = \|S ∩ T\|` |
| `enrichment` | `a / (\|S\| × \|T\| / \|C\|)` |
| `p_fisher` | One-tailed Fisher's exact test p-value |
| `odds_ratio` | Fisher's exact odds ratio |

Only pairs with `cooccurrence > 0` appear in the output. Disease–disease pairs use the upper triangle only (the relationship is symmetric; `doid_code_0 < doid_code_1`).

---

## Output Tables

### `disease_symptom_cooccurrence`

| Column | Description |
|---|---|
| `doid_code` | Disease Ontology ID |
| `mesh_id` | MeSH symptom descriptor ID (e.g. `D000544`) |
| `cooccurrence` | PMID intersection count |
| `enrichment` | Observed / expected ratio |
| `p_fisher` | One-tailed Fisher's exact p-value |
| `odds_ratio` | Fisher's exact odds ratio |
| `source_mesh` | MeSH ID used to query PMIDs for the disease |

### `disease_anatomy_cooccurrence`

Same statistics columns; entity columns are `doid_code` + `uberon_id`; includes both `source_mesh` (disease MeSH) and `target_mesh` (anatomy MeSH used for query).

### `disease_disease_cooccurrence`

Same statistics columns; entity columns are `doid_code_0` + `doid_code_1` with corresponding `mesh_id_0` + `mesh_id_1`. Upper triangle only.

---

## Known Gotchas

**EDirect is not a Python package** — it is a suite of Perl scripts installed separately. `pip install` alone is insufficient. The parser looks for `esearch`/`efetch` in `./edirect/` first, then falls back to the system `PATH`; if neither is found, `parse_data()` returns empty immediately.

**No incremental PMID caching** — `self.force = True` is hardcoded; every run re-fetches all PMID sets via EDirect. Caching from a previous run is never reused. Full runs take hours depending on entity count and API key rate limit.

**Anatomy terms without a MeSH xref are silently excluded** — `uberon_nodes.tsv` rows with an empty `mesh_id` column are dropped before Phase 1. UBERON terms that lack a MeSH cross-reference produce no anatomy co-occurrence edges.

**Disease → MeSH mapping takes first valid xref** — if a DOID has multiple `xref: MESH:` entries in `doid.obo`, only the first one that is present in `desc{year}.xml` is used for querying. Diseases with no valid MeSH xref are excluded.

**EDirect per-entity timeout is 600 seconds** — broad MeSH terms (e.g. a top-level disease concept indexed in millions of MEDLINE records) can cause individual calls to approach the timeout. Very large PMID sets are expected for high-level disease terms.

**Corpus size is computed per relation type** — the D-S corpus is `union(disease PMIDs) ∩ union(symptom PMIDs)`; the D-A corpus uses anatomy PMIDs; the D-D corpus is `union(disease PMIDs)` alone. Enrichment values are not comparable across the three output tables.
