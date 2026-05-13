# PubTator3 Operational Reference

---

## Setup

No API key required for basic use. A NCBI API key raises the rate limit from 3 to 10 requests/second.

**databases.yaml entry:**
```yaml
pubtator3:
  enabled: true
  args:
    base_url: "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"
    ncbi_api_key_env: NCBI_API_KEY   # optional; omit if not using a key
```

**Bulk FTP download** (avoids per-request rate limits; preferred for full-corpus extraction):
```
https://ftp.ncbi.nlm.nih.gov/pub/lu/PubTator3/
```
Key files: `bioconcepts2pubtatorcentral.gz` (all entity annotations), relation files per relation type (e.g. `gene2disease.gz`).

---

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/publications/search?text={query}&page={n}` | GET | Search articles by free text or entity ID; returns PMIDs |
| `/publications/export/biocjson?pmids={ids}` | GET | Fetch BioC JSON annotations for up to 100 PMIDs per request |
| `/entity/pmids/{entity_id}` | GET | Find articles mentioning a specific entity (by normalized ID) |
| `/relations?pmids={ids}&type={relation}` | GET | Extract relation triples from a set of PMIDs |

Pagination via `page` parameter (1-indexed). Delay 0.34 s between requests without a key; 0.1 s with one.

---

## Available Data

### Entity Types

PubTator3 annotates six bio-entity types using the AIONER model:

| Entity Type | Normalization Vocabulary | ID Format |
|---|---|---|
| Gene | NCBI Gene | Entrez Gene ID (integer) |
| Disease | MeSH | MeSH UID (e.g. `D000544`) |
| Chemical | MeSH | MeSH UID (e.g. `D000001`) |
| Variant | dbSNP (preferred) / HGVS | `rs{number}` or HGVS string |
| Species | NCBI Taxonomy | NCBI Taxonomy ID |
| Cell Line | Cellosaurus | Cellosaurus accession |

### Relation Types

PubTator3 extracts 12 relation types using BioREx. The `CONVERT` token appears in the API but is undocumented in the tutorial.

| Relation | Entity Pairs (domain → range) | Notes |
|---|---|---|
| `TREAT` | Chemical → Disease | Drug-disease treatment |
| `CAUSE` | Chemical → Disease, Variant → Disease | Causal associations |
| `ASSOCIATE` | Various | Non-specific association; catch-all |
| `INHIBIT` | Disease → Gene, Chemical → Variant | Negative regulation |
| `STIMULATE` | Disease → Gene, Disease → Variant | Positive regulation |
| `INTERACT` | Gene ↔ Gene, Gene ↔ Chemical, Chemical ↔ Variant | Physical interaction |
| `POSITIVE_CORRELATE` | Chemical → Gene, Gene ↔ Gene | Co-expression / upregulation |
| `NEGATIVE_CORRELATE` | Chemical → Gene, Gene ↔ Gene | Co-expression / downregulation |
| `PREVENT` | Variant → Disease | Protective variant effect |
| `COMPARE` | Chemical ↔ Chemical | Effect comparison between drugs |
| `COTREAT` | Chemical ↔ Chemical | Co-administration |
| `DRUG_INTERACT` | Chemical ↔ Chemical | Pharmacodynamic interaction |

### BioC JSON Annotation Format

Each article in BioC JSON has `passages` (title, abstract) with `annotations` (entity spans) and `relations` (relation triples). Key annotation fields: `id`, `infons.type` (entity type), `infons.identifier` (normalized ID), `text` (surface form), `locations` (character offsets). Relation fields: `infons.type` (relation label), `nodes` (list of `{refid, role}` pairs referencing annotation IDs).

---

## Known Gotchas

**Entity IDs are not always normalized** — some annotations have `infons.identifier` set to `-` (unmapped) or a semicolon-delimited list when a mention is ambiguous across multiple entities. Filter or explode accordingly.

**MeSH IDs for Disease and Chemical have no prefix in annotations** — the raw identifier is the bare UID (e.g. `D000544`), not `MESH:D000544`. Add the prefix when joining against MeSH-keyed KG nodes.

**`ASSOCIATE` is a catch-all and will dominate counts** — it covers all entity pairs and is assigned when a more specific relation type cannot be determined. For edge construction, prefer the specific relation types and treat `ASSOCIATE` as lower-confidence supplementary data.

**Gene IDs are Entrez (integer), not symbol** — cross-reference via NCBI Gene to resolve to HGNC symbol or Ensembl.

**Bulk FTP files are updated monthly** — the FTP dump and the live API may be out of sync by up to a month. Record the FTP file modification date for reproducibility.

**Rate limits are strict without a key** — 3 requests/second hard limit enforced server-side; responses return HTTP 429 on violation. A NCBI API key (free, from https://www.ncbi.nlm.nih.gov/account/) raises this to 10/second.

**`/publications/export/biocjson` accepts max 100 PMIDs per call** — batch larger PMID lists into 100-ID chunks.
