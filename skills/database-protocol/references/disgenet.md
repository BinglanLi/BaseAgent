# DisGeNET Operational Reference

---

## Setup

**Requirements:**
- DisGeNET API key (free academic registration at https://www.disgenet.org/signup)
- Set as the `DISGENET_API_KEY` environment variable

**databases.yaml entry:**
```yaml
disgenet:
  enabled: true
  args:
    api_key_env: DISGENET_API_KEY
    base_url: "https://www.disgenet.org/api"
```

**Disease scope** — the parser never queries the full database. It targets only the disease CUIs and search terms configured under `disease_scope` in `config/project.yaml`:
```yaml
disease_scope:
  primary_terms:
    - "<disease_term>"    # e.g. "parkinson"
  umls_cuis:
    - "<UMLS_CUI>"        # e.g. "C0030567"
```
`umls_cuis` are queried directly; `primary_terms` drive `GET /disease/search` to discover additional CUIs.

---

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/disease/search?q={term}` | GET | Discover disease CUIs by name |
| `/gda/disease/{diseaseid}?page_number={n}` | GET | Fetch GDAs for one CUI (100 records/page) |

Pagination stops when a page returns fewer than 100 records or a 404. Rate-limit responses (429) trigger a 10 s sleep and retry. A 0.5 s delay is inserted between CUI queries; 0.3 s between pages.

---

## Available Data

**GDA records** — one row per gene–disease pair. Key fields: `geneId` (NCBI Entrez), `geneSymbol`, `diseaseId` (UMLS CUI), `diseaseName`, `diseaseType` (`disease`/`group`/`phenotype`), `diseaseClasses_MSH`, `diseaseClasses_UMLS_ST`, `score` (GDA score 0–1), `EI` (evidence index 0–1), `nPubs`, `nSnps`, `diseaseMapping`, `gene_dsi`, `gene_dpi`, `gene_pli`.

**`diseaseMapping` field** — comma-separated `VOCAB_CODE` tokens encoding cross-references for the disease, e.g. `MESH_D000544,DO_0010524,ICD10_G30`. Split on `_` to get vocab and code; the identifier follows the first `_`. Recognized vocabularies: `MESH`/`MSH`, `ICD10`, `NCI`, `OMIM`, `ICD9CM`, `HPO`, `DO`, `MONDO`, `UMLS`, `EFO`, `ORDO`. A disease may have multiple codes per vocabulary; only the first is reliable since ordering is not guaranteed.

**Gene functional scores** returned per GDA row:
- `gene_dsi` — Disease Specificity Index (0–1; higher = more disease-specific)
- `gene_dpi` — Disease Pleiotropy Index (0–1; higher = more disease classes)
- `gene_pli` — Probability of LoF intolerance (gnomAD)

---

## Known Gotchas

**Gene name/Ensembl/UniProt not in GDA endpoint** — `geneName`, `ensemblId`, and `proteinId` are not returned by `/gda/disease/`. Supplement from a gene-info source (e.g. NCBI Gene) if these are needed.

**API field names vary by version** — older responses use `Npubmeds`/`NofSnps`; current responses use `nPubs`/`nSnps`. Both forms exist in the wild.

**`DISGENET_API_KEY` value, not name** — `databases.yaml` uses `api_key_env: DISGENET_API_KEY` to name the env var. Do not put the literal key value in `databases.yaml`.
