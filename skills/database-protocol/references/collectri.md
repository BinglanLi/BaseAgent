# CollectTRI Operational Reference

---

## Setup

No API key or account required. Data is fetched live from the OmniPath REST API.

**databases.yaml entry:**
```yaml
collectri:
  enabled: true
  args: {}
```

No `args` are needed; the OmniPath API endpoint is a fixed constant in the parser.

---

## Source

CollectTRI is a curated collection of signed TF-target interactions compiled from 12 resources (ChIP-seq, literature, databases). It is distributed via the OmniPath web service rather than as a standalone download.

**API endpoint:**
```
https://omnipathdb.org/interactions?datasets=collectri&genesymbols=1
```

The `genesymbols=1` parameter is required — it adds `source_genesymbol` and `target_genesymbol` columns (HGNC symbols) to the response. Without it the response contains only UniProt accessions.

**Response format:** Tab-separated text (TSV), returned directly as the HTTP response body. No decompression needed.

### Response Columns

| Column | Type | Description |
|---|---|---|
| `source` | string | UniProt accession of the transcription factor |
| `target` | string | UniProt accession of the target gene |
| `source_genesymbol` | string | TF HGNC gene symbol |
| `target_genesymbol` | string | Target gene HGNC gene symbol |
| `is_directed` | int (0/1) | Whether the interaction is directional |
| `is_stimulation` | int (0/1) | TF activates/stimulates the target |
| `is_inhibition` | int (0/1) | TF represses/inhibits the target |
| `consensus_direction` | float | Cross-resource consensus score for directionality |
| `consensus_stimulation` | float | Cross-resource consensus score for stimulation |
| `consensus_inhibition` | float | Cross-resource consensus score for inhibition |

Additional OmniPath metadata columns may appear in the response but are not used.

---

## Known Gotchas

**`databases.yaml` key is `collectri` (one `t`) but the parser file is `collecttri_parser.py`** — the spelling difference is intentional; do not rename either. The key controls `data/processed/collectri/`; the filename is incidental.

**No versioned snapshots** — the OmniPath API always serves the current release. There is no archive URL or version pin. Record the download date for reproducibility.

**Gene identifiers are HGNC symbols, not Entrez IDs** — `source_genesymbol` and `target_genesymbol` are gene symbols. Cross-source merging on gene nodes (which use Entrez IDs) requires a symbol→Entrez mapping (e.g. from NCBI Gene `Homo_sapiens.gene_info`).

**`is_stimulation` and `is_inhibition` are not mutually exclusive** — a small fraction of records have both flags set, reflecting conflicting evidence across the 12 source resources. Downstream KG construction should handle this ambiguity.

**Timeout is 120 seconds** — the full CollectTRI dataset is ~18,000–20,000 interactions. OmniPath is occasionally slow under load; the parser sets a 120 s timeout.
