# Bgee Operational Reference

---

## Setup

No API key or account required. The file is a direct FTP download.

**databases.yaml entry:**
```yaml
bgee:
  enabled: true
  args:
    source_url: "https://www.bgee.org/ftp/current/download/calls/expr_calls/Homo_sapiens_expr_simple.tsv.gz"
    tissue_filter: null
```

`source_url` points to the current-release file for Homo sapiens. `tissue_filter` accepts a list of UBERON IDs to restrict output (e.g. `["UBERON:0001876", "UBERON:0000955"]`); `null` includes all anatomies.

---

## Source File

`Homo_sapiens_expr_simple.tsv.gz` — gzipped TSV, one row per gene–anatomy expression call. The URL path contains `current/` and always serves the latest release.

**Columns:**

| Column | Type | Description |
|---|---|---|
| `Gene ID` | string | Ensembl gene ID (e.g. `ENSG00000000003`) |
| `Gene name` | string | HGNC gene symbol; double-quoted in raw file |
| `Anatomical entity ID` | string | Ontology term ID; includes both `UBERON:` and `CL:` prefixes |
| `Anatomical entity name` | string | Anatomy label; double-quoted in raw file |
| `Expression` | string | `present` or `absent` |
| `Call quality` | string | `gold quality` or `silver quality` |
| `FDR` | float | False Discovery Rate for the expression call |
| `Expression score` | float | 0–100; higher = higher relative expression |
| `Expression rank` | float | Lower rank = higher expression; large values appear in scientific notation (e.g. `1.75e4`) |

---

## Known Gotchas

**`Anatomical entity ID` is not exclusively UBERON** — the file contains both `UBERON:` (anatomy) and `CL:` (Cell Ontology) identifiers. Only `UBERON:`-prefixed rows map to BodyPart nodes in the knowledge graph.

**`tissue_filter` acts after the UBERON prefix filter** — providing a list of UBERON IDs refines further; it does not re-admit `CL:` entries.

**No versioned archive in the URL** — `current/` always resolves to the latest release. Record the download date for reproducibility.

**Gene identifiers are Ensembl only** — the file does not include Entrez Gene IDs or UniProt accessions. Cross-source merging on genes requires an Ensembl→Entrez mapping (e.g. from NCBI Gene `dbXrefs`).
