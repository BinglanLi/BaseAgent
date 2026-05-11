# Evolutionary Rate Covariation (ERC) Operational Reference

---

## Setup

No credentials required. The dataset is a public Dryad archive, but the download is protected by Anubis proof-of-work and AWS WAF JavaScript challenges.

**databases.yaml entry:**
```yaml
evolutionary_rate_covariation:
  enabled: true
  args:
    url: "https://datadryad.org/downloads/file_stream/3949673"
    file_path: "ERC_matrices/mammal_ftERC.RDS"
```

`url` is the Dryad file_stream endpoint for the ZIP archive. `file_path` is the path to the RDS matrix inside that ZIP. Both are required constructor args.

An optional `ft_threshold` arg sets the minimum Fisher-transformed ERC score to retain a gene pair. Default: `sqrt(120 - 3) ≈ 10.82` (derived from 120 mammalian species in the analysis).

**Requires:** `playwright` with Chromium (`playwright install chromium`), `pyreadr`, `numpy`.

---

## Source

**Dataset:** Clark et al. mammal functional ERC matrices, archived on Dryad.
- **Dryad dataset page:** `https://datadryad.org/dataset/doi:10.5061/dryad.6m905qg8q`
- **Full ZIP archive:** ~7.5 GB (ZIP64 format); contains multiple ERC matrices.
- **Target file inside ZIP:** `ERC_matrices/mammal_ftERC.RDS`

### Data Format: RDS Matrix

`mammal_ftERC.RDS` is an R data file (RDS) containing a square, symmetric gene × gene matrix of Fisher-transformed ERC scores. Dimensions are roughly 7,000–9,000 genes. Row and column labels are HGNC gene symbols.

| Property | Value |
|---|---|
| Format | R RDS (read with `pyreadr`) |
| Shape | N × N where N = number of mammalian orthologs |
| Values | Fisher-transformed ERC scores (float) |
| Row/column labels | HGNC gene symbols |
| Matrix symmetry | Symmetric; parser uses upper triangle only (no self-loops) |

**Fisher-transformed ERC score:** Each value represents the Fisher-transformed Pearson correlation of evolutionary rates between two genes across ~120 mammalian species. Higher absolute value = stronger coevolution signal. The default threshold (`sqrt(120 - 3) ≈ 10.82`) corresponds to a Pearson r near the significance boundary for 120 data points.

### Download Strategy

The Dryad file_stream URL serves the ZIP but is protected by two layers of bot detection:

1. **Anubis proof-of-work** — SHA-256 mining challenge embedded in the page; solved programmatically.
2. **AWS WAF JavaScript challenge** — requires a real browser context; handled by Playwright (headless Chromium).

The parser uses Playwright to navigate the Dryad page, intercepts the S3 presigned redirect URL from network traffic, then uses **HTTP range requests** to locate and download only `mammal_ftERC.RDS` from the remote ZIP without fetching the full 7.5 GB archive. Download is parallelised across 6 workers in 100 MB chunks.

---

## Known Gotchas

**S3 presigned URLs expire** — the URL intercepted via Playwright is a time-limited AWS presigned URL. Do not cache it between runs; the Playwright step must re-run each download session.

**Dryad file_stream ID may change on dataset update** — `3949673` in the `url` arg is the Dryad internal file_stream ID. If the dataset authors upload a new version, this ID changes and the `databases.yaml` `url` must be updated manually.

**ZIP is ZIP64 format** — the archive exceeds 4 GB, so standard ZIP EOCD parsing fails. The parser implements ZIP64 central directory parsing with HTTP range requests. Do not use Python's standard `zipfile` module to inspect the remote file.

**`pyreadr.read_r()` returns a dict keyed by `None`** — the RDS file contains an unnamed R matrix object. Access it as `result[None]`. Named objects would use a string key; `None` is specific to unnamed RDS exports.

**`ft_threshold` is species-count–dependent** — the default `sqrt(120 - 3)` is calibrated for 120 mammalian species. If the dataset is updated with a different species count, the threshold formula must be revised accordingly.

**Gene identifiers are HGNC symbols, not Entrez IDs** — `source_hgnc` and `target_hgnc` are gene symbols from the RDS row/column labels. Cross-referencing against Gene nodes (Entrez IDs) requires a symbol→Entrez mapping (e.g. from NCBI Gene `Homo_sapiens.gene_info`).

**Playwright requires Chromium to be installed separately** — `pip install playwright` alone is insufficient; `playwright install chromium` must be run to download the browser binary before the first parse.
