# NCBI Gene Operational Reference

---

## Setup

No API key or local database required. The parser downloads directly from NCBI FTP.

**databases.yaml entry:**
```yaml
ncbigene:
  enabled: true
  args:
    source_url: "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz"
```

`source_url` can be omitted — the parser defaults to the same URL. Override to point at a local copy or a different organism file.

---

## Source File

`Homo_sapiens.gene_info.gz` — gzipped TSV from NCBI FTP. The file is not versioned; the URL always serves the current release.

Official column order (16 columns, header starts with `#tax_id`):
`tax_id`, `GeneID`, `Symbol`, `LocusTag`, `Synonyms`, `dbXrefs`, `chromosome`, `map_location`, `description`, `type_of_gene`, `Symbol_from_nomenclature_authority`, `Full_name_from_nomenclature_authority`, `Nomenclature_status`, `Other_designations`, `Modification_date`, `Feature_type`

**`dbXrefs` field** — `|`-delimited list of `SourceDB:Identifier` entries, e.g. `MIM:138670|HGNC:HGNC:5|Ensembl:ENSG00000121410`. The source key is everything before the first `:`; the identifier is everything after (may itself contain `:`, e.g. `HGNC:HGNC:5`). Empty values are represented as `-`.

The set of source databases present in `dbXrefs` is dynamic and changes across releases. All `type_of_gene` values are present in the file (protein-coding, ncRNA, pseudo, etc.) — filter as needed for the use case.

---

## Known Gotchas

**`GeneID` (integer) is the stable identifier; `Symbol` is not** — gene symbols can be reassigned or retired. Downstream cross-source merging should key on `GeneID` (Entrez ID), not `Symbol`.

**`dbXrefs` source databases change across releases** — any downstream mapping that depends on a specific source name (e.g. `HGNC`, `MIM`, `Ensembl`) should verify those names are still present after a fresh download.

**No versioned archive** — the FTP URL always serves the latest release. Record the download date for reproducibility.
