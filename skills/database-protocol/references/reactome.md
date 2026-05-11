# Reactome Operational Reference

---

## Setup

No API key or account required. Both files are direct public downloads.

**databases.yaml entry:**
```yaml
reactome:
  enabled: true
  args: {}
```

No `args` are needed; the download URLs are fixed constants in the parser.

---

## Source Files

Two tab-separated files from the Reactome download area. Both URLs contain `current/` and always serve the latest release.

| File | URL | Purpose |
|---|---|---|
| `ReactomePathways.txt` | `https://reactome.org/download/current/ReactomePathways.txt` | Pathway node definitions (all species) |
| `NCBI2Reactome_All_Levels.txt` | `https://reactome.org/download/current/NCBI2Reactome_All_Levels.txt` | NCBI Gene → Pathway mappings (all species, all hierarchy levels) |

Both files have **no column header row**; column names are assigned explicitly by the parser.

### ReactomePathways.txt

Three tab-separated columns:

| Column | Description |
|---|---|
| `stable_id` | Reactome stable pathway ID (e.g. `R-HSA-109582`) |
| `pathway_name` | Human-readable pathway name |
| `species` | Species name (e.g. `Homo sapiens`, `Mus musculus`) |

Covers all Reactome species. The parser filters to `species == "Homo sapiens"`.

Human-specific stable IDs use the `R-HSA-` prefix. Other species use their own prefix (e.g. `R-MMU-` for mouse).

### NCBI2Reactome_All_Levels.txt

Six tab-separated columns:

| Column | Description |
|---|---|
| `ncbi_gene_id` | NCBI Entrez Gene ID |
| `reactome_id` | Reactome stable pathway ID |
| `url` | Reactome URL for the pathway entry |
| `event_name` | Pathway or reaction name |
| `evidence_code` | Evidence code (e.g. `IEA`, `TAS`) |
| `species` | Species name |

The parser keeps only `ncbi_gene_id`, `reactome_id`, and `evidence_code` from this file. Filtered to `Homo sapiens` before output.

---

## Known Gotchas

**No column header row in either file** — both files begin directly with data. The parser assigns names explicitly via `names=[...]`. Do not use `header=0`.

**"All Levels" means hierarchical roll-up** — `NCBI2Reactome_All_Levels.txt` contains associations at every level of the pathway hierarchy, not just the most specific leaf pathway. A gene participating in a sub-pathway will also appear mapped to every ancestor pathway. Row counts are substantially larger than a direct-membership file would produce; deduplication is applied but hierarchical redundancy remains.

**`R-HSA-` prefix is species-specific** — Homo sapiens pathways use `R-HSA-`; other species use different prefixes. The `species` column filter is the reliable selector; do not use the ID prefix as a proxy.

**URLs are unversioned** — `current/` always resolves to the latest Reactome release. Record the download date for reproducibility.

**Additional Reactome files exist but are not used by this parser** — `ReactomePathways.txt` and `NCBI2Reactome_All_Levels.txt` are the only two downloaded. Other files available from the same endpoint (`UniProt2Reactome_All_Levels.txt`, `ReactomePathwaysRelation.txt`) provide UniProt-based mappings and pathway hierarchy edges, but are not loaded.
