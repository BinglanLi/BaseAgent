# BindingDB Operational Reference

---

## Setup

No API key or account required. Data is a direct bulk download from the BindingDB website.

**databases.yaml entry:**
```yaml
bindingdb:
  enabled: true
  args: {}
```

No `args` are needed; the download page URL is a fixed constant in the parser.

---

## Source File

`BindingDB_All_YYYYMM_tsv.zip` — monthly-stamped ZIP archive containing a single large TSV (`BindingDB_All*.tsv`). ~500 MB compressed; several GB uncompressed. ~2.9 million binding measurements, ~1 million compounds, ~9,000 targets.

**Download page:**
```
https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp
```

**URL pattern for the bulk TSV zip:**
```
https://www.bindingdb.org/rwd/bind/downloads/BindingDB_All_YYYYMM_tsv.zip
```

The parser discovers the current URL by scraping the download page for the `BindingDB_All_\d+_tsv.zip` pattern. If discovery fails, it falls back to a hardcoded recent URL (updated in the parser source). Both methods return the same file.

### Key TSV Columns

BindingDB exports 50+ columns. The columns most relevant to KG construction:

| Column | Description |
|---|---|
| `DrugBank ID of Ligand` | DrugBank identifier (e.g. `DB00001`); may be empty |
| `Target Name` | Free-text protein target name as recorded by the curator |
| `Target Source Organism According to Curator or DataSource` | Organism string; human entries contain `Homo sapiens` |
| `UniProt (SwissProt) Primary ID of Target Chain` | UniProt accession of the target (may be empty) |
| `UniProt (SwissProt) Entry Name of Target Chain` | UniProt entry name |
| `Ligand SMILES` | SMILES string of the compound |
| `Ligand InChI` | InChI string |
| `ChEMBL ID of Ligand` | ChEMBL identifier (may be empty) |
| `Ki (nM)` | Inhibition constant in nM |
| `Kd (nM)` | Dissociation constant in nM |
| `IC50 (nM)` | Half-maximal inhibitory concentration in nM |
| `EC50 (nM)` | Half-maximal effective concentration in nM |
| `kon (M-1-s-1)` | Association rate constant |
| `koff (s-1)` | Dissociation rate constant |
| `pH` | Assay pH |
| `Temp (C)` | Assay temperature in °C |
| `Curation/DataSource` | Source of the measurement (publication, database, etc.) |
| `Article DOI` | DOI of the source article |
| `PMID` | PubMed ID of the source article |

---

## Known Gotchas

**ZIP URL contains a date stamp that changes monthly** — `BindingDB_All_YYYYMM_tsv.zip` is minted each release. The parser discovers the current URL dynamically from the download page. If the download page layout changes, the fallback URL in the parser source must be manually updated.

**File is very large** — the ZIP is ~500 MB and the extracted TSV is several GB. The parser uses streaming download with no read timeout (30 s connect timeout only) and loads only 3 columns via `usecols` to keep memory usage manageable. Do not `pd.read_csv` the full file without `usecols`.

**`Target Name` is free text, not a controlled identifier** — it is not a gene symbol or UniProt accession. Cross-referencing against Gene nodes in the KG requires a separate protein name → gene symbol mapping step; direct exact-match joins will miss many valid entries.

**Many measurements lack a DrugBank ID** — only rows with a populated `DrugBank ID of Ligand` are used. Compounds identified only by SMILES, ChEMBL ID, or CAS number are excluded. This reduces the ~2.9 M total measurements substantially.

**Organism filter uses substring matching** — the organism column contains free-text curator notes. The filter checks for `Homo sapiens` (case-insensitive substring); entries with typos or additional qualifiers (e.g. `recombinant Homo sapiens`) still pass.

**Binding affinity columns are not loaded by the parser** — Ki, Kd, IC50, and EC50 values are available in the source file but the parser produces only binary drug–target edges (`drug_binds_gene.tsv`) without affinity values. To use affinity data, load additional columns via `usecols`.
