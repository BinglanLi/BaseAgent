# Gene Ontology Operational Reference

---

## Setup

No API key or account required. Both files are public.

**databases.yaml entry:**
```yaml
gene_ontology:
  enabled: true
  args: {}
```

No `args` are needed; the download URLs are fixed constants in the parser.

**Requires:** `obonet` Python package (`pip install obonet`).

**NCBI gene_info dependency:** Gene–GO association files use gene symbols from the GOA file, which must be mapped to Entrez Gene IDs using `Homo_sapiens.gene_info` from the NCBI Gene source. The parser looks for it first in `data/raw/ncbigene/`, then in `data/raw/gene_ontology/`. Run the ncbigene parser before gene_ontology, or place the file manually.

---

## Source Files

Two files downloaded from OBOLibrary and the Gene Ontology Annotation (GOA) FTP. Both URLs always serve the current release.

| File | URL | Purpose |
|---|---|---|
| `go.obo` | `http://purl.obolibrary.org/obo/go.obo` | Full GO ontology; source of BP/MF/CC term data |
| `goa_human.gaf.gz` | `http://current.geneontology.org/annotations/goa_human.gaf.gz` | Human gene–GO annotations in GAF 2.2 format |

### go.obo — OBO Term Structure

Terms are `[Term]` stanzas. Fields used:

| OBO field | Format | Description |
|---|---|---|
| `id:` | `GO:XXXXXXX` | Term identifier |
| `name:` | plain text | GO term label |
| `namespace:` | `biological_process` / `molecular_function` / `cellular_component` | Determines which of the three node types this term belongs to |
| `def:` | `"text" [citation]` | Definition; plain text extracted, citation bracket discarded |
| `is_obsolete:` | `true` | Obsolete terms are excluded |

Only `GO:`-prefixed term IDs are retained. Obsolete terms are skipped.

### goa_human.gaf.gz — GAF 2.2 Annotation Format

Lines starting with `!` are comments and are skipped. Remaining lines are tab-separated with 17 columns:

| Column | Name | Description |
|---|---|---|
| 1 | `DB` | Database source (e.g. `UniProtKB`) |
| 2 | `DB_Object_ID` | Database object identifier (UniProt accession) |
| 3 | `DB_Object_Symbol` | Gene symbol; used for Entrez ID mapping |
| 4 | `Qualifier` | Annotation qualifier (e.g. `enables`, `NOT`) |
| 5 | `GO_ID` | GO term ID (e.g. `GO:0008150`) |
| 6 | `DB_Reference` | Literature/database reference for the annotation |
| 7 | `Evidence_Code` | GO evidence code (e.g. `IDA`, `IEA`, `TAS`, `HDA`) |
| 8 | `With_From` | Supporting evidence (optional; may be empty) |
| 9 | `Aspect` | GO namespace: `P`=Biological Process, `F`=Molecular Function, `C`=Cellular Component |
| 10 | `DB_Object_Name` | Full protein name |
| 11 | `DB_Object_Synonym` | Gene synonyms (pipe-delimited) |
| 12 | `DB_Object_Type` | Object type (e.g. `protein`) |
| 13 | `Taxon` | Taxonomy string (e.g. `taxon:9606`); must contain `taxon:9606` for human |
| 14 | `Date` | Annotation date (YYYYMMDD) |
| 15 | `Assigned_By` | Database that made the annotation |
| 16 | `Annotation_Extension` | Optional cross-ontology relations |
| 17 | `Gene_Product_Form_ID` | Isoform identifier (optional) |

Only rows where `Taxon` contains `taxon:9606` are retained.

---

## Known Gotchas

**GO namespace determines node type** — all three term types (BP, MF, CC) are stored in the same `go.obo` file and distinguished only by the `namespace:` field. There is no separate per-namespace file.

**`Aspect` codes are single letters** — `P` = biological_process, `F` = molecular_function, `C` = cellular_component. The full namespace name in the OBO `namespace:` field and the one-letter `Aspect` code in the GAF file use different conventions; they must be matched separately.

**Gene identifiers in GAF are symbols, not Entrez IDs** — `DB_Object_Symbol` is the HGNC gene symbol. Conversion to Entrez Gene IDs requires `Homo_sapiens.gene_info` from NCBI Gene. If this file is absent, all three association DataFrames will be empty.

**URLs are unversioned** — `go.obo` and `goa_human.gaf.gz` always serve the current release; `current/` in the GOA URL resolves to the latest dated snapshot. Record the download date for reproducibility.

**`go.obo` vs `go-basic.obo`** — The parser downloads `go.obo` (the full ontology including cross-ontology links) but falls back to a locally cached `go-basic.obo` if present. For reproducibility, note which file was actually used; the two files differ in inter-ontology relationship coverage.

**`Qualifier` field may contain `NOT`** — some annotations assert that a gene does _not_ have a given GO function. The parser does not filter these out; downstream KG construction should handle `NOT` qualifiers if needed.
