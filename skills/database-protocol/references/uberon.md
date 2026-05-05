# UBERON Operational Reference

---

## Setup

No API key, credentials, or special configuration required. Both OBO files are public.

**databases.yaml entry:**
```yaml
uberon:
  enabled: true
  args: {}
```

No `args` are needed; the download URLs are fixed constants in the parser.

**Requires:** `obonet` Python package (`pip install obonet`).

---

## Source Files

Two OBO files downloaded from OBOLibrary. Both URLs serve the current release with no version pin in the path.

| File | URL | Purpose |
|---|---|---|
| `basic.obo` | `http://purl.obolibrary.org/obo/uberon/basic.obo` | Core anatomy ontology; source of all term data |
| `human-view.obo` | `http://purl.obolibrary.org/obo/uberon/subsets/human-view.obo` | Human-specific subset; used to flag `is_human` |

### OBO Term Structure

Terms are `[Term]` stanzas. Fields used:

| OBO field | Format | Description |
|---|---|---|
| `id:` | `UBERON:XXXXXXX` | Term identifier |
| `name:` | plain text | Canonical anatomy name |
| `synonym:` | `"text" TYPE [refs]` | Synonym; TYPE is `EXACT`, `RELATED`, `BROAD`, or `NARROW` — text extracted, type discarded |
| `def:` | `"text" [citations]` | Definition; plain text extracted, citations discarded |
| `xref:` | `PREFIX:ID` | Cross-references; recognized prefixes: `MESH:`, `MSH:`, `FMA:`, `BTO:` |
| `subset:` | tag | Subset membership; relevant values: `uberon_slim`, `non_informative`, `upper_level`, `grouping_class` |
| `is_a:` | `UBERON:XXXXXXX ! label` | Parent term via is-a relationship |
| `relationship:` | `part_of UBERON:XXXXXXX ! label` | Parent term via part-of relationship |
| `is_obsolete:` | `true` | Obsolete terms are excluded |

Only `UBERON:`-prefixed parent IDs are retained in `is_a` and `part_of`; cross-ontology parents (e.g. `GO:`, `CL:`) are dropped.

### Term Filtering (Human Slim)

Three criteria must all pass for a term to be included:

1. Present in `human-view.obo` (`is_human == 1`)
2. `subsets` contains `uberon_slim`
3. `subsets` does **not** contain `non_informative`, `upper_level`, or `grouping_class`

---

## Known Gotchas

**MeSH xrefs use multiple prefix variants** — `MESH:`, `MSH:`, and occasionally `MeSH:` all appear in the OBO xref list. All are captured, but normalization is needed when cross-referencing against external MeSH databases.

**`human-view.obo` contains non-UBERON terms** — the file includes `CL:`, `GO:`, and other ontology IDs. Only `UBERON:`-prefixed entries from this file are used to determine `is_human`.

**URLs are unversioned** — `basic.obo` and `human-view.obo` always serve the current release. Record the download date for reproducibility.

**Synonym type is not preserved** — all synonym types (EXACT, RELATED, BROAD, NARROW) are concatenated into a single pipe-delimited string; downstream code cannot distinguish them.
