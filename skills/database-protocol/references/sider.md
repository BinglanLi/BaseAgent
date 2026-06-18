# SIDER Operational Reference

---

## Setup

No API key or account required. All files are direct bulk downloads.

**databases.yaml entry:**
```yaml
sider:
  enabled: true
  args: {}
```

No `args` are needed; all download URLs are fixed constants in the parser.

---

## Source Files

### `meddra_all_se.tsv.gz`

Drug–side-effect associations, one row per (drug, concept type, side-effect term) triple. The parser filters to `PT` (Preferred Term) rows only.

**Download URL:**
```
http://sideeffects.embl.de/media/download/meddra_all_se.tsv.gz
```

**Columns (6 total, tab-separated, no header):**

| Column | Name | Description |
|---|---|---|
| 1 | `stitch_id_flat` | STITCH flat compound ID (e.g. `CID100002441`); stereochemistry-collapsed |
| 2 | `stitch_id_stereo` | STITCH stereo compound ID; identifies a specific enantiomer or stereoisomer |
| 3 | `umls_label` | UMLS CUI of the source label term before PT canonicalization (may differ from col 5) |
| 4 | `meddra_concept_type` | `PT` (Preferred Term) or `LLT` (Lower-Level Term) |
| 5 | `umls_cui` | UMLS CUI of the MedDRA concept (always the PT CUI when col 4 is `PT`) |
| 6 | `side_effect_name` | MedDRA term label |

### `meddra.tsv.gz`

Cross-reference from UMLS CUI to numeric MedDRA concept ID. Used to obtain the `meddra_id` needed to join against ChemicalEffect nodes (populated by DrugCentral via `xrefMedDRA`).

**Download URL:**
```
http://sideeffects.embl.de/media/download/meddra.tsv.gz
```

**Columns (4 total, tab-separated, no header):**

| Column | Name | Description |
|---|---|---|
| 1 | `umls_cui` | UMLS CUI |
| 2 | `meddra_concept_type` | `PT` or `LLT` |
| 3 | `meddra_id` | Numeric MedDRA concept ID (e.g. `10000647`) |
| 4 | `meddra_name` | MedDRA term label |

---

## Processing

1. **PT filter** — both files are filtered to `meddra_concept_type == "PT"`. This keeps one canonical row per (drug, side effect) pair and drops LLT synonyms.

2. **STITCH → PubChem CID** — STITCH flat IDs use the format `CID1XXXXXXXX`, where `XXXXXXXX` is a zero-padded 8-digit PubChem CID. The parser strips the `CID1` prefix and converts to an integer string to remove leading zeros (e.g. `CID100002441` → `"2441"`). This is the cross-reference key for Drug nodes (`xrefPubchemCID`).

3. **meddra cross-reference join** — the meddra file is deduplicated to one row per `umls_cui` (PT only) and inner-joined onto the edge table. Rows without a mappable numeric MedDRA ID are dropped because they cannot match any ChemicalEffect node.

4. **Output** — one file, `chemical_causes_effect.tsv`:

| Column | Description |
|---|---|
| `pubchem_cid` | PubChem Compound ID — cross-references Drug nodes via `xrefPubchemCID` |
| `meddra_id` | Numeric MedDRA concept ID — cross-references ChemicalEffect nodes via `xrefMedDRA` |
| `source_database` | Literal `"SIDER"` |

---

## Cross-Reference Strategy

SIDER does not provide Drug or ChemicalEffect nodes. Its edges cross-reference nodes populated by other parsers:

| Edge endpoint | Column | Match property | Populated by |
|---|---|---|---|
| Drug (subject) | `pubchem_cid` | `xrefPubchemCID` | DrugCentral, CTD |
| ChemicalEffect (object) | `meddra_id` | `xrefMedDRA` | DrugCentral |

DrugCentral must be enabled and parsed before SIDER edges will match. If DrugCentral is disabled, all SIDER edges will silently produce zero matches.

---

## Known Gotchas

**URL uses `http://`, not `https://`** — the SIDER download endpoint is unencrypted HTTP. Do not substitute `https://` as it is not served on that scheme.

**`meddra_all_se` has no header row** — the file uses no `#`-prefixed comments and no header line; columns must be assigned explicitly. Do not use `header=0`.

**STITCH flat vs. stereo IDs** — the flat ID (`stitch_id_flat`) collapses all stereoisomers to one compound; the stereo ID (`stitch_id_stereo`) distinguishes enantiomers. Most drugs have one stereo form, but ~112 have two and ~7 have three. The parser uses the flat ID (converted to PubChem CID) because drug nodes from other sources do not carry stereo-level identifiers.

**~22 % of PT rows have `umls_label != umls_cui`** — the original drug label used an LLT (`umls_label`) that maps up to a different PT (`umls_cui`). The parser discards `umls_label`; only the canonical PT CUI is used for the meddra join.

**SIDER is no longer actively maintained** — the last release was SIDER 4.1 (October 2015). The download URLs remain live but the data will not be updated. Cross-reference coverage against newer drug nodes may be lower than for actively maintained sources.

**Inner join drops unmatched side effects** — approximately 3–5 % of PT rows lack a numeric MedDRA ID in the meddra cross-reference file. These are silently dropped by the inner join. Use a left join and inspect null `meddra_id` rows if completeness matters.
