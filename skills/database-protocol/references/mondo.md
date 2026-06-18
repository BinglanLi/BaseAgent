# MONDO Operational Reference

---

## Setup

No API key or account required. All files are direct bulk downloads.
---

## Source Files

### `mondo.obo`

Full Monarch Disease Ontology in OBO format. One node per disease term; cross-references to external vocabularies are embedded in `xref:` fields.

**Download URL:**
```
http://purl.obolibrary.org/obo/mondo.obo
```

**Relevant OBO fields parsed per term:**

| Field | Description |
|---|---|
| `id` | MONDO CURIE (e.g. `MONDO:0004975`) |
| `name` | Disease name |
| `def` | Text definition; OBO quotes and citation brackets are stripped |
| `synonym` | Synonym strings; extracted text is pipe-delimited in output |
| `is_a` | Parent MONDO IDs; pipe-delimited in output |
| `xref` | Cross-references — UMLS, OMIM, Orphanet, DOID, ICD10CM/ICD10, MESH, SNOMEDCT |
| `subset` | Subset tags used to exclude grouping/non-disease terms |
| `is_obsolete` | Obsolete flag; obsolete terms are excluded |

---

## Processing

1. **Prefix filter** — only terms with `id` starting with `MONDO:` are retained. Non-MONDO namespace terms (e.g. HP, GO) are skipped.

2. **Obsolete filter** — terms with `is_obsolete: true` are excluded.

3. **Subset filter** — terms tagged with any of `non_grouping`, `grouping_class`, `upper_level`, or `non_informative` are excluded. These mark administrative or hierarchical grouping nodes, not actual diseases.

4. **Scope filter** — if `disease_scope` in `project.yaml` is non-null, only terms matching at least one configured scope criterion are kept (MONDO ID match, OMIM ID match, UMLS CUI match, or disease name substring match). If `disease_scope` is null, all non-obsolete terms pass.

5. **xref extraction** — each prefix is extracted from the `xref:` list. For xrefs with multiple values (e.g. several OMIM IDs), only the first is stored. ICD-10-CM (`ICD10CM:`) is preferred over WHO ICD-10 (`ICD10:`); the fallback is used only when `ICD10CM:` is absent.

6. **Output** — one file, `disease_nodes.tsv`:

| Column | Description |
|---|---|
| `mondo_id` | MONDO ID (e.g. `MONDO:0004975`) — IRI column |
| `disease_name` | Disease name |
| `definition` | Text definition (OBO quotes and citation brackets stripped) |
| `synonyms` | Pipe-delimited synonym strings |
| `is_a` | Pipe-delimited parent MONDO IDs via `is_a` relationships |
| `umls_cui` | First UMLS CUI cross-reference |
| `omim_id` | First OMIM ID cross-reference |
| `orpha_id` | First Orphanet ID cross-reference |
| `doid` | First DOID cross-reference |
| `icd10cm` | First ICD-10-CM code (falls back to ICD-10 if absent) |
| `mesh_id` | First MeSH descriptor ID cross-reference |
| `snomed_id` | First SNOMED CT ID cross-reference |
| `source_database` | Literal `"MONDO"` |

---

## Cross-Reference Strategy

MONDO provides Disease nodes only. Its `mondo_id` column is the IRI used to create individuals; downstream parsers that populate Disease relationships must cross-reference by a shared property.

| Property | Source column | Populated by |
|---|---|---|
| `xrefMondoID` | `mondo_id` | MONDO |
| `xrefUmlsCUI` | `umls_cui` | MONDO, DisGeNET, DrugCentral |
| `xrefDiseaseOntology` | `doid` | MONDO, Disease Ontology |
| `xrefOMIM` | `omim_id` | MONDO, Disease Ontology |
| `xrefMeSH` | `mesh_id` | MONDO, Uberon, MeSH |
| `xrefSNOMED` | `snomed_id` | MONDO |
| `xrefICD10CM` | `icd10cm` | MONDO, Disease Ontology |

MONDO and Disease Ontology both populate Disease nodes. If both are enabled with `merge: false`, duplicate Disease individuals may be created. Set one to `merge: true` with a shared `merge_column` (e.g. `xrefUmlsCUI` or `xrefDiseaseOntology`) to coalesce them.

---

## Known Gotchas

**`orpha_id` is not mapped in `ontology_mappings.yaml`** — the parser extracts Orphanet IDs into `orpha_id`, but no `xrefORDO` mapping is currently defined for MONDO in `ontology_mappings.yaml`. Add a `orpha_id: xrefORDO` entry to `data_property_map` if Orphanet cross-references are needed.

**Multiple xrefs per prefix are truncated** — for xref types with multiple values (common for OMIM and UMLS), only the first is stored. Diseases with several OMIM entries (e.g. locus heterogeneity) will lose all but the first.

**Scope filter requires `disease_scope` to be non-null** — if `disease_scope: null` in `project.yaml`, all ~27 000 non-obsolete terms pass. Parsing and populating the full ontology is slow; enable scoping for disease-specific KG builds.

**MONDO subsumes older disease vocabularies** — MONDO integrates and cross-references OMIM, Orphanet, DOID, and others. It is intentionally broader and more granular than Disease Ontology slim terms. Enabling both MONDO and Disease Ontology will substantially increase Disease node count.

**`mondo.obo` is updated frequently** — releases occur approximately weekly. The parser always downloads the current version from the canonical URL; pin to a dated release URL if reproducibility across runs is required.