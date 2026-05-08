# MeSH Operational Reference

---

## Setup

No API key or account required. The XML file is a direct NLM download.

**databases.yaml entry:**
```yaml
mesh:
  enabled: true
  args:
    base_url: "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/"
```

`base_url` points to the NLM directory listing for MeSH XML files. The parser appends `desc{year}.xml` and tries years in descending order (2026, 2025, 2024, 2023) until a download succeeds.

**Requires:** `lxml` Python package (`pip install lxml`).

---

## Source File

`desc{year}.xml` — annual MeSH full descriptor file from NLM. ~300 MB uncompressed. Contains all MeSH descriptor types (Topical, Geographic, Publication Type, Check Tag, etc.) across all tree branches.

**File URL pattern:**
```
https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc{year}.xml
```

New files are published each year before January 1. NLM does not maintain a stable `latest` alias — the year must be embedded in the filename.

### XML Structure

The file is a sequence of `<DescriptorRecord>` elements. Fields used:

| XML path | Description |
|---|---|
| `DescriptorUI` | MeSH Descriptor UI (D-code, e.g. `D005221`) |
| `DescriptorName/String` | Preferred term name |
| `TreeNumberList/TreeNumber` | One or more MeSH tree numbers (e.g. `C23.888.592`) |

Each `<DescriptorRecord>` may have multiple `<TreeNumber>` entries if the concept appears in more than one tree branch. The tree numbers are used for filtering only and are not written to the output.

### Tree Number Hierarchy (MeSH C tree)

The C tree covers Diseases. The relevant subtree for symptom extraction:

| Tree code | Heading |
|---|---|
| `C23` | Pathological Conditions, Signs and Symptoms |
| `C23.888` | Signs and Symptoms (root descriptor: D012816) |
| `C23.888.*` | All descendant symptom descriptors |

Only descriptors with at least one tree number matching the prefix `C23.888.` (trailing dot) are included. The root descriptor `D012816` (Signs and Symptoms) itself is excluded by requiring the trailing dot.

---

## Known Gotchas

**Year-based filenames require annual update** — there is no `latest` or `current` URL alias. The `_CANDIDATE_YEARS` list in the parser must be updated each year to include the new release year. The parser tries years in descending order and stops at the first successful download.

**File is large (~300 MB)** — parsing uses `etree.iterparse` for streaming to avoid loading the full XML tree into memory. Do not use `etree.parse` on the full file.

**Full descriptor file, not a subset download** — NLM does not offer per-subtree downloads. The full `desc{year}.xml` is downloaded and filtered locally to the `C23.888.` subtree. Expect ~30,000+ total descriptors; the Signs and Symptoms subtree yields ~400–500 after filtering.

**`C23.888` root is excluded intentionally** — the prefix filter uses `C23.888.` with a trailing dot, so the root descriptor D012816 ("Signs and Symptoms") is not included in the output. Only its descendants are extracted.

**Tree numbers are not stable across years** — NLM occasionally reassigns tree numbers between annual releases. Do not use tree numbers as stable identifiers; use the Descriptor UI (D-code) for cross-year matching.
