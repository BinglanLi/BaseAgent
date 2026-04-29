# Evaluation Metrics

Metrics are grouped by the latest pipeline step whose output they require.

| Section | Triggered after | Artifacts |
|---------|-----------------|-----------|
| After Parser | `processed_tsv` | `data/processed/<source>/<name>.tsv` |
| After Ontology Population | `populated_rdf` | `data/output/*_populated.rdf` |
| After Memgraph Export | `exported_csv` | `data/output/nodes_*.csv, data/output/edges_*.csv` |

Each metric carries a release tier:

| Tier | Label | Meaning |
|------|-------|---------|
| 1 | **Block Release** | Failures prevent a build from shipping. |
| 2 | **Monitor Trends** | Tracked across builds; anomalies trigger investigation. |
| 3 | **Periodic Audit** | Scheduled or manual; not required per build. |

---

## 1. After Parser

### Tier 1 — Block Release

#### Coverage & Completeness

**Direct source download** `binary`  
Whether the data are extracted from direct API calls or web downloads, rather than from a GitHub precomputed resource like hetionet.

**Source database extraction** `binary — Pass/Fail`  
Completion status of a single parser. Failure may be due to frequent API requests, code errors, download errors, incorrect download links, and others.

**TSV structural integrity** `binary — Pass/Fail`  
Whether every row in the TSV has exactly the expected number of tab-delimited fields. This check catches embedded tabs or newlines in string fields that would corrupt rows during OWL population.

**Extracted record counts** `integer`  
Raw record count per source. For example, the number of protein nodes from UniProt.

**Filter pass rate** `float`  
Fraction of raw input records in a TSV that survive disease-relevance or other parser filters. An unexpected low rate signals over-filtering.

#### Mapping Integrity

**Duplication rate per ontology** `float`  
Fraction of duplicates per source. A practical check is counting duplicate identifiers (part of node or edge properties) per source TSV.

### Tier 2 — Monitor Trends

#### Coverage & Completeness

**Null/empty field rate per property** `float`  
Fraction of nodes with missing values for each property, stratified by required versus optional status, revealing incomplete data pulls.

#### Data Quality & Validity

**Identifier format validity rate per namespace** `float`  
Fraction of identifiers matching the expected regex pattern for each namespace (UniProt accession, Entrez Gene ID, HGNC symbol, etc.). This is a cheap, quick check that catches parsing errors without network calls.

**Property value constraint violations** `integer`  
Number of literal values in source TSVs (before string conversion) that violate expected types or ranges defined in the mapping configuration (e.g., non-numeric chromosome position, negative molecular weight). This may be a trivial test.

**Source schema conformance** `binary`  
For each source TSV, whether project-configured ontology terms and ontology population keys are present and non-empty. This catches upstream format changes before they propagate silently.

#### Reproducibility & Stability

**Extraction pipeline errors** `integer` *(not implemented)*  
Count of logged errors and warnings per extractor step per run. This is an early-warning indicator of upstream API changes or schema shifts.  
*Requires capturing live log streams during the pipeline run; not computable post-hoc from TSV artifacts.*

**Pipeline step wall-clock time — extract and export TSV** `HH:MM:SS` *(not implemented)*  
Elapsed time for the extract and export TSV pipeline steps per source. Runtime anomalies often precede data quality degradation.  
*Requires timing instrumentation inside `src/main.py`; not computable post-hoc from TSV artifacts.*

### Tier 3 — Periodic Audit

#### Freshness & Provenance

**Source database version/date** `Major.Minor.Patch | YYYY | build number` *(not implemented)*  
Release version and download date per source per run. This enables reproducibility and audit.  
*Version information is exposed differently by each source (HTTP headers, embedded metadata, filename conventions, API endpoints). No uniform extraction method exists across parsers.*

**Data age relative to source release** `integer (days)` *(not implemented)*  
Days between the source database's latest release and the extracted data. This tracks the currency of the knowledge base.  
*Blocked by Source database version/date not being implemented.*

**Extraction timestamp per source** `date`  
One timestamp per source run, derivable from file modification times in data/processed/. This supports incremental update workflows. Per-node timestamps require schema changes and are deferred to a future enhancement.

#### Data Quality — Sampling-Based

**Identifier resolution sampling rate** `float` *(not implemented)*  
Fraction of a random sample of identifiers per namespace that resolve against the live authoritative registry (UniProt, NCBI Gene, HGNC). This should be scheduled periodically rather than per-build to avoid rate limits and network dependencies.  
*Requires live network calls to external registries; intentionally deferred to periodic scheduled audits.*

---

## 2. After Ontology Population

### Tier 1 — Block Release

#### Coverage & Completeness

**Number of node types** `integer`  
Count of distinct OWL classes instantiated in the populated ontology.

**List of node types** `list[str]`  
Enumeration of all OWL class names for which at least one individual was created.

**Number of edge types** `integer`  
Count of distinct OWL ObjectProperties asserted in the populated ontology.

**List of edge types** `list[str]`  
Enumeration of all OWL ObjectProperty names for which at least one assertion was made.

**Ontology mapping activation rate** `float`  
Fraction of ontology terms (OWL classes and ObjectProperties declared in ontology_mappings.yaml) that successfully produce at least one node or edge. This measures completeness against the project's own declared ontology scope.

#### Ontology Adherence

**OWL class conformance rate** `float` · also needs `data/ontology/*.rdf`  
Fraction of node types in ontology_mappings.yaml that exactly match a project-declared OWL class in the reference ontology. This enforces schema consistency.

**OWL ObjectProperty conformance rate** `float` · also needs `data/ontology/*.rdf`  
Fraction of edge type labels in ontology_mappings.yaml that exactly match a project-declared ObjectProperty in the reference ontology. This enforces schema consistency.

---

## 3. After Memgraph Export

### Tier 1 — Block Release

#### Coverage & Completeness

**Total node count per OWL class** `integer`  
Absolute count of nodes per entity type per build. A zero count for any class is a blocking failure.

**Total edge count per OWL ObjectProperty** `integer`  
Absolute count of edges per relationship type per build. A zero count for any expected property is a blocking failure.

#### Ontology Adherence

**Domain/range constraint violation count** `integer` · also needs `data/ontology/*.rdf`  
Number of edges where the subject or object node type violates the declared domain (source node type) or range (target node type) of the ObjectProperty.

#### Mapping Integrity

**Relationship resolution rate per mapping** `float` · also needs `data/processed/<source>/<name>.tsv`  
Fraction of rows in each relationship source file where both the subject and object identifiers match an existing node in the graph. Low values expose silent join failures that produce zero edges.

**Merge match rate per source** `float` · also needs `data/processed/<source>/<name>.tsv`  
Fraction of rows that successfully merge with an existing node. Only applicable to sources configured with merge: true. Critical because merge failures can silently drop or collide entities.

### Tier 2 — Monitor Trends

#### Coverage & Completeness

**Orphan node rate** `float`  
Fraction of nodes with zero edges per OWL class. An increase signals failed join steps or missing relationship extractors.

**Internal cross-reference resolution rate** `float`  
Fraction of xref property values (e.g., xrefDrugbank, xrefMeSH) that match at least one other node's xref value. This measures within-graph entity linkage.

#### Data Quality & Validity

**Exact-IRI duplicate count** `integer`  
Count of nodes sharing the same IRI. This catches data loading bugs.

**Cross-reference duplicate count** `integer`  
Count of node pairs sharing at least one xref value but holding different IRIs, serving as a computable lower-bound estimate of unresolved duplicates.

**Duplicate edge rate** `float`  
Fraction of redundant subject–predicate–object triples (duplicates/total edges). Raw count in extra key `duplicate_edge_count`. This indicates deduplication failures in the merge step.

#### Graph Structural Integrity

**Largest connected component fraction** `float`  
Fraction of total nodes in the largest weakly connected component, with the count of disconnected components reported as supplementary detail. A sudden drop signals a broken join or missing relationship extractor.

**Average node degree per OWL class** `float`  
Mean number of edges per node type. Anomalously low or extreme high-degree values flag extraction or deduplication issues.

#### Reproducibility & Stability

**Run-to-run entity count delta** `object`  
Per-type delta dict (e.g. `{"Gene": +12, "Drug": 0}`) between consecutive runs on the same source version; `null` when no `--baseline` is provided. Non-zero deltas on identical inputs reveal non-determinism.

**Version-to-version incremental change rate** `float` *(not implemented)*  
Fraction of nodes and edges added, removed, or modified between consecutive source versions. This provides a changelog-level health signal for the knowledge base.  
*Requires comparing node identity (by IRI or xref) across builds, not just aggregate counts. The `--baseline` mechanism handles count delta only.*

**Pipeline step wall-clock time — populate and export graph** `HH:MM:SS` *(not implemented)*  
Elapsed time for the populate and export graph pipeline steps. Runtime anomalies often precede data quality degradation.  
*Requires timing instrumentation inside `src/main.py`; not computable post-hoc from graph CSV artifacts.*

### Tier 3 — Periodic Audit

#### Biological Benchmarking

**Known disease-gene recall rate** `float`  
Fraction of OMIM gene-phenotype entries (genemap2.txt, MIM type = 3) whose gene and disease both appear as connected nodes in the graph, matched by Entrez Gene ID. The benchmark file and version must be pinned.

**Drug-target coverage** `float`  
Fraction of approved drugs in DrugBank with at least one protein target present as a connected node, matched by DrugBank accession. Benchmark scope (approved only vs. all) must be specified.

#### Data Quality — Sampling-Based

**High-degree outlier count per ObjectProperty** `integer`  
Number of nodes exceeding the 99th-percentile degree threshold for each relationship type. This is a proxy for cardinality violations that does not require formal OWL cardinality axioms in the reference ontology.
