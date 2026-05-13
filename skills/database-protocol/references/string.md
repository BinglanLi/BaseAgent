# STRING Operational Reference

---

## Setup

No account or API key needed for bulk downloads.

**databases.yaml entry:**
```yaml
string:
  enabled: true
  args:
    version: "12.0"
    taxon_id: 9606
    score_threshold: 700
```

`score_threshold` applies to `combined_score` (0–1000 scale): 400 = medium confidence, 700 = high, 900 = very high.

---

## Source Files

Current version: **v12.0** (May 2023). Download index:
```
https://stringdb-static.org/download/
```

URL pattern for human files:
```
https://stringdb-static.org/download/{file_type}.v12.0/9606.{file_type}.v12.0.txt.gz
```

| File | Size | Contents |
|---|---|---|
| `9606.protein.links.v12.0.txt.gz` | 79.3 MB | All human PPI pairs + `combined_score` only |
| `9606.protein.links.detailed.v12.0.txt.gz` | 133.2 MB | Pairs + per-channel subscores |
| `9606.protein.physical.links.v12.0.txt.gz` | 8.5 MB | Physical subnetwork (experimental + curated evidence only) |
| `9606.protein.info.v12.0.txt.gz` | — | Preferred gene/protein name per ENSP |
| `9606.protein.aliases.v12.0.txt.gz` | — | Cross-references to Entrez, ENSG, UniProt, gene symbols |

### `protein.links` columns

Space-delimited (not tab):

| Column | Description |
|---|---|
| `protein1` | STRING ID of first protein (e.g. `9606.ENSP00000000233`) |
| `protein2` | STRING ID of second protein |
| `combined_score` | Integrated confidence score (0–1000) |

Pairs are undirected and listed once.

### `protein.links.detailed` additional columns

`neighborhood`, `neighborhood_transferred`, `fusion`, `cooccurence`, `homology`, `coexpression`, `coexpression_transferred`, `experiments`, `experiments_transferred`, `database`, `database_transferred`, `textmining`, `textmining_transferred`, `combined_score`.

Each `_transferred` column reflects evidence from other species propagated via orthology.

### `protein.info` columns

Tab-delimited: `#string_protein_id`, `preferred_name` (gene symbol), `protein_size`, `annotation`.

### `protein.aliases` columns

Tab-delimited: `#string_protein_id`, `alias`, `source`. Key `source` values for gene-level mapping:

| Source tag | Maps to |
|---|---|
| `Ensembl_HGNC` | Ensembl gene ID (ENSG) |
| `NCBI_gene` | Entrez Gene ID |
| `BioMart_HUGO` | HGNC gene symbol |

---

## License

**CC BY 4.0** — academic and commercial use permitted with attribution.
Cite: Szklarczyk et al., *Nucleic Acids Research*, 2023 (STRING v12.0).

---

## Known Gotchas

**STRING IDs are Ensembl protein (ENSP) IDs**, prefixed with the taxon ID (`9606.ENSP...`). They are not gene or transcript IDs. Use `protein.aliases` (not BioMart) to map to ENSG/Entrez — BioMart returns multiple ENSP per gene, causing ambiguity.

**`cooccurence` is misspelled** (one 'r') in every STRING flat file. Column-name-based code must use this exact spelling.

**ENSP IDs may be stale.** STRING carries deprecated Ensembl accessions that no longer exist in current Ensembl releases. Always use the STRING alias file as the canonical mapping source, not current Ensembl.

**One protein per gene locus.** STRING picks one representative ENSP per gene. External sources returning multiple ENSP per gene will have missing rows when joined to STRING.

**`homology` is a penalty, not evidence.** A high homology score means the two proteins are paralogs; STRING uses it to down-weight edges, not as positive interaction evidence.

**`combined_score` is confidence, not binding affinity.** A text-mined functional association can outscore a directly measured physical interaction.

**Physical links are ~10% of the full network.** Use `protein.physical.links` if modeling physical binding only (experimental + curated database channels); the full network includes co-expression, textmining, and co-occurrence.

**Interolog inflation.** Transferred channels (`_transferred` columns) contribute significantly to `combined_score` even for human. Filter on `experiments` directly when you need only species-native experimental evidence.
