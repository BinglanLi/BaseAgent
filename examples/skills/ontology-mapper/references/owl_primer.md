# OWL Primer: Biomedical Knowledge Graph Classes and Properties

Reference document for the `ontology-mapper` skill.

## Node classes

| Class | Description |
|---|---|
| `Gene` | A genomic locus encoding a functional product |
| `Protein` | A translated gene product |
| `Disease` | A pathological condition with a defined clinical presentation |
| `Drug` | A chemical compound used for therapeutic purposes |
| `Pathway` | A defined sequence of molecular interactions |
| `Phenotype` | An observable characteristic of an organism |

## Data properties

| Property | Domain | Range | Description |
|---|---|---|---|
| `skos:definition` | Any | `xsd:string` | Human-readable definition |
| `skos:prefLabel` | Any | `xsd:string` | Preferred display label |
| `schema:identifier` | Any | `xsd:string` | External database identifier |
| `faldo:location` | Gene | `xsd:string` | Genomic coordinates |
| `schema:molecularFormula` | Drug | `xsd:string` | Chemical formula |

## Object properties (relationships)

| Property | Domain | Range | Description |
|---|---|---|---|
| `ro:causes` | Disease | Phenotype | Disease causes a phenotype |
| `ro:participates_in` | Protein | Pathway | Protein participates in a pathway |
| `ro:has_target` | Drug | Protein | Drug targets a protein |
| `ro:associated_with` | Gene | Disease | Gene is associated with disease |
