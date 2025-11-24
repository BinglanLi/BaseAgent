"""
Environment Description with Pydantic Models

This module provides structured data lake and library definitions using
Pydantic models for better organization, validation, and tracking.

Usage:
    from BaseAgent.env_desc_v2 import data_lake_items, libraries
"""

from BaseAgent.resources import DataLakeItem, Library

# ==============================================================================
# Data Lake Items (using Pydantic models)
# ==============================================================================

data_lake_items: list[DataLakeItem] = [
    # # Protein-Protein Interactions
    # DataLakeItem(
    #     filename="affinity_capture-ms.parquet",
    #     description="Protein-protein interactions detected via affinity capture and mass spectrometry.",
    #     format="parquet",
    #     category="protein_interaction"
    # ),
    # DataLakeItem(
    #     filename="affinity_capture-rna.parquet",
    #     description="Protein-RNA interactions detected by affinity capture.",
    #     format="parquet",
    #     category="protein_rna_interaction"
    # ),
    # DataLakeItem(
    #     filename="co-fractionation.parquet",
    #     description="Protein-protein interactions from co-fractionation experiments.",
    #     format="parquet",
    #     category="protein_interaction"
    # ),
    
    # # Drug Discovery & Binding
    # DataLakeItem(
    #     filename="BindingDB_All_202409.tsv",
    #     description="Measured binding affinities between proteins and small molecules for drug discovery.",
    #     format="tsv",
    #     category="drug_discovery",
    #     size_mb=450.5
    # ),
    # DataLakeItem(
    #     filename="broad_repurposing_hub_molecule_with_smiles.parquet",
    #     description="Molecules from Broad Institute's Drug Repurposing Hub with SMILES annotations.",
    #     format="parquet",
    #     category="drug_discovery"
    # ),
    # DataLakeItem(
    #     filename="broad_repurposing_hub_phase_moa_target_info.parquet",
    #     description="Drug phases, mechanisms of action, and target information from Broad Institute.",
    #     format="parquet",
    #     category="drug_discovery"
    # ),
    
    # # Drug-Drug Interactions (DDInter Database)
    # DataLakeItem(
    #     filename="ddinter_alimentary_tract_metabolism.csv",
    #     description="Drug-drug interactions for alimentary tract and metabolism drugs from DDInter 2.0 database.",
    #     format="csv",
    #     category="drug_interaction"
    # ),
    # DataLakeItem(
    #     filename="ddinter_antineoplastic.csv",
    #     description="Drug-drug interactions for antineoplastic and immunomodulating agents from DDInter 2.0 database.",
    #     format="csv",
    #     category="drug_interaction"
    # ),
    # DataLakeItem(
    #     filename="ddinter_antiparasitic.csv",
    #     description="Drug-drug interactions for antiparasitic products from DDInter 2.0 database.",
    #     format="csv",
    #     category="drug_interaction"
    # ),
    # DataLakeItem(
    #     filename="ddinter_blood_organs.csv",
    #     description="Drug-drug interactions for blood and blood forming organs drugs from DDInter 2.0 database.",
    #     format="csv",
    #     category="drug_interaction"
    # ),
    # DataLakeItem(
    #     filename="ddinter_dermatological.csv",
    #     description="Drug-drug interactions for dermatological drugs from DDInter 2.0 database.",
    #     format="csv",
    #     category="drug_interaction"
    # ),
    
    # # Cancer Cell Lines (DepMap)
    # DataLakeItem(
    #     filename="DepMap_CRISPRGeneDependency.csv",
    #     description="Gene dependency probability estimates for cancer cell lines, including all DepMap models.",
    #     format="csv",
    #     category="cancer_genomics",
    #     size_mb=850.0
    # ),
    # DataLakeItem(
    #     filename="DepMap_CRISPRGeneEffect.csv",
    #     description="Genome-wide CRISPR gene effect estimates for cancer cell lines, including all DepMap models.",
    #     format="csv",
    #     category="cancer_genomics",
    #     size_mb=920.0
    # ),
    # DataLakeItem(
    #     filename="DepMap_Model.csv",
    #     description="Metadata describing all cancer models/cell lines which are referenced by a dataset contained within the DepMap portal.",
    #     format="csv",
    #     category="cancer_genomics"
    # ),
    # DataLakeItem(
    #     filename="DepMap_OmicsExpressionProteinCodingGenesTPMLogp1.csv",
    #     description="Gene expression in TPMs for cancer cell lines, including all DepMap models.",
    #     format="csv",
    #     category="cancer_genomics",
    #     size_mb=1200.0
    # ),
    
    # # Single Cell Data
    # DataLakeItem(
    #     filename="czi_census_datasets_v4.parquet",
    #     description="Datasets from the Chan Zuckerberg Initiative's Cell Census.",
    #     format="parquet",
    #     category="single_cell"
    # ),
    
    # # Other datasets
    # DataLakeItem(
    #     filename="txgnn_prediction.pkl",
    #     description="Prediction data for TXGNN (Treatment-Gene Graph Neural Network).",
    #     format="pkl",
    #     category="ml_predictions"
    # ),
]


# ==============================================================================
# Software Libraries (using Pydantic models)
# ==============================================================================

libraries: list[Library] = [
    # # === PYTHON PACKAGES ===
    
    # # Core Bioinformatics
    # Library(
    #     name="biopython",
    #     description="A set of tools for biological computation including parsers for bioinformatics files, access to online services, and interfaces to common bioinformatics programs.",
    #     type="Python",
    #     category="bioinformatics",
    #     version="1.79",
    #     installation_cmd="pip install biopython"
    # ),
    # Library(
    #     name="biom-format",
    #     description="The Biological Observation Matrix (BIOM) format is designed for representing biological sample by observation contingency tables with associated metadata.",
    #     type="Python",
    #     category="bioinformatics"
    # ),
    # Library(
    #     name="scanpy",
    #     description="A scalable toolkit for analyzing single-cell gene expression data, specifically designed for large datasets using AnnData.",
    #     type="Python",
    #     category="single_cell",
    #     version="1.9.0",
    #     installation_cmd="pip install scanpy"
    # ),
    # Library(
    #     name="scikit-bio",
    #     description="Data structures, algorithms, and educational resources for bioinformatics, including sequence analysis, phylogenetics, and ordination methods.",
    #     type="Python",
    #     category="bioinformatics"
    # ),
    # Library(
    #     name="anndata",
    #     description="A Python package for handling annotated data matrices in memory and on disk, primarily used for single-cell genomics data.",
    #     type="Python",
    #     category="single_cell",
    #     version="0.9.0"
    # ),
    # Library(
    #     name="mudata",
    #     description="A Python package for multimodal data storage and manipulation, extending AnnData to handle multiple modalities.",
    #     type="Python",
    #     category="single_cell"
    # ),
    # Library(
    #     name="pyliftover",
    #     description="A Python implementation of UCSC liftOver tool for converting genomic coordinates between genome assemblies.",
    #     type="Python",
    #     category="genomics"
    # ),
    # Library(
    #     name="biopandas",
    #     description="A package that provides pandas DataFrames for working with molecular structures and biological data.",
    #     type="Python",
    #     category="bioinformatics"
    # ),
    # Library(
    #     name="biotite",
    #     description="A comprehensive library for computational molecular biology, providing tools for sequence analysis, structure analysis, and more.",
    #     type="Python",
    #     category="bioinformatics"
    # ),
    # Library(
    #     name="lazyslide",
    #     description="A Python framework that brings interoperable, reproducible whole slide image analysis, enabling seamless histopathology workflows from preprocessing to deep learning.",
    #     type="Python",
    #     category="pathology"
    # ),
    
    # # Genomics & Variant Analysis
    # Library(
    #     name="gget",
    #     description="A toolkit for accessing genomic databases and retrieving sequences, annotations, and other genomic data.",
    #     type="Python",
    #     category="genomics",
    #     installation_cmd="pip install gget"
    # ),
    # Library(
    #     name="lifelines",
    #     description="A complete survival analysis library for fitting models, plotting, and statistical tests.",
    #     type="Python",
    #     category="statistics"
    # ),
    # Library(
    #     name="pysam",
    #     description="Python interface for reading/writing SAM/BAM/CRAM files and accessing genomic data.",
    #     type="Python",
    #     category="genomics",
    #     installation_cmd="pip install pysam"
    # ),
    # Library(
    #     name="pyvcf",
    #     description="A Python parser for VCF (Variant Call Format) files used in genomics.",
    #     type="Python",
    #     category="genomics"
    # ),
    
    # # Cheminformatics
    # Library(
    #     name="rdkit",
    #     description="Open-source cheminformatics software for molecular structure manipulation, descriptor calculation, and molecular similarity.",
    #     type="Python",
    #     category="cheminformatics",
    #     installation_cmd="pip install rdkit"
    # ),
    # Library(
    #     name="deeppurpose",
    #     description="A deep learning library for drug-target interaction prediction and drug repurposing.",
    #     type="Python",
    #     category="drug_discovery"
    # ),
    
    # # Machine Learning & Data Science
    # Library(
    #     name="scikit-learn",
    #     description="Machine learning library with classification, regression, clustering, and dimensionality reduction algorithms.",
    #     type="Python",
    #     category="machine_learning",
    #     version="1.3.0",
    #     installation_cmd="pip install scikit-learn"
    # ),
    # Library(
    #     name="pandas",
    #     description="Data manipulation and analysis library providing DataFrames for structured data.",
    #     type="Python",
    #     category="data_science",
    #     version="2.0.0",
    #     installation_cmd="pip install pandas"
    # ),
    # Library(
    #     name="numpy",
    #     description="Fundamental package for scientific computing with multi-dimensional arrays and mathematical functions.",
    #     type="Python",
    #     category="data_science",
    #     version="1.24.0"
    # ),
    # Library(
    #     name="scipy",
    #     description="Scientific computing library with modules for optimization, linear algebra, integration, and statistics.",
    #     type="Python",
    #     category="data_science",
    #     version="1.11.0"
    # ),
    
    # # Visualization
    # Library(
    #     name="matplotlib",
    #     description="Comprehensive plotting library for creating static, animated, and interactive visualizations.",
    #     type="Python",
    #     category="visualization",
    #     version="3.7.0",
    #     installation_cmd="pip install matplotlib"
    # ),
    # Library(
    #     name="seaborn",
    #     description="Statistical data visualization library based on matplotlib with high-level interface.",
    #     type="Python",
    #     category="visualization",
    #     version="0.12.0",
    #     installation_cmd="pip install seaborn"
    # ),
    # Library(
    #     name="plotly",
    #     description="Interactive graphing library for creating web-based visualizations.",
    #     type="Python",
    #     category="visualization",
    #     installation_cmd="pip install plotly"
    # ),
    
    # # Gene Set Enrichment
    # Library(
    #     name="gseapy",
    #     description="Gene Set Enrichment Analysis in Python for pathway analysis and visualization.",
    #     type="Python",
    #     category="pathway_analysis",
    #     installation_cmd="pip install gseapy"
    # ),
    
    # # Single Cell Data Access
    # Library(
    #     name="cellxgene-census",
    #     description="API for accessing single-cell datasets from the CZ CELLxGENE Discover platform.",
    #     type="Python",
    #     category="single_cell",
    #     installation_cmd="pip install cellxgene-census"
    # ),
    
    # # === R PACKAGES ===
    
    # Library(
    #     name="ggplot2",
    #     description="Data visualization package implementing the grammar of graphics.",
    #     type="R",
    #     category="visualization",
    #     installation_cmd="install.packages('ggplot2')"
    # ),
    # Library(
    #     name="dplyr",
    #     description="Grammar of data manipulation with functions for data transformation.",
    #     type="R",
    #     category="data_manipulation",
    #     installation_cmd="install.packages('dplyr')"
    # ),
    # Library(
    #     name="tidyr",
    #     description="Tools for tidying data and reshaping datasets.",
    #     type="R",
    #     category="data_manipulation",
    #     installation_cmd="install.packages('tidyr')"
    # ),
    # Library(
    #     name="DESeq2",
    #     description="Differential gene expression analysis for RNA-seq data.",
    #     type="R",
    #     category="genomics",
    #     installation_cmd="BiocManager::install('DESeq2')"
    # ),
    # Library(
    #     name="Seurat",
    #     description="Comprehensive toolkit for single-cell RNA-seq analysis and visualization.",
    #     type="R",
    #     category="single_cell",
    #     version="5.0.0",
    #     installation_cmd="install.packages('Seurat')"
    # ),
    # Library(
    #     name="clusterProfiler",
    #     description="Statistical analysis and visualization of functional profiles for genes and gene clusters.",
    #     type="R",
    #     category="pathway_analysis",
    #     installation_cmd="BiocManager::install('clusterProfiler')"
    # ),
    # Library(
    #     name="edgeR",
    #     description="Differential expression analysis of digital gene expression data.",
    #     type="R",
    #     category="genomics",
    #     installation_cmd="BiocManager::install('edgeR')"
    # ),
    # Library(
    #     name="limma",
    #     description="Linear models for microarray and RNA-seq data analysis.",
    #     type="R",
    #     category="genomics",
    #     installation_cmd="BiocManager::install('limma')"
    # ),
    
    # # === CLI TOOLS ===
    
    # Library(
    #     name="samtools",
    #     description="Suite of programs for interacting with high-throughput sequencing data (SAM/BAM/CRAM).",
    #     type="CLI",
    #     category="genomics",
    #     installation_cmd="conda install -c bioconda samtools"
    # ),
    # Library(
    #     name="bcftools",
    #     description="Tools for variant calling and manipulating VCF/BCF files.",
    #     type="CLI",
    #     category="genomics",
    #     installation_cmd="conda install -c bioconda bcftools"
    # ),
    # Library(
    #     name="bedtools",
    #     description="Suite of utilities for genomic interval arithmetic and comparison.",
    #     type="CLI",
    #     category="genomics",
    #     installation_cmd="conda install -c bioconda bedtools"
    # ),
    # Library(
    #     name="blast",
    #     description="Basic Local Alignment Search Tool for sequence similarity searching.",
    #     type="CLI",
    #     category="bioinformatics",
    #     installation_cmd="conda install -c bioconda blast"
    # ),
    # Library(
    #     name="bowtie2",
    #     description="Fast and memory-efficient tool for aligning sequencing reads to long reference sequences.",
    #     type="CLI",
    #     category="genomics",
    #     installation_cmd="conda install -c bioconda bowtie2"
    # ),
    # Library(
    #     name="bwa",
    #     description="Burrows-Wheeler Aligner for mapping low-divergent sequences against a reference genome.",
    #     type="CLI",
    #     category="genomics",
    #     installation_cmd="conda install -c bioconda bwa"
    # ),
    # Library(
    #     name="fastqc",
    #     description="Quality control tool for high throughput sequence data.",
    #     type="CLI",
    #     category="genomics",
    #     installation_cmd="conda install -c bioconda fastqc"
    # ),
    # Library(
    #     name="gatk",
    #     description="Genome Analysis Toolkit for variant discovery in high-throughput sequencing data.",
    #     type="CLI",
    #     category="genomics",
    #     installation_cmd="conda install -c bioconda gatk4"
    # ),
]

