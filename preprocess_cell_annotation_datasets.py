#!/usr/bin/env python3
"""
Preprocess cell type annotation datasets to match the 1,000-gene vocabulary
used by both FFN and MoE BioFormer models.
"""
import os
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_model_vocabulary(vocab_file="/mnt/nasdev2/dung/preprocessed/selected_genes.txt"):
    """Load the 1,000-gene vocabulary used by both models"""
    with open(vocab_file, 'r') as f:
        genes = [line.strip() for line in f.readlines()]
    return genes

def load_gene_mapping(mapping_file="/home/tripham/scgpt/trial_3_based_moe/gene_mapping.csv"):
    """Load gene mapping from ENSEMBL IDs to gene symbols"""
    df = pd.read_csv(mapping_file)
    # Create mapping from gene symbol to ENSEMBL ID
    symbol_to_ensembl = {}
    ensembl_to_symbol = {}
    
    for _, row in df.iterrows():
        if row['gene_symbol']:  # Only add if gene symbol exists
            symbol_to_ensembl[row['gene_symbol']] = row['ensembl_id']
            ensembl_to_symbol[row['ensembl_id']] = row['gene_symbol']
    
    return symbol_to_ensembl, ensembl_to_symbol

def preprocess_dataset(input_path, output_path, model_genes, cell_type_col, symbol_to_ensembl):
    """
    Preprocess a single dataset to match model vocabulary
    
    Args:
        input_path: Path to input h5ad file
        output_path: Path to save filtered h5ad file
        model_genes: List of genes in model vocabulary
        cell_type_col: Column name containing cell type labels
        symbol_to_ensembl: Dictionary mapping gene symbols to ENSEMBL IDs
    """
    print(f"Processing {input_path}...")
    
    # Load dataset
    adata = sc.read_h5ad(input_path)
    print(f"Original shape: {adata.shape}")
    print(f"Original genes: {adata.n_vars}")
    
    # Get gene symbols from dataset
    gene_symbols = adata.var_names.values
    
    # Find overlap with model vocabulary using gene symbol mapping
    model_genes_set = set(model_genes)
    gene_overlap = []
    gene_indices = []
    
    for i, gene_symbol in enumerate(gene_symbols):
        # Check if gene symbol can be mapped to ENSEMBL ID in model vocabulary
        if gene_symbol in symbol_to_ensembl:
            ensembl_id = symbol_to_ensembl[gene_symbol]
            if ensembl_id in model_genes_set:
                gene_overlap.append(ensembl_id)  # Use ENSEMBL ID
                gene_indices.append(i)
    
    print(f"Gene overlap: {len(gene_overlap)}/{len(model_genes)} ({len(gene_overlap)/len(model_genes)*100:.1f}%)")
    
    if len(gene_overlap) == 0:
        raise ValueError("No genes found in model vocabulary!")
    
    # Filter dataset to only include overlapping genes
    adata_filtered = adata[:, gene_indices].copy()
    
    # Update gene names to match model vocabulary order
    adata_filtered.var_names = gene_overlap
    
    # Ensure cell type column exists
    if cell_type_col not in adata_filtered.obs.columns:
        raise ValueError(f"Cell type column '{cell_type_col}' not found in dataset")
    
    # Remove cells with missing cell types
    valid_cells = ~adata_filtered.obs[cell_type_col].isna()
    adata_filtered = adata_filtered[valid_cells, :]
    
    print(f"Filtered shape: {adata_filtered.shape}")
    print(f"Cell types: {adata_filtered.obs[cell_type_col].nunique()}")
    
    # Save filtered dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata_filtered.write(output_path)
    print(f"Saved to: {output_path}")
    
    return adata_filtered

def main():
    """Main preprocessing function"""
    print("=== Preprocessing Cell Type Annotation Datasets ===")
    
    # Load model vocabulary
    model_genes = load_model_vocabulary()
    print(f"Loaded {len(model_genes)} genes from model vocabulary")
    
    # Load gene mapping
    symbol_to_ensembl, ensembl_to_symbol = load_gene_mapping()
    print(f"Loaded gene mapping for {len(symbol_to_ensembl)} gene symbols")
    
    # Define datasets to process
    datasets = [
        {
            'name': 'MS',
            'input': '/home/tripham/scgpt/trial_3_based_moe/data/external_datasets/MS/c_data.h5ad',
            'output': '/home/tripham/scgpt/trial_3_based_moe/data/external_datasets/MS/filtered_ms_adata.h5ad',
            'cell_type_col': 'celltype'
        },
        {
            'name': 'Myeloid_Reference',
            'input': '/home/tripham/scgpt/trial_3_based_moe/data/external_datasets/Myeloid/reference_adata.h5ad',
            'output': '/home/tripham/scgpt/trial_3_based_moe/data/external_datasets/Myeloid/filtered_reference_adata.h5ad',
            'cell_type_col': 'cell_type'
        },
        {
            'name': 'Myeloid_Query',
            'input': '/home/tripham/scgpt/trial_3_based_moe/data/external_datasets/Myeloid/query_adata.h5ad',
            'output': '/home/tripham/scgpt/trial_3_based_moe/data/external_datasets/Myeloid/filtered_query_adata.h5ad',
            'cell_type_col': 'cell_type'
        },
        {
            'name': 'hPancreas_Train',
            'input': '/home/tripham/scgpt/trial_3_based_moe/data/external_datasets/hPancreas/demo_train.h5ad',
            'output': '/home/tripham/scgpt/trial_3_based_moe/data/external_datasets/hPancreas/filtered_demo_train.h5ad',
            'cell_type_col': 'Celltype'
        },
        {
            'name': 'hPancreas_Test',
            'input': '/home/tripham/scgpt/trial_3_based_moe/data/external_datasets/hPancreas/demo_test.h5ad',
            'output': '/home/tripham/scgpt/trial_3_based_moe/data/external_datasets/hPancreas/filtered_demo_test.h5ad',
            'cell_type_col': 'Celltype'
        }
    ]
    
    # Process each dataset
    for dataset in datasets:
        try:
            print(f"\n--- Processing {dataset['name']} ---")
            preprocess_dataset(
                input_path=Path(dataset['input']),
                output_path=Path(dataset['output']),
                model_genes=model_genes,
                cell_type_col=dataset['cell_type_col'],
                symbol_to_ensembl=symbol_to_ensembl
            )
        except Exception as e:
            print(f"Error processing {dataset['name']}: {e}")
            continue
    
    print("\n=== Preprocessing Complete ===")

if __name__ == "__main__":
    main()