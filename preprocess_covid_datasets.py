#!/usr/bin/env python3
"""
Preprocess COVID-19 datasets for batch integration testing with correct gene vocabulary.
"""
import os
import sys
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def quantile_binning(data, n_bins=51):
    """Apply quantile binning to gene expression data."""
    quantiles = np.linspace(0, 1, n_bins)
    return np.searchsorted(np.quantile(data, quantiles), data, side='right') - 1

def load_gene_vocabulary(genes_file):
    """Load gene vocabulary from file."""
    with open(genes_file, 'r') as f:
        genes = [line.strip() for line in f.readlines()]
    return genes

def load_gene_mapping(mapping_file):
    """Load gene mapping from ENSEMBL IDs to gene symbols."""
    df = pd.read_csv(mapping_file)
    return dict(zip(df['ensembl_id'], df['gene_symbol']))

def preprocess_covid_data():
    """Preprocess COVID-19 datasets for batch integration."""
    
    # Paths
    test_path = "/home/tripham/scgpt/trial_3_based_moe/data/additional_datasets/COVID-19-splitted/covid/batch_covid_subsampled_test.h5ad"
    train_path = "/home/tripham/scgpt/trial_3_based_moe/data/additional_datasets/COVID-19-splitted/covid/batch_covid_subsampled_train.h5ad"
    genes_file = "/mnt/nasdev2/dung/preprocessed/selected_genes.txt"
    mapping_file = "/home/tripham/scgpt/trial_3_based_moe/gene_mapping.csv"
    output_dir = "/home/tripham/scgpt/trial_3_based_moe/preprocessed_covid"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load gene vocabulary and mapping
    target_genes = load_gene_vocabulary(genes_file)
    gene_mapping = load_gene_mapping(mapping_file)
    print(f"Target vocabulary: {len(target_genes)} genes")
    print(f"Gene mapping: {len(gene_mapping)} mappings")
    
    # Process both train and test datasets
    for split, file_path in [("train", train_path), ("test", test_path)]:
        print(f"\nProcessing COVID-19 {split} dataset...")
        
        # Load dataset
        adata = sc.read_h5ad(file_path)
        print(f"Original shape: {adata.shape}")
        print(f"Original genes: {adata.n_vars}")
        print(f"Studies: {adata.obs['batch'].unique()}")
        print(f"Cell types: {adata.obs['cell_type'].unique()}")
        
        # Map ENSEMBL IDs to gene symbols for comparison
        target_symbols = [gene_mapping.get(ensembl_id, ensembl_id) for ensembl_id in target_genes]
        
        # Filter genes to match target vocabulary
        gene_overlap = [gene for gene in target_symbols if gene in adata.var_names]
        print(f"Gene overlap: {len(gene_overlap)}/{len(target_genes)} genes")
        
        if len(gene_overlap) < 100:
            print(f"Warning: Low gene overlap ({len(gene_overlap)} genes)")
            continue
            
        # Filter to overlapping genes
        adata_filtered = adata[:, gene_overlap].copy()
        
        # Reorder genes to match target vocabulary order (using symbols)
        gene_order = [gene for gene in target_symbols if gene in gene_overlap]
        adata_filtered = adata_filtered[:, gene_order]
        
        # Map back to ENSEMBL IDs for gene names
        symbol_to_ensembl = {v: k for k, v in gene_mapping.items()}
        ensembl_order = [symbol_to_ensembl.get(symbol, symbol) for symbol in gene_order]
        
        print(f"Filtered shape: {adata_filtered.shape}")
        
        # Apply quantile binning
        X_dense = adata_filtered.X
        if hasattr(X_dense, 'toarray'):
            X_dense = X_dense.toarray()
        
        X_binned = np.zeros_like(X_dense)
        for i in range(adata_filtered.n_vars):
            gene_expr = X_dense[:, i]
            X_binned[:, i] = quantile_binning(gene_expr, n_bins=51)
        
        # Create non-zero mask
        non_zero_mask = (adata_filtered.X > 0).astype(np.float32)
        if hasattr(non_zero_mask, 'toarray'):
            non_zero_mask = non_zero_mask.toarray()
        
        # Encode categorical variables
        batch_encoder = pd.Categorical(adata_filtered.obs['batch'])
        cell_type_encoder = pd.Categorical(adata_filtered.obs['cell_type'])
        
        # Save preprocessed data
        output_file = os.path.join(output_dir, f"preprocessed_covid_{split}.h5")
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('binned_expr', data=X_binned.astype(np.int32))
            f.create_dataset('non_zero_mask', data=non_zero_mask)
            f.create_dataset('cell_type', data=[ct.encode('utf-8') for ct in adata_filtered.obs['cell_type']])
            f.create_dataset('study_ids', data=batch_encoder.codes.astype(np.int32))
            f.create_dataset('gene_names', data=[gene.encode('utf-8') for gene in ensembl_order])
        
        print(f"Saved preprocessed data to: {output_file}")
        print(f"Binned expression shape: {X_binned.shape}")
        print(f"Unique studies: {len(batch_encoder.categories)}")
        print(f"Unique cell types: {len(cell_type_encoder.categories)}")

if __name__ == "__main__":
    preprocess_covid_data()
    print("\nCOVID-19 preprocessing completed!")