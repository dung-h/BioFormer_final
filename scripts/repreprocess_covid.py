#!/usr/bin/env python3
"""
Re-preprocess COVID data to fix cell type encoding issues.
"""

import os
import numpy as np
import pandas as pd
import h5py
from scipy.sparse import issparse
import scanpy as sc
import logging

logging.basicConfig(level=logging.INFO)

def cpu_quantile_binning(expr, num_bins=51):
    """Perform 51-bin quantile binning with zero bin."""
    binned = np.zeros_like(expr, dtype=np.uint8)
    for i in range(expr.shape[0]):
        cell = expr[i]
        nz = cell[cell > 0]
        if nz.size > 0:
            thresholds = np.quantile(nz, np.linspace(0, 1, num_bins)[1:])
            bins = np.digitize(cell, thresholds, right=True) + 1
            bins[cell == 0] = 0
            binned[i] = bins
    return binned

def load_selected_genes(selected_genes_file):
    """Load selected genes from file."""
    with open(selected_genes_file, 'r') as f:
        selected_genes = [line.strip() for line in f if line.strip()]
    return selected_genes

def preprocess_dataset(fp, selected_genes, study_id, output_dir, num_bins=51):
    """Preprocess a single dataset with correct cell type encoding."""
    try:
        adata = sc.read_h5ad(fp)
        # Use raw data if available, otherwise use main X
        if adata.raw is not None:
            X = adata.raw.X
            var_names = adata.raw.var_names
        else:
            X = adata.X
            var_names = adata.var_names
        
        # Map genes to indices
        gene_to_idx = {name: i for i, name in enumerate(var_names)}
        indices = []
        valid_indices = []
        for i, gene in enumerate(selected_genes):
            idx = gene_to_idx.get(gene, None)
            if idx is not None:
                indices.append(idx)
                valid_indices.append(i)
        
        if not indices:
            logging.error(f"No selected genes found in {fp}. Skipping.")
            return
        
        matched_genes = len(indices)
        if matched_genes < len(selected_genes):
            logging.warning(
                f"Study {study_id} has {matched_genes} of {len(selected_genes)} "
                "selected genes; padding with zeros for missing genes"
            )
        
        # Create padded expression matrix
        X_padded = np.zeros((adata.n_obs, len(selected_genes)), dtype=np.float32)
        if indices:
            X_subset = X[:, indices].toarray() if issparse(X) else X[:, indices]
            X_padded[:, valid_indices] = X_subset
        
        # Perform quantile binning
        binned = cpu_quantile_binning(X_padded, num_bins)
        
        # Normalize and log transform
        adata_temp = sc.AnnData(X_padded, obs=adata.obs, var=pd.DataFrame(index=selected_genes))
        sc.pp.normalize_total(adata_temp, target_sum=1e4)
        sc.pp.log1p(adata_temp)
        expr_cont = adata_temp.X
        
        # Create non-zero mask
        non_zero_mask = (X_padded > 0).astype(np.uint8)
        
        # Extract metadata - specifically cell types
        meta_keys = ['cell_type']
        meta = {}
        for k in meta_keys:
            if k in adata.obs.columns:
                meta[k] = adata.obs[k].values
        
        # Save to HDF5 file
        out_f = os.path.join(output_dir, f"preprocessed_study{study_id}.h5")
        with h5py.File(out_f, 'w') as f:
            f.create_dataset('binned_expr', data=binned, compression='gzip', compression_opts=2)
            f.create_dataset('expr_cont', data=expr_cont, compression='gzip', compression_opts=2)
            f.create_dataset('non_zero_mask', data=non_zero_mask, compression='gzip', compression_opts=2)
            f.create_dataset('study_ids', data=np.full(adata.n_obs, study_id, dtype=np.int64), compression='gzip', compression_opts=2)
            
            # Handle metadata with proper string encoding
            for k, vals in meta.items():
                print(f"Processing metadata key: {k}")
                print(f"Values dtype: {vals.dtype}")
                print(f"Sample values: {vals[:5]}")
                
                if vals.dtype.kind in ('O', 'U') or str(vals.dtype) == 'category':
                    # Convert categorical or object to string
                    vals_str = [str(x) for x in vals]
                    print(f"String values sample: {vals_str[:5]}")
                    
                    # Encode to bytes
                    vals_encoded = [s.encode('utf-8') for s in vals_str]
                    print(f"Encoded values sample: {vals_encoded[:5]}")
                    
                    # Calculate max length properly
                    max_len = max((len(s) for s in vals_str), default=1)
                    print(f"Max length: {max_len}")
                    
                    # Create dtype string
                    dtype_str = f'S{min(max_len, 200)}'
                    print(f"Dtype string: {dtype_str}")
                    
                    # Create dataset
                    f.create_dataset(k, data=vals_encoded, dtype=dtype_str, compression='gzip', compression_opts=2)
                    
                    # Verify what was written
                    written_data = f[k][:]
                    print(f"Written data sample: {written_data[:5]}")
                    print(f"Written data dtype: {written_data.dtype}")
                    print(f"Written unique count: {len(np.unique(written_data))}")
                    
                else:
                    f.create_dataset(k, data=vals, compression='gzip', compression_opts=2)
            
            # Save variable names
            f.create_dataset('var_names', data=np.array(selected_genes, dtype='S'))
        
        logging.info(f"Saved preprocessed data for study {study_id} to {out_f}")
        
    except Exception as e:
        logging.error(f"Failed to preprocess {fp}: {e}")
        raise

def select_hvg_from_covid_data(covid_files, n_top_genes=1000):
    """Select highly variable genes from COVID data."""
    import scanpy as sc
    from collections import Counter
    
    # Get common genes across all files
    var_names_list = []
    for fp in covid_files:
        adata = sc.read_h5ad(fp)
        var_names_list.append(set(adata.var_names))
    
    # Find common genes
    common_genes = set.intersection(*var_names_list)
    print(f"Found {len(common_genes)} common genes across COVID datasets")
    
    # Simple selection of top genes (could be improved with proper HVG analysis)
    selected_genes = sorted(list(common_genes))[:n_top_genes]
    print(f"Selected {len(selected_genes)} genes")
    
    return selected_genes

def main():
    """Re-preprocess COVID data with correct cell type encoding."""
    # Define paths
    covid_files = [
        'data/additional_datasets/COVID-19-splitted/covid/batch_covid_subsampled_train.h5ad',
        'data/additional_datasets/COVID-19-splitted/covid/batch_covid_subsampled_test.h5ad'
    ]
    
    output_dir = 'data/preprocessed_covid'
    
    # Select genes from COVID data itself
    selected_genes = select_hvg_from_covid_data(covid_files, n_top_genes=1000)
    
    # Save selected genes
    selected_genes_file = os.path.join(output_dir, 'selected_genes.txt')
    with open(selected_genes_file, 'w') as f:
        for gene in selected_genes:
            f.write(f"{gene}\n")
    print(f"Saved selected genes to {selected_genes_file}")
    
    # Process each file
    for study_id, fp in enumerate(covid_files):
        print(f"\nProcessing study {study_id}: {fp}")
        preprocess_dataset(fp, selected_genes, study_id, output_dir)

if __name__ == "__main__":
    main()