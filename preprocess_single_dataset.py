#!/usr/bin/env python3
"""
Preprocess a single dataset using existing selected_genes.txt
"""
import os
import sys
import numpy as np
import pandas as pd
import h5py
from scipy.sparse import issparse
import scanpy as sc
import logging

# Add the parent directory to sys.path to import from preprocess.py
sys.path.append('/home/tripham/scgpt')
from preprocess import preprocess_dataset

logging.basicConfig(level=logging.INFO)

def preprocess_single_dataset(dataset_path, study_id, selected_genes_file, output_dir):
    """Preprocess a single dataset with existing selected genes."""
    
    # Load selected genes
    if not os.path.exists(selected_genes_file):
        logging.error(f"Selected genes file not found: {selected_genes_file}")
        return False
    
    with open(selected_genes_file, 'r') as f:
        selected_genes = [line.strip() for line in f if line.strip()]
    
    logging.info(f"Loaded {len(selected_genes)} selected genes from {selected_genes_file}")
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        logging.error(f"Dataset not found: {dataset_path}")
        return False
    
    # Test gene compatibility first
    try:
        logging.info(f"Testing gene compatibility for {dataset_path}")
        adata = sc.read_h5ad(dataset_path, backed='r')
        
        # Get gene names
        if adata.raw is not None:
            var_names = adata.raw.var_names
        else:
            var_names = adata.var_names
        
        # Check overlap
        gene_set = set(var_names)
        overlap = len(set(selected_genes) & gene_set)
        total_genes = len(selected_genes)
        
        logging.info(f"Gene compatibility: {overlap}/{total_genes} genes found ({overlap/total_genes*100:.1f}%)")
        
        if overlap < total_genes * 0.5:  # Less than 50% overlap
            logging.warning(f"Low gene overlap ({overlap/total_genes*100:.1f}%). Proceeding with padding.")
        
        adata.file.close()
        
    except Exception as e:
        logging.error(f"Error testing gene compatibility: {e}")
        return False
    
    # Preprocess the dataset
    try:
        logging.info(f"Starting preprocessing of {dataset_path}")
        preprocess_dataset(dataset_path, selected_genes, study_id, output_dir)
        logging.info(f"Successfully preprocessed {dataset_path}")
        return True
        
    except Exception as e:
        logging.error(f"Error preprocessing {dataset_path}: {e}")
        return False

def main():
    # Parameters - can be overridden via command line
    import sys
    if len(sys.argv) >= 3:
        dataset_path = sys.argv[1]
        study_id = int(sys.argv[2])
    else:
        dataset_path = "/mnt/nasdev2/dung/data/data24.h5ad"
        study_id = 23  # data24 is study_id 23 (0-indexed)
    selected_genes_file = "/mnt/nasdev2/dung/preprocessed_5000/selected_genes.txt"
    output_dir = "/mnt/nasdev2/dung/preprocessed_5000"
    
    logging.info("="*60)
    logging.info("PREPROCESSING REPLACEMENT DATASET")
    logging.info("="*60)
    logging.info(f"Dataset: {dataset_path}")
    logging.info(f"Study ID: {study_id}")
    logging.info(f"Selected genes file: {selected_genes_file}")
    logging.info(f"Output directory: {output_dir}")
    
    success = preprocess_single_dataset(dataset_path, study_id, selected_genes_file, output_dir)
    
    if success:
        logging.info("✅ Preprocessing completed successfully!")
        logging.info(f"Output file: {output_dir}/preprocessed_study{study_id}.h5")
    else:
        logging.error("❌ Preprocessing failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())