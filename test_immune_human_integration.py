#!/usr/bin/env python3
"""
Test MoE BioFormer Integration with Immune Human dataset from Luecken et al. 2022
Dataset: "Benchmarking atlas-level data integration in single-cell genomics"
Reference: https://www.nature.com/articles/s41592-021-01336-8
"""
import os
import sys
import numpy as np
import torch
import scanpy as sc
import anndata
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
import argparse
import warnings
warnings.filterwarnings('ignore')

# Add paths for MoE model
sys.path.append('/home/tripham/scgpt/trial_3_based_moe')
sys.path.append('/home/tripham/scgpt/benchmark/BioFormer')

from models.bioformer_with_ffn_moe import BioFormerMoE
from utils.metrics import compute_graph_connectivity

def check_immune_dataset(data_path):
    """Check if Immune Human dataset is available and valid."""
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        return False
    
    # Check file size (should be ~2GB)
    file_size = os.path.getsize(data_path)
    expected_size = 2064748344  # bytes from Figshare
    
    if file_size < expected_size * 0.9:  # Allow 10% tolerance
        print(f"Dataset appears incomplete. Current size: {file_size/1e9:.2f}GB, Expected: {expected_size/1e9:.2f}GB")
        return False
    
    try:
        # Try to read the dataset
        adata = sc.read_h5ad(data_path)
        print(f"✓ Dataset loaded successfully")
        print(f"  Shape: {adata.shape}")
        print(f"  Cells: {adata.n_obs:,}")
        print(f"  Genes: {adata.n_vars:,}")
        return True
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return False

def preprocess_immune_dataset(adata, target_genes=None, max_cells=None):
    """Preprocess Immune Human dataset for MoE testing."""
    print(f"Original dataset shape: {adata.shape}")
    
    # Print available metadata
    print(f"Available obs columns: {list(adata.obs.columns)}")
    print(f"Available var columns: {list(adata.var.columns)}")
    
    # Check batch information
    if 'batch' in adata.obs.columns:
        print(f"Number of batches: {adata.obs['batch'].nunique()}")
        print(f"Batch distribution:\n{adata.obs['batch'].value_counts()}")
    
    # Check cell type information
    if 'final_annotation' in adata.obs.columns:
        print(f"Number of cell types: {adata.obs['final_annotation'].nunique()}")
        print(f"Top 10 cell types:\n{adata.obs['final_annotation'].value_counts().head(10)}")
    elif 'cell_type' in adata.obs.columns:
        print(f"Number of cell types: {adata.obs['cell_type'].nunique()}")
        print(f"Top 10 cell types:\n{adata.obs['cell_type'].value_counts().head(10)}")
    
    # Subsample if requested
    if max_cells and adata.n_obs > max_cells:
        print(f"Subsampling to {max_cells:,} cells...")
        sc.pp.subsample(adata, n_obs=max_cells, random_state=42)
    
    # Filter genes if target genes provided
    if target_genes:
        print(f"Filtering to {len(target_genes)} target genes...")
        
        # Check if target genes are Ensembl IDs and dataset uses gene symbols
        if target_genes[0].startswith('ENSG') and not adata.var_names[0].startswith('ENSG'):
            print("⚠ Target genes are Ensembl IDs but dataset uses gene symbols")
            print("Using top highly variable genes instead of exact matching")
            # Skip gene filtering and use highly variable genes later
            gene_mask = None
        else:
            # Match genes by symbol or ensembl ID
            if 'gene_symbols' in adata.var.columns:
                gene_mask = adata.var['gene_symbols'].isin(target_genes)
            else:
                gene_mask = adata.var_names.isin(target_genes)
            
            adata = adata[:, gene_mask]
            print(f"After gene filtering: {adata.shape}")
    
    # Basic preprocessing
    print("Performing basic preprocessing...")
    
    # Make variable names unique
    adata.var_names_make_unique()
    
    # Filter cells and genes
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    # Store raw counts
    adata.raw = adata
    
    # Normalize and log transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Find highly variable genes
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    
    print(f"Final preprocessed shape: {adata.shape}")
    return adata

def create_dataset_for_moe(adata, vocab_size=1000):
    """Convert AnnData to format compatible with MoE model."""
    print("Converting dataset for MoE model...")
    
    # Select top variable genes up to vocab_size
    if adata.n_vars > vocab_size:
        # Use highly variable genes if available
        if 'highly_variable' in adata.var.columns:
            hv_genes = adata.var['highly_variable']
            if hv_genes.sum() >= vocab_size:
                top_genes = adata.var[hv_genes].head(vocab_size).index
            else:
                # Fall back to variance-based selection
                var_genes = adata.var.loc[adata.var['highly_variable']].index
                remaining = vocab_size - len(var_genes)
                if remaining > 0:
                    other_genes = adata.var.loc[~adata.var['highly_variable']].head(remaining).index
                    top_genes = list(var_genes) + list(other_genes)
                else:
                    top_genes = var_genes
        else:
            # Select by variance
            X_dense = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
            gene_var = np.var(X_dense, axis=0)
            top_gene_idx = np.argsort(gene_var)[-vocab_size:]
            top_genes = adata.var_names[top_gene_idx]
        
        adata_subset = adata[:, top_genes]
        print(f"Selected {len(top_genes)} genes for MoE model")
    else:
        adata_subset = adata.copy()
        print(f"Using all {adata.n_vars} genes for MoE model")
    
    # Convert to discrete bins (as expected by MoE model)
    X = adata_subset.X.toarray() if hasattr(adata_subset.X, 'toarray') else adata_subset.X
    
    # Bin expression values (simple quantile-based binning)
    n_bins = 50  # Adjust as needed
    binned_X = np.zeros_like(X, dtype=int)
    
    for i in range(X.shape[1]):
        gene_expr = X[:, i]
        # Use quantile-based binning
        bins = np.quantile(gene_expr[gene_expr > 0], np.linspace(0, 1, n_bins))
        bins = np.unique(bins)  # Remove duplicates
        binned_X[:, i] = np.digitize(gene_expr, bins)
    
    # Create cell type encoding
    if 'final_annotation' in adata_subset.obs.columns:
        le_celltype = LabelEncoder()
        cell_types_encoded = le_celltype.fit_transform(adata_subset.obs['final_annotation'])
    elif 'cell_type' in adata_subset.obs.columns:
        le_celltype = LabelEncoder()
        cell_types_encoded = le_celltype.fit_transform(adata_subset.obs['cell_type'])
    else:
        cell_types_encoded = np.zeros(adata_subset.n_obs)
        le_celltype = None
    
    # Create batch encoding
    if 'batch' in adata_subset.obs.columns:
        le_batch = LabelEncoder()
        batches_encoded = le_batch.fit_transform(adata_subset.obs['batch'])
    else:
        batches_encoded = np.zeros(adata_subset.n_obs)
        le_batch = None
    
    # Create non-zero mask
    non_zero_mask = (X > 0).astype(float)
    
    return {
        'binned_expr': binned_X,
        'cell_types': cell_types_encoded,
        'batches': batches_encoded,
        'non_zero_mask': non_zero_mask,
        'adata': adata_subset,
        'le_celltype': le_celltype,
        'le_batch': le_batch,
        'gene_names': adata_subset.var_names.tolist()
    }

def test_moe_with_immune_data(data_dict, checkpoint_path, device="cuda", batch_size=64):
    """Test MoE model with Immune Human dataset."""
    print("Testing MoE model with Immune Human dataset...")
    
    # Prepare data
    binned_expr = torch.tensor(data_dict['binned_expr'], dtype=torch.long)
    cell_types = torch.tensor(data_dict['cell_types'], dtype=torch.long)
    batches = torch.tensor(data_dict['batches'], dtype=torch.long)
    non_zero_mask = torch.tensor(data_dict['non_zero_mask'], dtype=torch.float32)
    
    # Load MoE model
    model = BioFormerMoE(
        vocab_size=1000,
        num_cell_types=max(200, len(np.unique(data_dict['cell_types']))),
        num_studies=max(15, len(np.unique(data_dict['batches']))),
        d_model=256,
        nhead=4,
        num_layers=8,
        num_experts=4
    ).to(device)
    
    # Load checkpoint if available
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        
        # Load compatible parameters
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in state_dict and state_dict[k].shape == v.shape:
                    v.copy_(state_dict[k])
                else:
                    print(f"Skipping {k} due to shape mismatch or missing key")
        print("✓ Checkpoint loaded")
    else:
        print("⚠ No checkpoint found, using random initialization")
    
    model.eval()
    
    # Extract embeddings
    embeddings = []
    n_samples = binned_expr.shape[0]
    
    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc="Extracting embeddings"):
            end_idx = min(i + batch_size, n_samples)
            
            batch_binned = binned_expr[i:end_idx].to(device)
            batch_ct = cell_types[i:end_idx].to(device)
            batch_batches = batches[i:end_idx].to(device)
            batch_mask = non_zero_mask[i:end_idx].to(device)
            
            # Forward pass
            out = model(
                binned_expr=batch_binned,
                cell_type=batch_ct,
                study_id=batch_batches,
                non_zero_mask=batch_mask
            )
            
            # Extract CLS embeddings
            cls_embeddings = out[2][:, 0, :]  # Hidden states, first token
            embeddings.append(cls_embeddings.cpu().numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)
    print(f"Extracted embeddings shape: {embeddings.shape}")
    
    return embeddings

def evaluate_integration(embeddings, cell_types, batches, cell_type_encoder=None):
    """Evaluate integration quality."""
    print("Evaluating integration quality...")
    
    # Create AnnData for analysis
    adata = anndata.AnnData(X=embeddings)
    adata.obs['cell_type'] = cell_types
    adata.obs['batch'] = batches
    
    # Compute neighbors and clustering
    sc.pp.neighbors(adata, use_rep="X", n_neighbors=15, metric="cosine")
    sc.tl.leiden(adata, key_added="leiden")
    
    # Encode cell types for metrics
    if cell_type_encoder:
        unique_cts = cell_type_encoder.classes_
        ct_encoded = cell_type_encoder.transform(cell_types)
    else:
        le = LabelEncoder()
        ct_encoded = le.fit_transform(cell_types)
        unique_cts = le.classes_
    
    # Compute metrics
    nmi = normalized_mutual_info_score(ct_encoded, adata.obs["leiden"])
    ari = adjusted_rand_score(ct_encoded, adata.obs["leiden"])
    sil = (silhouette_score(embeddings, ct_encoded, metric="cosine") + 1) / 2
    
    # Graph connectivity (batch mixing)
    try:
        gcon = compute_graph_connectivity(adata, cell_types, batches, n_neighbors=50)
    except:
        gcon = 0.0  # Fallback if computation fails
    
    results = {
        'nmi': nmi,
        'ari': ari,
        'silhouette': sil,
        'graph_connectivity': gcon,
        'avg_bio': (nmi + ari + sil) / 3,
        'avg_batch': gcon,
        'adata': adata
    }
    
    print(f"\n{'='*50}")
    print(f"MoE BioFormer Results on Immune Human Dataset:")
    print(f"{'='*50}")
    print(f"  NMI (cell type coherence): {nmi:.4f}")
    print(f"  ARI (cell type coherence): {ari:.4f}")
    print(f"  Silhouette score:          {sil:.4f}")
    print(f"  Graph connectivity:        {gcon:.4f}")
    print(f"  AvgBIO (biological):       {(nmi + ari + sil)/3:.4f}")
    print(f"  AvgBATCH (integration):    {gcon:.4f}")
    print(f"{'='*50}")
    
    return results

def create_visualizations(adata, save_prefix="immune_human_integration"):
    """Create UMAP visualizations."""
    print("Creating UMAP visualizations...")
    
    # Compute UMAP
    sc.tl.umap(adata)
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot by cell type
    unique_cts = adata.obs['cell_type'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, min(len(unique_cts), 20)))
    
    for i, ct in enumerate(unique_cts[:20]):  # Limit to first 20 for visibility
        mask = adata.obs['cell_type'] == ct
        axes[0].scatter(adata.obsm['X_umap'][mask, 0], 
                       adata.obsm['X_umap'][mask, 1],
                       c=[colors[i]], label=ct, s=1, alpha=0.7)
    
    axes[0].set_title('Cell Types')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Plot by batch
    unique_batches = adata.obs['batch'].unique()
    batch_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_batches)))
    
    for i, batch in enumerate(unique_batches):
        mask = adata.obs['batch'] == batch
        axes[1].scatter(adata.obsm['X_umap'][mask, 0],
                       adata.obsm['X_umap'][mask, 1],
                       c=[batch_colors[i]], label=f'Batch {batch}', s=1, alpha=0.7)
    
    axes[1].set_title('Batches')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{save_prefix}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"UMAP plots saved to: {save_prefix}.png")

def main():
    parser = argparse.ArgumentParser(description="Test MoE BioFormer with Immune Human dataset")
    parser.add_argument("--data_path", 
                       default="/home/tripham/scgpt/trial_3_based_moe/data/Immune_ALL_human.h5ad",
                       help="Path to Immune Human dataset")
    parser.add_argument("--checkpoint", 
                       default="/mnt/nasdev2/dung/training1/checkpoints/checkpoint_epoch_1_20250618_140100_512d_8head_12layer.pt",
                       help="MoE model checkpoint path")
    parser.add_argument("--genes_file", 
                       default="/mnt/nasdev2/dung/preprocessed/selected_genes.txt",
                       help="Gene vocabulary file")
    parser.add_argument("--max_cells", type=int, default=50000,
                       help="Maximum number of cells to use (for memory constraints)")
    parser.add_argument("--vocab_size", type=int, default=1000,
                       help="Vocabulary size for gene selection")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    
    args = parser.parse_args()
    
    print(f"Immune Human Dataset Integration Test")
    print(f"Dataset: {args.data_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Genes file: {args.genes_file}")
    print(f"Max cells: {args.max_cells:,}")
    
    # Check if dataset is ready
    if not check_immune_dataset(args.data_path):
        print("❌ Dataset not ready. Please wait for download to complete.")
        return
    
    # Load gene vocabulary
    target_genes = None
    if os.path.exists(args.genes_file):
        print(f"Loading gene vocabulary from {args.genes_file}")
        with open(args.genes_file, 'r') as f:
            target_genes = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(target_genes)} target genes")
    else:
        print(f"⚠ Gene vocabulary file not found: {args.genes_file}")
        print("Will use top variable genes instead")
    
    # Load and preprocess dataset
    print("Loading dataset...")
    adata = sc.read_h5ad(args.data_path)
    
    # Preprocess with target genes
    adata_processed = preprocess_immune_dataset(adata, target_genes=target_genes, max_cells=args.max_cells)
    
    # Convert for MoE
    data_dict = create_dataset_for_moe(adata_processed, vocab_size=args.vocab_size)
    
    # Test with MoE
    embeddings = test_moe_with_immune_data(data_dict, args.checkpoint, args.device, args.batch_size)
    
    # Evaluate
    results = evaluate_integration(
        embeddings, 
        data_dict['adata'].obs['final_annotation'].values if 'final_annotation' in data_dict['adata'].obs else data_dict['cell_types'],
        data_dict['adata'].obs['batch'].values if 'batch' in data_dict['adata'].obs else data_dict['batches'],
        data_dict['le_celltype']
    )
    
    # Create visualizations
    create_visualizations(results['adata'], "immune_human_moe_integration")
    
    print("✅ Immune Human integration test completed!")

if __name__ == "__main__":
    main()