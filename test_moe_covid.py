#!/usr/bin/env python3
"""
MoE BioFormer COVID-19 Batch Integration Test
"""
import os
import sys
import h5py
import numpy as np
import torch
import torch.nn as nn
import scanpy as sc
import anndata
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from torch.utils.data import DataLoader, Dataset
import argparse
import warnings
warnings.filterwarnings('ignore')

# Add paths for MoE model
sys.path.append('/home/tripham/scgpt/trial_3_based_moe')
sys.path.append('/home/tripham/scgpt/benchmark/BioFormer')

from models.bioformer_with_ffn_moe import BioFormerMoE
from utils.metrics import compute_graph_connectivity

class CovidDataset(Dataset):
    """Dataset for loading preprocessed COVID-19 HDF5 files."""
    def __init__(self, data_dir, num_cell_types=185):
        self.data_dir = Path(data_dir)
        self.num_cell_types = num_cell_types
        
        self.files = ["preprocessed_covid_train.h5", "preprocessed_covid_test.h5"]
        self.data = []
        
        for filename in self.files:
            file_path = self.data_dir / filename
            if not file_path.exists():
                continue
                
            with h5py.File(file_path, 'r') as f:
                binned_expr = f['binned_expr'][:]
                cell_type = f['cell_type'][:]
                non_zero_mask = f['non_zero_mask'][:]
                study_ids = f['study_ids'][:]
                
                cell_type_str = [ct.decode('utf-8') if isinstance(ct, bytes) else str(ct) for ct in cell_type]
                
                for i in range(len(binned_expr)):
                    self.data.append({
                        'binned_expr': binned_expr[i],
                        'cell_type_str': cell_type_str[i],
                        'non_zero_mask': non_zero_mask[i],
                        'study_id': study_ids[i]
                    })
        
        all_cell_types = [item['cell_type_str'] for item in self.data]
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(all_cell_types)
        
        unique_studies = list(set([item['study_id'] for item in self.data]))
        print(f"Loaded {len(self.data)} cells from {len(self.files)} files")
        print(f"Found {len(self.label_encoder.classes_)} unique cell types")
        print(f"Found {len(unique_studies)} unique studies")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        cell_type_encoded = self.label_encoder.transform([item['cell_type_str']])[0]
        
        return {
            'binned_expr': torch.tensor(item['binned_expr'], dtype=torch.long),
            'cell_type': torch.tensor(cell_type_encoded, dtype=torch.long),
            'non_zero_mask': torch.tensor(item['non_zero_mask'], dtype=torch.float32),
            'study_id': torch.tensor(item['study_id'], dtype=torch.long),
            'cell_type_str': item['cell_type_str']
        }

def custom_collate_fn(batch):
    """Custom collate function."""
    return {
        'binned_expr': torch.stack([item['binned_expr'] for item in batch]),
        'cell_type': torch.stack([item['cell_type'] for item in batch]),
        'non_zero_mask': torch.stack([item['non_zero_mask'] for item in batch]),
        'study_id': torch.stack([item['study_id'] for item in batch]),
        'cell_type_str': [item['cell_type_str'] for item in batch]
    }

def run_moe_covid_test(data_dir, checkpoint_path, genes_file, batch_size=64, device="cuda"):
    """Run COVID-19 batch integration test using MoE BioFormer model."""
    print(f"\n{'='*60}")
    print(f"Running MoE BioFormer COVID-19 Integration Test: {data_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")
    
    # Load vocabulary
    with open(genes_file, 'r') as f:
        genes = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(genes)} genes from vocabulary")
    
    # Load dataset
    dataset = CovidDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    # Load MoE model (adjusted for 454 genes instead of 1000)
    print("Loading MoE BioFormer model...")
    model = BioFormerMoE(
        vocab_size=454,  # 454 overlapping genes
        num_cell_types=dataset.num_cell_types,
        num_studies=15,  # Standard value
        d_model=256,
        nhead=4,
        num_layers=8,
        num_experts=4
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # Load state dict with shape matching
    with torch.no_grad():
        for k, v in model.state_dict().items():
            if k in state_dict and state_dict[k].shape == v.shape:
                v.copy_(state_dict[k])
            else:
                print(f"Skipping {k} due to shape mismatch or missing key")
    
    model.eval()
    print("MoE model loaded successfully")
    
    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = []
    cell_types = []
    study_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            binned_expr = batch['binned_expr'].to(device)
            cell_type = batch['cell_type'].to(device)
            non_zero_mask = batch['non_zero_mask'].to(device)
            study_id = batch['study_id'].to(device)
            
            # Get embeddings from MoE model
            out = model(
                binned_expr=binned_expr,
                cell_type=cell_type,
                study_id=study_id,
                non_zero_mask=non_zero_mask
            )
            
            # Extract CLS token (first token of output)
            cls_embeddings = out[2][:, 0, :]  # Assuming output format is (mlm, cont, hidden, attention)
            
            embeddings.append(cls_embeddings.cpu().numpy())
            cell_types.extend(batch['cell_type_str'])
            study_ids.extend(batch['study_id'].cpu().numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)
    cell_types = np.array(cell_types)
    study_ids = np.array(study_ids)
    
    print(f"Extracted embeddings shape: {embeddings.shape}")
    
    # Create AnnData and compute metrics
    adata = anndata.AnnData(X=embeddings, obs={'cell_type': cell_types, 'study_id': study_ids})
    sc.pp.neighbors(adata, use_rep="X", n_neighbors=15, metric="cosine")
    sc.tl.leiden(adata, key_added="leiden")
    
    le = LabelEncoder().fit(cell_types)
    ct_encoded = le.transform(cell_types)
    
    nmi = normalized_mutual_info_score(ct_encoded, adata.obs["leiden"])
    ari = adjusted_rand_score(ct_encoded, adata.obs["leiden"])
    sil = (silhouette_score(embeddings, ct_encoded, metric="cosine") + 1) / 2
    gcon = compute_graph_connectivity(adata, cell_types, study_ids, n_neighbors=50)
    
    print(f"\nMoE BioFormer COVID-19 Results:")
    print(f"  NMI = {nmi:.4f}")
    print(f"  ARI = {ari:.4f}")
    print(f"  Silhouette = {sil:.4f}")
    print(f"  Graph connectivity = {gcon:.4f}")
    print(f"  AvgBIO = {(nmi + ari + sil)/3:.4f}")
    print(f"  AvgBATCH = {gcon:.4f}")
    
    # Generate UMAP with scientific colors
    print("\nGenerating UMAP with scientific colors...")
    generate_scientific_umap(adata, embeddings, cell_types, study_ids)
    
    return {
        'nmi': nmi, 'ari': ari, 'silhouette': sil, 'graph_connectivity': gcon,
        'avg_bio': (nmi + ari + sil)/3, 'avg_batch': gcon,
        'adata': adata
    }

def generate_scientific_umap(adata, embeddings, cell_types, study_ids):
    """Generate UMAP with scientific color palette"""
    import matplotlib.pyplot as plt
    
    # Scientific color palette - muted and professional
    scientific_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',  # Classic matplotlib colors (muted)
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173',  # Darker variants
        '#5254a3', '#6b6ecf', '#9c9ede', '#ad494a', '#d6616b',  # Perceptually uniform
        '#e7969c', '#7b4173', '#a55194', '#ce6dbd', '#de9ed6',
        '#3182bd', '#6baed6', '#9ecae1', '#c6dbef', '#e6550d',  # Blues and oranges
        '#fd8d3c', '#fdae6b', '#fdd0a2', '#31a354', '#74c476',  # Professional greens
        '#a1d99b', '#c7e9c0', '#756bb1', '#9e9ac8', '#bcbddc'   # Purples and more
    ]
    
    # Compute UMAP
    sc.tl.umap(adata)
    
    # Get unique cell types and assign scientific colors
    unique_cell_types = sorted(adata.obs['cell_type'].unique())
    n_types = len(unique_cell_types)
    
    # Create color mapping using scientific colors
    if n_types <= len(scientific_colors):
        colors = scientific_colors[:n_types]
    else:
        # If more cell types than colors, cycle through
        colors = (scientific_colors * ((n_types // len(scientific_colors)) + 1))[:n_types]
    
    color_map = dict(zip(unique_cell_types, colors))
    
    # Plot UMAP with scientific colors - Cell types
    fig, ax = plt.subplots(figsize=(14, 10))
    
    for cell_type in unique_cell_types:
        mask = adata.obs['cell_type'] == cell_type
        ax.scatter(adata.obsm['X_umap'][mask, 0], 
                  adata.obsm['X_umap'][mask, 1],
                  c=color_map[cell_type], 
                  label=cell_type, 
                  s=0.8, 
                  alpha=0.8)
    
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title('COVID-19 Dataset Cell Type Clustering\n(BioFormer MoE, 454/1000 gene overlap)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Create legend with smaller font and outside plot
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7, markerscale=4)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    
    plt.tight_layout()
    plt.savefig('umap_moe_covid-19_scientific.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot UMAP - Batch/Study view
    fig, ax = plt.subplots(figsize=(10, 8))
    
    study_colors = ['#1f77b4', '#d62728']  # Professional blue and red
    unique_studies = sorted(adata.obs['study_id'].unique())
    study_color_map = dict(zip(unique_studies, study_colors[:len(unique_studies)]))
    
    for study in unique_studies:
        mask = adata.obs['study_id'] == study
        ax.scatter(adata.obsm['X_umap'][mask, 0], 
                  adata.obsm['X_umap'][mask, 1],
                  c=study_color_map[study], 
                  label=f'Study {study}', 
                  s=0.8, 
                  alpha=0.8)
    
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title('COVID-19 Dataset Batch Integration\n(BioFormer MoE, 454/1000 gene overlap)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, markerscale=4)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    
    plt.tight_layout()
    plt.savefig('umap_moe_covid-19_batch_scientific.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved scientific UMAPs:")
    print("- umap_moe_covid-19_scientific.png (cell types)")
    print("- umap_moe_covid-19_batch_scientific.png (batch integration)")

def main():
    parser = argparse.ArgumentParser(description="MoE BioFormer COVID-19 Integration Test")
    parser.add_argument("--data_dir", default="/home/tripham/scgpt/trial_3_based_moe/preprocessed_covid",
                       help="Preprocessed COVID data directory")
    parser.add_argument("--checkpoint", 
                       default="/mnt/nasdev2/dung/training1/checkpoints/checkpoint_epoch_4_20250615_081604_256d_4head_8layers.pt",
                       help="MoE model checkpoint path")
    parser.add_argument("--genes_file", 
                       default="/mnt/nasdev2/dung/preprocessed/selected_genes.txt",
                       help="Gene vocabulary file")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    
    args = parser.parse_args()
    
    results = run_moe_covid_test(
        args.data_dir, args.checkpoint, args.genes_file, 
        args.batch_size, args.device
    )
    
    print("\nMoE COVID-19 Integration Test Completed!")

if __name__ == "__main__":
    main()