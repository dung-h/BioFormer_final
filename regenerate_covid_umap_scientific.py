#!/usr/bin/env python3
"""
Regenerate COVID-19 UMAP with scientific colors
"""
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import h5py
import sys
import torch
import warnings
warnings.filterwarnings('ignore')

# Add paths for MoE model
sys.path.append('/home/tripham/scgpt/trial_3_based_moe')
sys.path.append('/home/tripham/scgpt/benchmark/BioFormer')

from models.bioformer_with_ffn_moe import BioFormerMoE
from utils.metrics import compute_graph_connectivity
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Scientific color palette - more muted and professional
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
                        'cell_type': cell_type[i],
                        'cell_type_str': cell_type_str[i],
                        'non_zero_mask': non_zero_mask[i],
                        'study_id': study_ids[i]
                    })
        
        print(f"Loaded {len(self.data)} cells from {len(self.files)} files")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def custom_collate_fn(batch):
    return {
        'binned_expr': torch.stack([torch.FloatTensor(item['binned_expr']) for item in batch]),
        'cell_type': torch.stack([torch.LongTensor([item['cell_type']]) for item in batch]).squeeze(),
        'cell_type_str': [item['cell_type_str'] for item in batch],
        'non_zero_mask': torch.stack([torch.BoolTensor(item['non_zero_mask']) for item in batch]),
        'study_id': torch.stack([torch.LongTensor([item['study_id']]) for item in batch]).squeeze()
    }

def extract_embeddings_and_create_umap():
    """Extract embeddings and create scientific UMAP"""
    
    # Load dataset
    dataset = CovidDataset("/home/tripham/scgpt/trial_3_based_moe/preprocessed_covid")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=custom_collate_fn)
    
    # Load genes
    with open("/mnt/nasdev2/dung/preprocessed/selected_genes.txt", "r") as f:
        genes = [line.strip() for line in f.readlines()]
    
    print(f"Loaded {len(genes)} genes from vocabulary")
    print(f"Loaded {len(dataset)} cells from {dataset.num_cell_types} cell types")
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = BioFormerMoE(
        vocab_size=len(genes),
        num_cell_types=dataset.num_cell_types,
        num_studies=15,
        d_model=256,
        nhead=4,
        num_layers=8,
        num_experts=4
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = "/mnt/nasdev2/dung/training1/checkpoints/checkpoint_epoch_4_20250615_081604_256d_4head_8layers.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # Remove mismatched keys
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if key in model_state_dict and model_state_dict[key].shape == value.shape:
            filtered_state_dict[key] = value
        else:
            print(f"Skipping {key} due to shape mismatch or missing key")
    
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    
    print("MoE model loaded successfully")
    
    # Extract embeddings
    embeddings = []
    cell_types = []
    study_ids = []
    
    print("Extracting embeddings...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            binned_expr = batch['binned_expr'].to(device)
            cell_type = batch['cell_type'].to(device)
            study_id = batch['study_id'].to(device)
            non_zero_mask = batch['non_zero_mask'].to(device)
            
            out = model(
                binned_expr=binned_expr,
                cell_type=cell_type,
                study_id=study_id,
                non_zero_mask=non_zero_mask
            )
            
            # Extract CLS token (first token of output)
            cls_embeddings = out[2][:, 0, :]
            
            embeddings.append(cls_embeddings.cpu().numpy())
            cell_types.extend(batch['cell_type_str'])
            study_ids.extend(batch['study_id'].cpu().numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)
    cell_types = np.array(cell_types)
    study_ids = np.array(study_ids)
    
    print(f"Extracted embeddings shape: {embeddings.shape}")
    
    # Save embeddings for future use
    np.save('covid_moe_embeddings.npy', embeddings)
    np.save('covid_cell_types.npy', cell_types)
    np.save('covid_study_ids.npy', study_ids)
    print("Saved embeddings and metadata")
    
    # Create AnnData object
    adata = sc.AnnData(X=embeddings)
    adata.obs['cell_type'] = cell_types
    adata.obs['study_id'] = study_ids
    
    # Set up scanpy settings for better plots
    sc.settings.set_figure_params(dpi=300, facecolor='white')
    
    # Compute UMAP
    print("Computing UMAP...")
    sc.pp.neighbors(adata, n_neighbors=15, use_rep='X', metric='cosine')
    sc.tl.umap(adata)
    
    # Get unique cell types and assign scientific colors
    unique_cell_types = sorted(adata.obs['cell_type'].unique())
    n_types = len(unique_cell_types)
    print(f"Found {n_types} unique cell types")
    
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
    plt.savefig('umap_moe_covid-19_scientific.pdf', bbox_inches='tight')
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
    plt.savefig('umap_moe_covid-19_batch_scientific.pdf', bbox_inches='tight')
    plt.close()
    
    print("\nSaved scientific UMAPs:")
    print("- umap_moe_covid-19_scientific.png (cell types)")
    print("- umap_moe_covid-19_batch_scientific.png (batch integration)")
    print("- PDF versions also saved")

if __name__ == "__main__":
    extract_embeddings_and_create_umap()