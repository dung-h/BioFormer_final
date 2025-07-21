#!/usr/bin/env python3
"""
Fix COVID-19 UMAP colors for scientific paper
"""
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import h5py
import sys
import warnings
warnings.filterwarnings('ignore')

# Set better color palette for scientific papers
scientific_colors = [
    '#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83',  # Deep blues, purples, oranges, reds
    '#4B7C59', '#A64942', '#722F37', '#9B5D73', '#B4A7D6',  # Greens, browns, muted purples
    '#8E7DBE', '#7A6C5D', '#A8763E', '#6B5B95', '#88D8B0',  # More muted tones
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57',  # Softer contrast colors
    '#48CAE4', '#023E8A', '#7209B7', '#F72585', '#4CC9F0',  # Professional blues and accents
    '#606C38', '#283618', '#DDA15E', '#BC6C25', '#8B5CF6',  # Earth tones and purple
    '#EF4444', '#F97316', '#EAB308', '#22C55E', '#3B82F6',  # Modern systematic colors
    '#8B5A2B', '#2F4F4F', '#8FBC8F', '#CD853F', '#4682B4'   # Additional scientific colors
]

def load_covid_embeddings():
    """Load pre-computed embeddings from MoE COVID results"""
    # Try to find embedding files
    embedding_files = [
        'moe_covid_embeddings.npy',
        'covid_moe_embeddings.npy', 
        'covid_embeddings.npy'
    ]
    
    for filename in embedding_files:
        if Path(filename).exists():
            print(f"Loading embeddings from {filename}")
            return np.load(filename)
    
    print("No embedding files found. Need to run COVID test first.")
    return None

def load_covid_metadata():
    """Load COVID dataset metadata"""
    data_dir = Path('/home/tripham/scgpt/trial_3_based_moe/preprocessed_covid')
    
    cell_types = []
    study_ids = []
    
    files = ["preprocessed_covid_train.h5", "preprocessed_covid_test.h5"]
    
    for filename in files:
        file_path = data_dir / filename
        if not file_path.exists():
            continue
            
        with h5py.File(file_path, 'r') as f:
            cell_type = f['cell_type'][:]
            study_id = f['study_ids'][:]
            
            cell_type_str = [ct.decode('utf-8') if isinstance(ct, bytes) else str(ct) for ct in cell_type]
            cell_types.extend(cell_type_str)
            study_ids.extend(study_id)
    
    return cell_types, study_ids

def create_scientific_umap():
    """Create UMAP with scientific color scheme"""
    
    # Load embeddings and metadata
    embeddings = load_covid_embeddings()
    if embeddings is None:
        print("Cannot create UMAP without embeddings. Please run COVID test first.")
        return
    
    cell_types, study_ids = load_covid_metadata()
    
    if len(cell_types) != embeddings.shape[0]:
        print(f"Mismatch: {len(cell_types)} cell types vs {embeddings.shape[0]} embeddings")
        return
    
    # Create AnnData object
    adata = sc.AnnData(X=embeddings)
    adata.obs['cell_type'] = cell_types
    adata.obs['study_id'] = study_ids
    
    # Set up scanpy settings for better plots
    sc.settings.set_figure_params(dpi=300, facecolor='white', figsize=(10, 8))
    
    # Compute UMAP
    print("Computing UMAP...")
    sc.pp.neighbors(adata, n_neighbors=15, use_rep='X')
    sc.tl.umap(adata)
    
    # Get unique cell types and assign scientific colors
    unique_cell_types = sorted(adata.obs['cell_type'].unique())
    n_types = len(unique_cell_types)
    
    # Create color mapping
    if n_types <= len(scientific_colors):
        colors = scientific_colors[:n_types]
    else:
        # If more cell types than colors, extend palette
        colors = scientific_colors * (n_types // len(scientific_colors) + 1)
        colors = colors[:n_types]
    
    color_map = dict(zip(unique_cell_types, colors))
    
    # Plot UMAP with scientific colors
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for cell_type in unique_cell_types:
        mask = adata.obs['cell_type'] == cell_type
        ax.scatter(adata.obsm['X_umap'][mask, 0], 
                  adata.obsm['X_umap'][mask, 1],
                  c=color_map[cell_type], 
                  label=cell_type, 
                  s=1, 
                  alpha=0.7)
    
    ax.set_xlabel('UMAP 1', fontsize=14)
    ax.set_ylabel('UMAP 2', fontsize=14)
    ax.set_title('BioFormer COVID-19 Integration\n(454/1000 gene overlap)', fontsize=16, fontweight='bold')
    
    # Create legend with smaller font and outside plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, markerscale=3)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('umap_moe_covid-19_scientific.png', dpi=300, bbox_inches='tight')
    plt.savefig('umap_moe_covid-19_scientific.pdf', bbox_inches='tight')
    
    print("Saved scientific UMAP as:")
    print("- umap_moe_covid-19_scientific.png")
    print("- umap_moe_covid-19_scientific.pdf")
    
    # Also create study-colored version
    fig, ax = plt.subplots(figsize=(10, 8))
    
    study_colors = ['#2E86AB', '#C73E1D']  # Professional blue and red
    unique_studies = sorted(adata.obs['study_id'].unique())
    study_color_map = dict(zip(unique_studies, study_colors[:len(unique_studies)]))
    
    for study in unique_studies:
        mask = adata.obs['study_id'] == study
        ax.scatter(adata.obsm['X_umap'][mask, 0], 
                  adata.obsm['X_umap'][mask, 1],
                  c=study_color_map[study], 
                  label=f'Study {study}', 
                  s=1, 
                  alpha=0.7)
    
    ax.set_xlabel('UMAP 1', fontsize=14)
    ax.set_ylabel('UMAP 2', fontsize=14)
    ax.set_title('BioFormer COVID-19 Batch Integration\n(454/1000 gene overlap)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, markerscale=3)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('umap_moe_covid-19_batch_scientific.png', dpi=300, bbox_inches='tight')
    plt.savefig('umap_moe_covid-19_batch_scientific.pdf', bbox_inches='tight')
    
    print("- umap_moe_covid-19_batch_scientific.png")
    print("- umap_moe_covid-19_batch_scientific.pdf")

if __name__ == "__main__":
    create_scientific_umap()