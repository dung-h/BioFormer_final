#!/usr/bin/env python3
"""
MoE BioFormer Integration Test with correct gene vocabulary
"""
import os
import sys
import h5py
import numpy as np
import torch
import torch.nn as nn
import scanpy as sc
import anndata
import matplotlib.pyplot as plt
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

class SingleCellDataset(Dataset):
    """Dataset for loading preprocessed HDF5 files."""
    def __init__(self, data_dir, num_cell_types=185):
        self.data_dir = Path(data_dir)
        self.num_cell_types = num_cell_types
        
        self.files = sorted(self.data_dir.glob("preprocessed_study*.h5"))
        self.data = []
        
        for file in self.files:
            with h5py.File(file, 'r') as f:
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
        
        print(f"Loaded {len(self.data)} cells from {len(self.files)} studies")
        print(f"Found {len(self.label_encoder.classes_)} unique cell types")

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

def run_moe_integration_test(data_dir, checkpoint_path, genes_file, batch_size=64, device="cuda"):
    """Run integration test using MoE BioFormer model."""
    print(f"\n{'='*60}")
    print(f"Running MoE BioFormer Integration Test: {data_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")
    
    # Load vocabulary
    with open(genes_file, 'r') as f:
        genes = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(genes)} genes from vocabulary")
    
    # Load dataset
    dataset = SingleCellDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    # Load MoE model
    print("Loading MoE BioFormer model...")
    model = BioFormerMoE(
        vocab_size=1000,  # Match checkpoint vocab size
        num_cell_types=199,  # Match checkpoint cell types
        num_studies=15,  # Match actual number of studies
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
    
    print(f"\nMoE BioFormer Results:")
    print(f"  NMI = {nmi:.4f}")
    print(f"  ARI = {ari:.4f}")
    print(f"  Silhouette = {sil:.4f}")
    print(f"  Graph connectivity = {gcon:.4f}")
    print(f"  AvgBIO = {(nmi + ari + sil)/3:.4f}")
    print(f"  AvgBATCH = {gcon:.4f}")
    
    # Generate UMAP visualizations
    print("\nGenerating UMAP visualizations...")
    # Extract epoch info from checkpoint path for filename
    import re
    epoch_match = re.search(r'epoch_(\d+)', checkpoint_path)
    epoch_num = epoch_match.group(1) if epoch_match else "unknown"
    umap_filename = f"umap_moe_validation_epoch_{epoch_num}"
    create_umap_plots(adata, cell_types, study_ids, f"MoE BioFormer Epoch {epoch_num}", umap_filename)
    
    return {
        'nmi': nmi, 'ari': ari, 'silhouette': sil, 'graph_connectivity': gcon,
        'avg_bio': (nmi + ari + sil)/3, 'avg_batch': gcon,
        'adata': adata
    }

def create_umap_plots(adata, cell_types, study_ids, title, save_prefix):
    """Create UMAP plots colored by cell types and batches."""
    # Compute UMAP using scanpy
    sc.tl.umap(adata)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot by cell type with distinct colors
    unique_cell_types = sorted(list(set(cell_types)))
    n_types = len(unique_cell_types)
    
    # Use a qualitative colormap with enough distinct colors
    if n_types <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_types))
    else:
        # For more than 20 types, use hsv for maximum distinction
        colors = plt.cm.hsv(np.linspace(0, 1, n_types))
    
    # Plot cell types
    for i, cell_type in enumerate(unique_cell_types):
        mask = cell_types == cell_type
        ax1.scatter(adata.obsm['X_umap'][mask, 0], 
                   adata.obsm['X_umap'][mask, 1], 
                   c=[colors[i]], 
                   label=cell_type, 
                   s=3, 
                   alpha=0.7)
    
    ax1.set_title(f'{title} - Cell Types')
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, markerscale=3)
    
    # Plot by study/batch
    unique_studies = sorted(list(set(study_ids.astype(str))))
    study_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_studies)))
    
    for i, study in enumerate(unique_studies):
        mask = study_ids.astype(str) == study
        ax2.scatter(adata.obsm['X_umap'][mask, 0], 
                   adata.obsm['X_umap'][mask, 1], 
                   c=[study_colors[i]], 
                   label=f'Study {study}', 
                   s=3, 
                   alpha=0.7)
    
    ax2.set_title(f'{title} - Batches')
    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('UMAP 2')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, markerscale=3)
    
    plt.tight_layout()
    save_path = f"{save_prefix}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"UMAP plots saved to: {save_path}")
    return save_path

def main():
    parser = argparse.ArgumentParser(description="MoE BioFormer Integration Test")
    parser.add_argument("--data_dir", default="/home/tripham/scgpt/benchmark/preprocessed_pbmc",
                       help="Preprocessed data directory")
    parser.add_argument("--checkpoint", 
                       default="/mnt/nasdev2/dung/training1/checkpoints/checkpoint_epoch_1_20250618_140100_512d_8head_12layer.pt",
                       help="MoE model checkpoint path")
    parser.add_argument("--genes_file", 
                       default="/mnt/nasdev2/dung/preprocessed/selected_genes.txt",
                       help="Gene vocabulary file")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    
    args = parser.parse_args()
    
    results = run_moe_integration_test(
        args.data_dir, args.checkpoint, args.genes_file, 
        args.batch_size, args.device
    )
    
    print("\nMoE Integration Test Completed!")

if __name__ == "__main__":
    main()