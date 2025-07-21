#!/usr/bin/env python3
"""
Generate UMAP plots for FFN vs MoE BioFormer comparison
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
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.append('/home/tripham/scgpt/trial_3_based_moe')
sys.path.append('/home/tripham/scgpt/benchmark/BioFormer')

from models.bioformer_with_ffn_moe import BioFormerMoE

class scGPT(nn.Module):
    """FFN-based BioFormer model."""
    def __init__(self, vocab_size, num_cell_types, num_bins=51, d_model=256, nhead=8, num_layers=8, dropout=0.1):
        super(scGPT, self).__init__()
        self.d_model = d_model
        self.num_bins = num_bins
        
        self.gene_embedding = nn.Embedding(vocab_size, d_model)
        self.value_embedding = nn.Embedding(num_bins, d_model)
        self.cell_type_embedding = nn.Embedding(num_cell_types, d_model)
        
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        
        self.mlm_head = nn.Linear(d_model, num_bins)
        self.cont_head = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        
        self._init_weights()

    def _init_weights(self):
        for emb in [self.gene_embedding, self.value_embedding, self.cell_type_embedding]:
            nn.init.xavier_uniform_(emb.weight)
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.xavier_uniform_(self.mlm_head.weight)
        nn.init.xavier_uniform_(self.cont_head.weight)

    def forward(self, binned_expr, cell_type, non_zero_mask=None):
        batch_size, seq_len = binned_expr.shape
        device = binned_expr.device
        
        gene_emb = self.gene_embedding(torch.arange(seq_len, device=device).expand(batch_size, seq_len))
        value_emb = self.value_embedding(binned_expr)
        cell_type_emb = self.cell_type_embedding(cell_type).unsqueeze(1)
        
        emb = gene_emb + value_emb + cell_type_emb
        emb = self.norm(emb)
        
        if non_zero_mask is not None:
            emb = emb * non_zero_mask.unsqueeze(-1)
        
        output = self.transformer(emb)
        cls_token = output[:, 0, :]
        mlm_logits = self.mlm_head(output)
        cont_pred = self.cont_head(output).squeeze(-1)
        
        return mlm_logits, cont_pred, cls_token, output

class UniversalDataset(Dataset):
    """Universal dataset for loading preprocessed HDF5 files."""
    def __init__(self, data_dir, num_cell_types=185):
        self.data_dir = Path(data_dir)
        self.num_cell_types = num_cell_types
        
        # Look for all preprocessed files
        self.files = []
        for file in self.data_dir.glob("*.h5"):
            if "preprocessed" in file.name:
                self.files.append(file.name)
        
        if not self.files:
            # Fallback to standard names
            self.files = ["preprocessed_pbmc_train.h5", "preprocessed_pbmc_test.h5"]
        
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

def extract_embeddings(model, dataloader, device):
    """Extract embeddings from model."""
    embeddings = []
    cell_types = []
    study_ids = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            binned_expr = batch['binned_expr'].to(device)
            cell_type = batch['cell_type'].to(device)
            non_zero_mask = batch['non_zero_mask'].to(device)
            study_id = batch['study_id'].to(device)
            
            if hasattr(model, 'forward') and 'study_id' in model.forward.__code__.co_varnames:
                # MoE model
                out = model(
                    binned_expr=binned_expr,
                    cell_type=cell_type,
                    study_id=study_id,
                    non_zero_mask=non_zero_mask
                )
                cls_embeddings = out[2][:, 0, :]  # CLS token
            else:
                # FFN model
                _, _, cls_token, _ = model(binned_expr, cell_type, non_zero_mask)
                cls_embeddings = cls_token
            
            embeddings.append(cls_embeddings.cpu().numpy())
            cell_types.extend(batch['cell_type_str'])
            study_ids.extend(batch['study_id'].cpu().numpy())
    
    return np.concatenate(embeddings, axis=0), np.array(cell_types), np.array(study_ids)

def create_umap_plot(embeddings, cell_types, study_ids, title, save_path):
    """Create UMAP plot with distinct colors for cell types."""
    # Create AnnData object
    adata = anndata.AnnData(X=embeddings)
    adata.obs['cell_type'] = cell_types
    adata.obs['study_id'] = study_ids.astype(str)
    
    # Compute UMAP
    sc.pp.neighbors(adata, use_rep="X", n_neighbors=15, metric="cosine")
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
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return adata

def generate_all_umaps():
    """Generate UMAP plots for all model-dataset combinations."""
    device = "cuda"
    batch_size = 64
    
    # Dataset configurations
    datasets = [
        {
            'name': 'PBMC10k',
            'data_dir': '/home/tripham/scgpt/benchmark/preprocessed_pbmc',
            'genes_file': '/mnt/nasdev2/dung/preprocessed/selected_genes.txt',
            'vocab_size': 1000
        },
        {
            'name': 'COVID-19',
            'data_dir': '/home/tripham/scgpt/trial_3_based_moe/preprocessed_covid',
            'genes_file': '/mnt/nasdev2/dung/preprocessed/selected_genes.txt',
            'vocab_size': 454
        }
    ]
    
    # Model configurations
    models = [
        {
            'name': 'FFN',
            'checkpoint': '/mnt/nasdev2/dung/preprocessed/training1/checkpoints/checkpoint_epoch_2_20250705_082455.pt',
            'type': 'ffn'
        },
        {
            'name': 'MoE',
            'checkpoint': '/mnt/nasdev2/dung/training1/checkpoints/checkpoint_epoch_4_20250615_081604_256d_4head_8layers.pt',
            'type': 'moe'
        }
    ]
    
    for dataset_config in datasets:
        print(f"\n{'='*60}")
        print(f"Processing {dataset_config['name']} dataset")
        print(f"{'='*60}")
        
        # Load dataset
        dataset = UniversalDataset(dataset_config['data_dir'])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
        
        for model_config in models:
            print(f"\nGenerating UMAP for {model_config['name']} on {dataset_config['name']}")
            
            # Load model
            if model_config['type'] == 'ffn':
                model = scGPT(
                    vocab_size=dataset_config['vocab_size'],
                    num_cell_types=dataset.num_cell_types,
                    d_model=256, nhead=8, num_layers=8
                ).to(device)
            else:  # moe
                model = BioFormerMoE(
                    vocab_size=dataset_config['vocab_size'],
                    num_cell_types=dataset.num_cell_types,
                    num_studies=15,
                    d_model=256, nhead=4, num_layers=8, num_experts=4
                ).to(device)
            
            # Load checkpoint
            checkpoint = torch.load(model_config['checkpoint'], map_location=device)
            state_dict = checkpoint['model_state_dict']
            
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if k in state_dict and state_dict[k].shape == v.shape:
                        v.copy_(state_dict[k])
                    else:
                        print(f"Skipping {k} due to shape mismatch")
            
            # Extract embeddings
            embeddings, cell_types, study_ids = extract_embeddings(model, dataloader, device)
            
            # Create UMAP plot
            title = f"{model_config['name']} BioFormer - {dataset_config['name']}"
            save_path = f"/home/tripham/scgpt/trial_3_based_moe/umap_{model_config['name'].lower()}_{dataset_config['name'].lower()}.png"
            
            adata = create_umap_plot(embeddings, cell_types, study_ids, title, save_path)
            print(f"Saved UMAP plot: {save_path}")
            print(f"Embedding shape: {embeddings.shape}")
            print(f"Cell types: {len(set(cell_types))}")
            print(f"Studies: {len(set(study_ids))}")

if __name__ == "__main__":
    generate_all_umaps()
    print("\nAll UMAP plots generated successfully!")