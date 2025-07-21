#!/usr/bin/env python3
"""
FFN BioFormer Integration Test with correct gene vocabulary (same as MoE test)
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
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import warnings
warnings.filterwarnings('ignore')

# Add BioFormer utils to path
sys.path.append('/home/tripham/scgpt/benchmark/BioFormer')
from utils.metrics import compute_graph_connectivity

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

def run_ffn_integration_test():
    """Run integration test using FFN BioFormer model."""
    data_dir = "/home/tripham/scgpt/benchmark/preprocessed_pbmc"
    checkpoint_path = "/mnt/nasdev2/dung/preprocessed/training1/checkpoints/checkpoint_epoch_2_20250705_082455.pt"
    genes_file = "/mnt/nasdev2/dung/preprocessed/selected_genes.txt"
    batch_size = 64
    device = "cuda"
    
    print(f"\n{'='*60}")
    print(f"Running FFN BioFormer Integration Test: {data_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")
    
    # Load vocabulary
    with open(genes_file, 'r') as f:
        genes = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(genes)} genes from vocabulary")
    
    # Load dataset
    dataset = SingleCellDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    # Load FFN model
    print("Loading FFN BioFormer model...")
    model = scGPT(vocab_size=len(genes), num_cell_types=dataset.num_cell_types)
    model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    with torch.no_grad():
        for k, v in model.state_dict().items():
            if k in state_dict and state_dict[k].shape == v.shape:
                v.copy_(state_dict[k])
            else:
                print(f"Skipping {k} due to shape mismatch or missing key")
    
    model.eval()
    print("FFN model loaded successfully")
    
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
            
            _, _, cls_token, _ = model(binned_expr, cell_type, non_zero_mask)
            
            embeddings.append(cls_token.cpu().numpy())
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
    
    print(f"\nFFN BioFormer Results:")
    print(f"  NMI = {nmi:.4f}")
    print(f"  ARI = {ari:.4f}")
    print(f"  Silhouette = {sil:.4f}")
    print(f"  Graph connectivity = {gcon:.4f}")
    print(f"  AvgBIO = {(nmi + ari + sil)/3:.4f}")
    print(f"  AvgBATCH = {gcon:.4f}")
    
    return {
        'nmi': nmi, 'ari': ari, 'silhouette': sil, 'graph_connectivity': gcon,
        'avg_bio': (nmi + ari + sil)/3, 'avg_batch': gcon
    }

if __name__ == "__main__":
    results = run_ffn_integration_test()
    print("\nFFN Integration Test Completed!")