#!/usr/bin/env python3
"""
Universal Integration Test for BioFormer Models
Supports testing on multiple datasets with different model types (MoE, FFN)
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
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import argparse
import warnings
warnings.filterwarnings('ignore')

# Add BioFormer utils to path
sys.path.append('/home/tripham/scgpt/benchmark/BioFormer')
from utils.metrics import compute_graph_connectivity

def quantile_bin(x, B: int = 51) -> np.ndarray:
    """51-bin quantile binning with dedicated zero-bin."""
    out = np.zeros_like(x, dtype=np.uint8)
    for i, row in enumerate(x):
        nz = row[row > 0]
        if nz.size:
            qs = np.quantile(nz, np.linspace(0, 1, B)[1:])
            bins = np.digitize(row, qs, right=True) + 1
            bins[row == 0] = 0
            out[i] = bins
    return out

def _get_per_cell_types(adata: sc.AnnData) -> np.ndarray:
    """Extract cell types from AnnData object."""
    if "cell_type" in adata.obs:
        names = adata.obs["cell_type"].astype(str).values
    elif "str_labels" in adata.obs:
        names = adata.obs["str_labels"].astype(str).values
    elif {"labels"} <= set(adata.obs.columns) and "cell_types" in adata.uns:
        lut = dict(enumerate(map(str, adata.uns["cell_types"])))
        names = adata.obs["labels"].map(lut).astype(str).values
    else:
        names = np.full(adata.n_obs, "unknown", dtype=str)
    return np.array([s.encode() for s in names], dtype="S")

def preprocess_h5ad_to_h5(h5ad_file: Path, genes: list, study_id: int, out_dir: Path):
    """Preprocess .h5ad file to .h5 format aligned with gene vocabulary."""
    print(f"Processing {h5ad_file.name}...")
    
    adata = sc.read_h5ad(h5ad_file)
    if adata.raw is not None:
        adata = adata.raw
    
    # Handle gene vocabulary alignment
    if 'gene_ids-1-0-1' in adata.var.columns:
        # Use ENSEMBL IDs
        gene_ids = adata.var['gene_ids-1-0-1'].values
        g2ix = {g: i for i, g in enumerate(gene_ids)}
        print(f"  Using ENSEMBL IDs from 'gene_ids-1-0-1' column")
    else:
        # Use gene names from index
        g2ix = {g: i for i, g in enumerate(adata.var_names)}
        print(f"  Using gene names from index")
    
    # Pad to vocabulary size
    V = len(genes)
    Xpad = np.zeros((adata.n_obs, V), dtype=np.float32)
    src = [g2ix[g] for g in genes if g in g2ix]
    dst = [j for j, g in enumerate(genes) if g in g2ix]
    
    if src:
        Xsrc = adata.X[:, src]
        Xsrc = Xsrc.toarray() if not isinstance(Xsrc, np.ndarray) else Xsrc
        Xpad[:, dst] = Xsrc
    
    overlap = len(src) / len(genes) * 100
    print(f"  Gene overlap: {overlap:.1f}% ({len(src)}/{len(genes)})")
    
    # Continuous expression (log-TPM)
    adata_tmp = sc.AnnData(Xpad)
    sc.pp.normalize_total(adata_tmp, target_sum=1e4)
    sc.pp.log1p(adata_tmp)
    expr_cont = adata_tmp.X.astype(np.float32)
    
    # Discrete binning and masks
    binned = quantile_bin(Xpad)
    nz_mask = (Xpad > 0).astype(np.uint8)
    
    # Handle cell types
    cell_types = _get_per_cell_types(adata)
    
    # Save processed data
    out_file = out_dir / f"preprocessed_study{study_id}.h5"
    with h5py.File(out_file, "w") as h5:
        h5.create_dataset("binned_expr", data=binned, compression="gzip")
        h5.create_dataset("expr_cont", data=expr_cont, compression="gzip")
        h5.create_dataset("non_zero_mask", data=nz_mask, compression="gzip")
        h5.create_dataset("study_ids", data=np.full(adata.n_obs, study_id, dtype=np.int64))
        h5.create_dataset("cell_type", data=cell_types)
        h5.create_dataset("var_names", data=np.array(genes, dtype="S"))
    
    print(f"  Saved to {out_file.name}")
    return overlap

class scGPT(nn.Module):
    """FFN-based BioFormer model."""
    def __init__(self, vocab_size, num_cell_types, num_bins=51, d_model=512, nhead=8, num_layers=8, dropout=0.1):
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

def run_integration_test(data_dir, checkpoint_path, genes_file, batch_size=64, device="cuda"):
    """Run integration test on preprocessed data."""
    print(f"\n{'='*60}")
    print(f"Running Integration Test: {data_dir}")
    print(f"{'='*60}")
    
    # Load vocabulary
    with open(genes_file, 'r') as f:
        genes = [line.strip() for line in f.readlines()]
    
    # Load dataset
    dataset = SingleCellDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    # Load model
    model = scGPT(vocab_size=len(genes), num_cell_types=dataset.num_cell_types)
    model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    with torch.no_grad():
        for k, v in model.state_dict().items():
            if k in state_dict and state_dict[k].shape == v.shape:
                v.copy_(state_dict[k])
    
    model.eval()
    
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
    
    print(f"\nResults:")
    print(f"  NMI = {nmi:.4f}")
    print(f"  ARI = {ari:.4f}")
    print(f"  Silhouette = {sil:.4f}")
    print(f"  Graph connectivity = {gcon:.4f}")
    print(f"  AvgBIO = {(nmi + ari + sil)/3:.4f}")
    print(f"  AvgBATCH = {gcon:.4f}")
    
    # Generate UMAP if requested
    sc.tl.umap(adata, min_dist=0.3)
    
    return {
        'nmi': nmi, 'ari': ari, 'silhouette': sil, 'graph_connectivity': gcon,
        'avg_bio': (nmi + ari + sil)/3, 'avg_batch': gcon,
        'adata': adata
    }

def main():
    parser = argparse.ArgumentParser(description="Universal BioFormer Integration Test")
    parser.add_argument("--dataset", required=True, 
                       choices=['pbmc10k', 'covid19', 'lung_kim'],
                       help="Dataset to test")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--genes_file", 
                       default="/mnt/nasdev2/dung/preprocessed/selected_genes.txt",
                       help="Gene vocabulary file")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_umap", action="store_true", help="Save UMAP plots")
    
    args = parser.parse_args()
    
    # Define dataset paths
    dataset_configs = {
        'pbmc10k': {
            'raw_files': [
                '/home/tripham/scgpt/benchmark/data/pbmc10k/pbmc_batch0.h5ad',
                '/home/tripham/scgpt/benchmark/data/pbmc10k/pbmc_batch1.h5ad'
            ],
            'preprocessed_dir': '/tmp/pbmc10k_preprocessed'
        },
        'covid19': {
            'raw_files': [
                '/home/tripham/scgpt/trial_3_based_moe/data/additional_datasets/COVID-19-splitted/covid/batch_covid_subsampled_train.h5ad',
                '/home/tripham/scgpt/trial_3_based_moe/data/additional_datasets/COVID-19-splitted/covid/batch_covid_subsampled_test.h5ad'
            ],
            'preprocessed_dir': '/tmp/covid19_preprocessed'
        },
        'lung_kim': {
            'raw_files': [
                '/home/tripham/scgpt/trial_3_based_moe/data/lung_kim_ensembl/lung_train_ensembl.h5ad',
                '/home/tripham/scgpt/trial_3_based_moe/data/lung_kim_ensembl/lung_test_ensembl.h5ad'
            ],
            'preprocessed_dir': '/tmp/lung_kim_preprocessed'
        }
    }
    
    config = dataset_configs[args.dataset]
    
    # Load genes
    with open(args.genes_file, 'r') as f:
        genes = [line.strip() for line in f.readlines()]
    
    # Create preprocessed directory
    prep_dir = Path(config['preprocessed_dir'])
    prep_dir.mkdir(exist_ok=True)
    
    # Preprocess files
    print(f"Preprocessing {args.dataset} dataset...")
    overlaps = []
    for i, raw_file in enumerate(config['raw_files']):
        if Path(raw_file).exists():
            overlap = preprocess_h5ad_to_h5(Path(raw_file), genes, i, prep_dir)
            overlaps.append(overlap)
        else:
            print(f"Warning: {raw_file} not found")
    
    # Save genes file
    (prep_dir / "selected_genes.txt").write_text("\n".join(genes))
    
    print(f"Average gene overlap: {np.mean(overlaps):.1f}%")
    
    # Run integration test
    results = run_integration_test(
        prep_dir, args.checkpoint, args.genes_file, 
        args.batch_size, args.device
    )
    
    # Save UMAP if requested
    if args.save_umap:
        figures_dir = Path("figures")
        figures_dir.mkdir(exist_ok=True)
        
        plt.figure(figsize=(8, 6))
        sc.pl.umap(results['adata'], color="study_id", show=False, frameon=False, 
                  title=f"{args.dataset} - Study ID")
        plt.savefig(figures_dir / f"umap_{args.dataset}_study.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        plt.figure(figsize=(8, 6))
        sc.pl.umap(results['adata'], color="cell_type", show=False, frameon=False,
                  title=f"{args.dataset} - Cell Type")
        plt.savefig(figures_dir / f"umap_{args.dataset}_celltype.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"UMAP plots saved to {figures_dir}/")

if __name__ == "__main__":
    main()