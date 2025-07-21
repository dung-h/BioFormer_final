#!/usr/bin/env python3
"""
scGPT vs BioFormer comparison on PBMC batch integration
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('/home/tripham/scgpt/trial_3_based_moe')

def load_pbmc_data():
    """Load PBMC data for comparison"""
    print("📊 Loading PBMC data...")
    
    batch0_path = '/home/tripham/scgpt/benchmark/data/pbmc10k/pbmc_batch0.h5ad'
    batch1_path = '/home/tripham/scgpt/benchmark/data/pbmc10k/pbmc_batch1.h5ad'
    
    adata0 = sc.read_h5ad(batch0_path)
    adata1 = sc.read_h5ad(batch1_path)
    
    # Add batch info
    adata0.obs['batch'] = 'batch_0'
    adata1.obs['batch'] = 'batch_1'
    
    # Combine
    adata = adata0.concatenate(adata1, batch_key='study_id', batch_categories=['batch_0', 'batch_1'])
    
    print(f"  Dataset shape: {adata.shape}")
    print(f"  Cell types: {len(adata.obs['str_labels'].unique())}")
    print(f"  Batches: {len(adata.obs['study_id'].unique())}")
    
    return adata

def run_scgpt_benchmark(adata):
    """Run scGPT benchmark for batch integration"""
    print("\n🔬 Running scGPT benchmark...")
    
    try:
        import scgpt as scg
        from scgpt.model import TransformerModel
        from scgpt.tokenizer import tokenize_and_pad_batch
        from scgpt.utils import set_seed
        from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
        
        start_time = time.time()
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Prepare data for scGPT
        adata_scgpt = adata.copy()
        
        # Normalize if needed
        if adata_scgpt.X.max() > 20:
            sc.pp.normalize_total(adata_scgpt, target_sum=1e4)
            sc.pp.log1p(adata_scgpt)
        
        # Get top genes (scGPT can handle variable number of genes)
        sc.pp.highly_variable_genes(adata_scgpt, n_top_genes=2000, flavor='seurat_v3')
        adata_scgpt = adata_scgpt[:, adata_scgpt.var.highly_variable]
        
        print(f"  Using {adata_scgpt.shape[1]} genes")
        
        # Create vocabulary from gene names
        vocab = scg.utils.get_vocab_from_genes(adata_scgpt.var_names.tolist())
        
        # Model configuration
        model_config = {
            'ntoken': len(vocab),
            'n_input_bins': 51,  # Number of expression bins
            'd_model': 512,
            'nhead': 8,
            'nlayers': 12,
            'nlayers_cls': 3,
            'n_cls': 1,
            'dropout': 0.2,
            'pad_token': vocab['<pad>'],
            'n_hvg': adata_scgpt.shape[1],
            'max_seq_len': adata_scgpt.shape[1] + 1,
        }
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        model = TransformerModel(
            ntoken=model_config['ntoken'],
            d_model=model_config['d_model'],
            nhead=model_config['nhead'],
            nlayers=model_config['nlayers'],
            nlayers_cls=model_config['nlayers_cls'],
            n_cls=model_config['n_cls'],
            vocab=vocab,
            dropout=model_config['dropout'],
            pad_token=model_config['pad_token'],
            pad_value=0,
            do_mvc=True,
            do_dab=False,
            use_batch_labels=True,
            n_input_bins=model_config['n_input_bins'],
            ecs_threshold=0.0,
            explicit_zero_prob=True,
            use_fast_transformer=True,
        )
        
        model.to(device)
        model.eval()
        
        print("  Model initialized")
        
        # Tokenize data
        print("  Tokenizing data...")
        
        # Convert to gene expression matrix
        X = adata_scgpt.X.toarray() if hasattr(adata_scgpt.X, 'toarray') else adata_scgpt.X
        
        # Bin expression values
        def bin_expression(X, n_bins=51):
            """Bin expression values"""
            X_binned = np.zeros_like(X, dtype=np.int32)
            
            for i in range(X.shape[0]):
                cell_expr = X[i]
                # Handle zeros separately
                nonzero_mask = cell_expr > 0
                if np.sum(nonzero_mask) > 0:
                    nonzero_expr = cell_expr[nonzero_mask]
                    # Create bins from 1 to n_bins-1 for non-zero values
                    bins = np.linspace(np.min(nonzero_expr), np.max(nonzero_expr), n_bins-1)
                    binned_nonzero = np.digitize(nonzero_expr, bins) + 1
                    X_binned[i, nonzero_mask] = binned_nonzero
                    # Zero values remain 0
            
            return X_binned
        
        X_binned = bin_expression(X, model_config['n_input_bins'])
        
        # Create gene tokens
        gene_tokens = []
        for gene_name in adata_scgpt.var_names:
            if gene_name in vocab:
                gene_tokens.append(vocab[gene_name])
            else:
                gene_tokens.append(vocab['<unk>'])
        
        # Extract embeddings
        print("  Extracting embeddings...")
        embeddings = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(X_binned), batch_size):
                try:
                    batch_end = min(i + batch_size, len(X_binned))
                    batch_X = X_binned[i:batch_end]
                    batch_size_actual = batch_X.shape[0]
                    
                    # Create input tensors
                    batch_genes = torch.tensor(gene_tokens).unsqueeze(0).repeat(batch_size_actual, 1).to(device)
                    batch_values = torch.tensor(batch_X).to(device)
                    
                    # Forward pass
                    with torch.cuda.amp.autocast():
                        output = model(batch_genes, batch_values)
                        # Use the cell embedding (CLS token)
                        cell_embeddings = output["cell_emb"].cpu().numpy()
                        embeddings.append(cell_embeddings)
                        
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"    Batch {i//batch_size} failed: {e}")
                    # Use zero embeddings as fallback
                    batch_size_actual = min(batch_size, len(X_binned) - i)
                    batch_emb = np.zeros((batch_size_actual, model_config['d_model']))
                    embeddings.append(batch_emb)
        
        embeddings = np.vstack(embeddings)
        print(f"  Embeddings shape: {embeddings.shape}")
        
        # Evaluation metrics
        print("  Computing evaluation metrics...")
        
        # Cell type labels
        cell_types = pd.Categorical(adata.obs['str_labels']).codes
        batch_labels = pd.Categorical(adata.obs['study_id']).codes
        
        # Batch integration metrics
        from scgpt.utils import eval_scib_metrics
        
        # Create temporary AnnData with embeddings
        adata_emb = adata.copy()
        adata_emb.obsm['X_emb'] = embeddings
        
        # Calculate silhouette scores
        sil_batch = silhouette_score(embeddings, batch_labels)
        sil_celltype = silhouette_score(embeddings, cell_types)
        
        # Calculate mixing metrics
        batch_nmi = normalized_mutual_info_score(batch_labels, 
                                                 pd.Categorical(sc.pp.neighbors(adata_emb, use_rep='X_emb', copy=True).obs['leiden']).codes)
        
        # Bio conservation (cell type preservation)
        celltype_nmi = normalized_mutual_info_score(cell_types, 
                                                   pd.Categorical(sc.pp.neighbors(adata_emb, use_rep='X_emb', copy=True).obs['leiden']).codes)
        
        # LISI scores (if available)
        try:
            from scib.metrics import lisi_graph
            lisi_batch = lisi_graph(adata_emb, batch_key='study_id', type_='knn', use_rep='X_emb')
            lisi_celltype = lisi_graph(adata_emb, batch_key='str_labels', type_='knn', use_rep='X_emb')
        except:
            lisi_batch = None
            lisi_celltype = None
        
        runtime = time.time() - start_time
        
        results = {
            'model': 'scGPT',
            'runtime_seconds': runtime,
            'n_cells': int(adata.shape[0]),
            'n_genes': int(adata_scgpt.shape[1]),
            'embedding_dim': int(embeddings.shape[1]),
            'silhouette_batch': float(sil_batch),
            'silhouette_celltype': float(sil_celltype),
            'batch_nmi': float(batch_nmi),
            'celltype_nmi': float(celltype_nmi),
            'lisi_batch': float(lisi_batch.mean()) if lisi_batch is not None else None,
            'lisi_celltype': float(lisi_celltype.mean()) if lisi_celltype is not None else None,
            'model_config': model_config
        }
        
        print(f"  Results: Sil_batch={sil_batch:.4f}, Sil_celltype={sil_celltype:.4f}, Batch_NMI={batch_nmi:.4f}")
        return results
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return {'model': 'scGPT', 'error': str(e)}

def run_bioformer_benchmark(adata):
    """Run BioFormer benchmark using proper integration script approach"""
    print("\n🧠 Running BioFormer benchmark...")
    
    try:
        from models.bioformer_with_ffn_moe import BioFormerMoE
        from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
        
        start_time = time.time()
        
        # Load gene vocabulary
        genes_file = '/home/tripham/scgpt/trial_3_based_moe/data/selected_genes.txt'
        with open(genes_file, 'r') as f:
            selected_genes = [line.strip() for line in f.readlines()]
        
        print(f"  Using {len(selected_genes)} genes from vocabulary")
        
        # Model setup
        model_config = {
            'vocab_size': 1000,
            'num_cell_types': 199,
            'num_studies': 15,
            'd_model': 256,
            'nhead': 4,
            'num_layers': 8,
            'dropout': 0.1,
            'num_experts': 4
        }
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = BioFormerMoE(**model_config)
        
        # Load checkpoint
        checkpoint_path = '/mnt/nasdev2/dung/training1/checkpoints/checkpoint_epoch_4_20250615_081604_256d_4head_8layers.pt'
        
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
                # Shape-aware loading
                state_dict = checkpoint['model_state_dict']
                model_state = model.state_dict()
                
                filtered_state = {}
                for k, v in state_dict.items():
                    if k in model_state and model_state[k].shape == v.shape:
                        filtered_state[k] = v
                
                model.load_state_dict(filtered_state, strict=False)
                print("  ✅ Checkpoint loaded with shape checking")
            except Exception as e:
                print(f"  ⚠️ Checkpoint failed: {e}")
        
        model.to(device)
        model.eval()
        
        # Preprocess data
        print("  Preprocessing data...")
        adata_bio = adata.copy()
        
        # Normalize
        if adata_bio.X.max() > 20:
            sc.pp.normalize_total(adata_bio, target_sum=1e4)
            sc.pp.log1p(adata_bio)
        
        # Select genes matching vocabulary
        X = adata_bio.X.toarray() if hasattr(adata_bio.X, 'toarray') else adata_bio.X
        
        # Find intersection of genes
        available_genes = adata_bio.var_names.tolist()
        gene_indices = []
        
        for gene in selected_genes:
            if gene in available_genes:
                gene_indices.append(available_genes.index(gene))
        
        print(f"  Found {len(gene_indices)} matching genes")
        
        # Select and pad to 1000 genes
        if len(gene_indices) > 0:
            X_selected = X[:, gene_indices]
            # Pad to 1000 if needed
            if X_selected.shape[1] < 1000:
                padding = np.zeros((X_selected.shape[0], 1000 - X_selected.shape[1]))
                X_selected = np.hstack([X_selected, padding])
            else:
                X_selected = X_selected[:, :1000]
        else:
            # Fallback: use HVGs
            sc.pp.highly_variable_genes(adata_bio, n_top_genes=1000, flavor='seurat_v3')
            hvg_indices = np.where(adata_bio.var.highly_variable)[0][:1000]
            X_selected = X[:, hvg_indices]
            if X_selected.shape[1] < 1000:
                padding = np.zeros((X_selected.shape[0], 1000 - X_selected.shape[1]))
                X_selected = np.hstack([X_selected, padding])
        
        # Quantile binning (same as training)
        def quantile_binning(expr, num_bins=51):
            binned = np.zeros_like(expr, dtype=np.uint8)
            for i in range(expr.shape[0]):
                cell = expr[i]
                nz = cell[cell > 0]
                if nz.size > 1:
                    try:
                        q_points = np.linspace(0, 1, num_bins)[1:]
                        thresholds = np.quantile(nz, q_points)
                        bins = np.digitize(cell, thresholds, right=True) + 1
                        bins[cell == 0] = 0
                        binned[i] = np.clip(bins, 0, num_bins - 1)
                    except:
                        max_val = np.max(cell)
                        if max_val > 0:
                            bins = (cell / max_val * (num_bins - 1)).astype(int)
                            binned[i] = bins
            return binned
        
        print("  Quantile binning...")
        X_binned = quantile_binning(X_selected)
        
        # Extract embeddings
        print("  Extracting embeddings...")
        embeddings = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(X_binned), batch_size):
                try:
                    batch_data = torch.from_numpy(X_binned[i:i+batch_size]).long().to(device)
                    batch_cell_types = torch.zeros(batch_data.shape[0], dtype=torch.long).to(device)
                    
                    _, _, output, _ = model(batch_data, batch_cell_types)
                    batch_emb = output[:, 0].cpu().numpy()
                    embeddings.append(batch_emb)
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"    Batch {i//batch_size} failed: {e}")
                    batch_size_actual = min(batch_size, len(X_binned) - i)
                    batch_emb = np.zeros((batch_size_actual, model_config['d_model']))
                    embeddings.append(batch_emb)
        
        embeddings = np.vstack(embeddings)
        print(f"  Embeddings shape: {embeddings.shape}")
        
        # Evaluation metrics
        print("  Computing evaluation metrics...")
        
        cell_types = pd.Categorical(adata.obs['str_labels']).codes
        batch_labels = pd.Categorical(adata.obs['study_id']).codes
        
        # Batch integration metrics
        sil_batch = silhouette_score(embeddings, batch_labels)
        sil_celltype = silhouette_score(embeddings, cell_types)
        
        # Create temporary AnnData with embeddings for clustering
        adata_emb = adata.copy()
        adata_emb.obsm['X_emb'] = embeddings
        
        # Clustering-based metrics
        from sklearn.cluster import KMeans
        
        # Batch mixing (lower is better)
        batch_kmeans = KMeans(n_clusters=2, random_state=42)
        batch_pred = batch_kmeans.fit_predict(embeddings)
        batch_nmi = normalized_mutual_info_score(batch_labels, batch_pred)
        
        # Cell type preservation
        celltype_kmeans = KMeans(n_clusters=len(np.unique(cell_types)), random_state=42)
        celltype_pred = celltype_kmeans.fit_predict(embeddings)
        celltype_nmi = normalized_mutual_info_score(cell_types, celltype_pred)
        
        runtime = time.time() - start_time
        
        results = {
            'model': 'BioFormer',
            'runtime_seconds': runtime,
            'n_cells': int(adata.shape[0]),
            'n_genes': 1000,
            'embedding_dim': int(embeddings.shape[1]),
            'silhouette_batch': float(sil_batch),
            'silhouette_celltype': float(sil_celltype),
            'batch_nmi': float(batch_nmi),
            'celltype_nmi': float(celltype_nmi),
            'genes_matched': len(gene_indices),
            'model_config': model_config
        }
        
        print(f"  Results: Sil_batch={sil_batch:.4f}, Sil_celltype={sil_celltype:.4f}, Batch_NMI={batch_nmi:.4f}")
        return results
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return {'model': 'BioFormer', 'error': str(e)}

def main():
    """Run scGPT vs BioFormer comparison"""
    print("🚀 scGPT vs BioFormer Batch Integration Comparison")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Load data
    adata = load_pbmc_data()
    
    # Run benchmarks
    print(f"\n🎯 Benchmarking on {adata.shape[0]} cells")
    
    scgpt_results = run_scgpt_benchmark(adata)
    bioformer_results = run_bioformer_benchmark(adata)
    
    # Compile results
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'dataset_info': {
            'shape': list(adata.shape),
            'n_batches': len(adata.obs['study_id'].unique()),
            'cell_types': len(adata.obs['str_labels'].unique())
        },
        'results': {
            'scgpt': scgpt_results,
            'bioformer': bioformer_results
        }
    }
    
    # Add comparison
    if 'error' not in scgpt_results and 'error' not in bioformer_results:
        comparison = {}
        for metric in ['silhouette_batch', 'silhouette_celltype', 'batch_nmi', 'celltype_nmi']:
            if metric in scgpt_results and metric in bioformer_results:
                improvement = bioformer_results[metric] - scgpt_results[metric]
                comparison[f'{metric}_improvement'] = improvement
                
                # For batch metrics, lower is better (better integration)
                if 'batch' in metric:
                    comparison[f'{metric}_winner'] = 'BioFormer' if improvement < 0 else 'scGPT'
                else:
                    comparison[f'{metric}_winner'] = 'BioFormer' if improvement > 0 else 'scGPT'
        
        final_results['comparison'] = comparison
    
    # Save results
    output_path = '/home/tripham/scgpt/trial_3_based_moe/scgpt_vs_bioformer_results.json'
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    
    # Print summary
    print("\n📊 COMPARISON RESULTS")
    print("=" * 60)
    
    for name, result in [('scGPT', scgpt_results), ('BioFormer', bioformer_results)]:
        if 'error' not in result:
            print(f"{name:12} | Sil_batch: {result.get('silhouette_batch', 0):.4f} | Sil_celltype: {result.get('silhouette_celltype', 0):.4f} | Batch_NMI: {result.get('batch_nmi', 0):.4f}")
        else:
            print(f"{name:12} | ERROR: {result['error']}")
    
    if 'comparison' in final_results:
        print("\n🏆 WINNER SUMMARY:")
        comp = final_results['comparison']
        for metric in ['silhouette_batch', 'silhouette_celltype', 'batch_nmi', 'celltype_nmi']:
            if f'{metric}_winner' in comp:
                winner = comp[f'{metric}_winner']
                improvement = comp[f'{metric}_improvement']
                print(f"  {metric:18}: {winner} ({improvement:+.4f})")
    
    print("\n✅ Comparison completed!")
    return final_results

if __name__ == '__main__':
    main()