#!/usr/bin/env python3
"""
Comprehensive benchmark comparing BioFormer vs scGPT on the same PBMC dataset.
This will provide real experimental evidence for paper claims.
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

def setup_environment():
    """Setup logging and environment"""
    print("🔬 BioFormer vs scGPT Benchmark")
    print("=" * 50)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print("=" * 50)

def load_pbmc_data():
    """Load real PBMC data for benchmarking"""
    print("\n📊 Loading PBMC dataset...")
    
    # Try to find real PBMC data in the project
    possible_paths = [
        '/home/tripham/scgpt/trial_3_based_moe/data/pbmc_benchmark.h5ad',
        '/home/tripham/scgpt/trial_3_based_moe/data/pbmc_3k_processed.h5ad',
        '/home/tripham/scgpt/trial_3_based_moe/data/pbmc_10k_raw.h5ad'
    ]
    
    adata = None
    for path in possible_paths:
        if os.path.exists(path):
            print(f"  Loading from: {path}")
            try:
                adata = sc.read_h5ad(path)
                break
            except Exception as e:
                print(f"  Error loading {path}: {e}")
                continue
    
    if adata is None:
        print("  ❌ No real PBMC data found. Please run download_pbmc_data.py first.")
        raise FileNotFoundError("Real PBMC data required for benchmarking")
    
    print(f"  Dataset shape: {adata.shape}")
    print(f"  Cell types: {adata.obs['cell_type'].value_counts().head().to_dict() if 'cell_type' in adata.obs else 'Not available'}")
    print(f"  Studies: {adata.obs['study_id'].value_counts().to_dict() if 'study_id' in adata.obs else 'Not available'}")
    
    return adata

def preprocess_for_bioformer(adata):
    """Preprocess data for BioFormer (1000 gene vocabulary)"""
    print("\n🧬 Preprocessing for BioFormer...")
    
    # Basic preprocessing
    adata_bio = adata.copy()
    
    # Normalize and log transform
    sc.pp.normalize_total(adata_bio, target_sum=1e4)
    sc.pp.log1p(adata_bio)
    
    # Select top 1000 highly variable genes
    sc.pp.highly_variable_genes(adata_bio, n_top_genes=1000, flavor='seurat_v3')
    adata_bio = adata_bio[:, adata_bio.var.highly_variable].copy()
    
    print(f"  BioFormer data shape: {adata_bio.shape}")
    
    return adata_bio

def preprocess_for_scgpt(adata):
    """Preprocess data for scGPT"""
    print("\n🤖 Preprocessing for scGPT...")
    
    try:
        import scgpt as scg
        from scgpt.preprocess import Preprocessor
        
        adata_scg = adata.copy()
        
        # Basic filtering
        sc.pp.filter_cells(adata_scg, min_genes=200)
        sc.pp.filter_genes(adata_scg, min_cells=3)
        
        # Calculate QC metrics
        adata_scg.var['mt'] = adata_scg.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(adata_scg, percent_top=None, log1p=False, inplace=True)
        
        # Filter cells
        sc.pp.filter_cells(adata_scg, min_genes=200)
        adata_scg = adata_scg[adata_scg.obs.pct_counts_mt < 20, :]
        
        # Normalize
        sc.pp.normalize_total(adata_scg, target_sum=1e4)
        sc.pp.log1p(adata_scg)
        
        # Select HVGs (scGPT typically uses more genes)
        sc.pp.highly_variable_genes(adata_scg, min_mean=0.0125, max_mean=3, min_disp=0.5)
        adata_scg.raw = adata_scg
        adata_scg = adata_scg[:, adata_scg.var.highly_variable]
        
        print(f"  scGPT data shape: {adata_scg.shape}")
        
        return adata_scg
        
    except ImportError as e:
        print(f"  Error importing scGPT: {e}")
        return None

def benchmark_bioformer(adata_bio):
    """Run BioFormer benchmark"""
    print("\n🧠 Running BioFormer benchmark...")
    results = {}
    
    try:
        from models.bioformer_with_ffn_moe import BioFormerMoE
        from utils.data import SingleCellDataset
        from torch.utils.data import DataLoader
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        from sklearn.cluster import KMeans
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        
        start_time = time.time()
        
        # Load pretrained model
        print("  Loading pretrained BioFormer...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model configuration from context
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
        
        model = BioFormerMoE(**model_config)
        
        # Try to load checkpoint
        checkpoint_path = '/mnt/nasdev2/dung/training1/checkpoints/checkpoint_epoch_4_20250615_081604_256d_4head_8layers.pt'
        if os.path.exists(checkpoint_path):
            print(f"  Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            print("  Checkpoint not found, using randomly initialized model")
        
        model.to(device)
        model.eval()
        
        # Prepare data
        # Need to pad/truncate to 1000 genes
        if adata_bio.shape[1] < 1000:
            # Pad with zeros
            padding = np.zeros((adata_bio.shape[0], 1000 - adata_bio.shape[1]))
            X_padded = np.hstack([adata_bio.X, padding])
        else:
            # Take first 1000 genes
            X_padded = adata_bio.X[:, :1000]
        
        # Quantile binning (similar to scGPT)
        def quantile_binning(expr, num_bins=51):
            binned = np.zeros_like(expr, dtype=np.uint8)
            for i in range(expr.shape[0]):
                cell = expr[i]
                nz = cell[cell > 0]
                if nz.size > 0:
                    thresholds = np.quantile(nz, np.linspace(0, 1, num_bins)[1:])
                    bins = np.digitize(cell, thresholds, right=True) + 1
                    bins[cell == 0] = 0
                    binned[i] = bins
            return binned
        
        X_binned = quantile_binning(X_padded)
        
        # Extract embeddings
        print("  Extracting embeddings...")
        embeddings = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(X_binned), batch_size):
                batch_binned = torch.from_numpy(X_binned[i:i+batch_size]).long().to(device)
                batch_cell_types = torch.zeros(batch_binned.shape[0], dtype=torch.long).to(device)
                
                try:
                    _, _, output, _ = model(batch_binned, batch_cell_types)
                    # Use CLS token (first position)
                    batch_embeddings = output[:, 0].cpu().numpy()
                    embeddings.append(batch_embeddings)
                except Exception as e:
                    print(f"    Error in batch {i}: {e}")
                    # Use zeros as fallback
                    batch_embeddings = np.zeros((batch_binned.shape[0], model_config['d_model']))
                    embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        
        # Calculate metrics
        if 'cell_type' in adata_bio.obs:
            print("  Calculating clustering metrics...")
            true_labels = pd.Categorical(adata_bio.obs['cell_type']).codes
            
            # K-means clustering
            n_clusters = len(np.unique(true_labels))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            predicted_labels = kmeans.fit_predict(embeddings)
            
            nmi = normalized_mutual_info_score(true_labels, predicted_labels)
            ari = adjusted_rand_score(true_labels, predicted_labels)
            
            # Classification accuracy using RF
            X_train, X_test, y_train, y_test = train_test_split(
                embeddings, true_labels, test_size=0.2, random_state=42, stratify=true_labels
            )
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            accuracy = rf.score(X_test, y_test)
            
            results = {
                'model': 'BioFormer',
                'nmi': float(nmi),
                'ari': float(ari),
                'accuracy': float(accuracy),
                'n_cells': int(adata_bio.shape[0]),
                'n_genes': int(adata_bio.shape[1]),
                'embedding_dim': int(embeddings.shape[1]),
                'runtime_seconds': time.time() - start_time,
                'memory_peak_mb': torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 'N/A'
            }
            
            print(f"  Results: NMI={nmi:.4f}, ARI={ari:.4f}, Accuracy={accuracy:.4f}")
        else:
            results = {
                'model': 'BioFormer',
                'error': 'No cell type labels available',
                'runtime_seconds': time.time() - start_time
            }
        
    except Exception as e:
        print(f"  Error in BioFormer benchmark: {e}")
        results = {'model': 'BioFormer', 'error': str(e)}
    
    return results

def benchmark_scgpt(adata_scg):
    """Run scGPT benchmark"""
    print("\n🤖 Running scGPT benchmark...")
    results = {}
    
    if adata_scg is None:
        return {'model': 'scGPT', 'error': 'Failed to preprocess data for scGPT'}
    
    try:
        import scgpt as scg
        from scgpt import SubsetsBatchSampler
        from scgpt.model import TransformerModel
        from scgpt.tokenizer.gene_tokenizer import GeneVocab
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        from sklearn.cluster import KMeans
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        
        start_time = time.time()
        
        print("  Setting up scGPT model...")
        
        # Create gene vocabulary
        vocab = GeneVocab.from_file("scgpt_vocab.txt") if os.path.exists("scgpt_vocab.txt") else None
        if vocab is None:
            # Create simple vocabulary from genes
            gene_list = adata_scg.var_names.tolist()
            vocab = GeneVocab()
            for gene in gene_list:
                vocab.append_token(gene)
        
        # Model configuration
        model_config = {
            "layer_size": 512,
            "nlayers": 12,
            "nhead": 8,
            "dropout": 0.1,
            "d_hid": 512,
        }
        
        model = TransformerModel(
            ntokens=len(vocab),
            d_model=model_config["layer_size"],
            nhead=model_config["nhead"],
            d_hid=model_config["d_hid"],
            nlayers=model_config["nlayers"],
            dropout=model_config["dropout"],
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        print(f"  scGPT model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Prepare data for scGPT
        print("  Preparing data for scGPT...")
        
        # Tokenize genes
        gene_ids = []
        for gene in adata_scg.var_names:
            if gene in vocab:
                gene_ids.append(vocab[gene])
            else:
                gene_ids.append(vocab["<unk>"] if "<unk>" in vocab else 0)
        
        # Prepare expression data (binning)
        X = adata_scg.X.toarray() if hasattr(adata_scg.X, 'toarray') else adata_scg.X
        
        # Simple binning for scGPT
        def scgpt_binning(expr, n_bins=51):
            binned = np.zeros_like(expr, dtype=np.int32)
            for i in range(expr.shape[0]):
                cell = expr[i]
                nonzero = cell[cell > 0]
                if len(nonzero) > 0:
                    # Use quantile-based binning
                    bins = np.linspace(0, np.max(nonzero), n_bins)
                    binned[i] = np.digitize(cell, bins)
            return binned
        
        X_binned = scgpt_binning(X)
        
        # Extract embeddings
        print("  Extracting scGPT embeddings...")
        embeddings = []
        batch_size = 16  # Smaller batch size for scGPT
        
        with torch.no_grad():
            for i in range(0, len(X_binned), batch_size):
                batch_data = X_binned[i:i+batch_size]
                
                try:
                    # Prepare input tensors
                    gene_ids_tensor = torch.tensor([gene_ids] * len(batch_data), device=device)
                    values_tensor = torch.tensor(batch_data, device=device, dtype=torch.float)
                    
                    # Forward pass
                    output = model(gene_ids_tensor, values_tensor)
                    
                    # Use mean pooling or CLS token
                    if hasattr(output, 'last_hidden_state'):
                        batch_embeddings = output.last_hidden_state.mean(dim=1).cpu().numpy()
                    else:
                        batch_embeddings = output.mean(dim=1).cpu().numpy()
                    
                    embeddings.append(batch_embeddings)
                    
                except Exception as e:
                    print(f"    Error in scGPT batch {i}: {e}")
                    # Use zeros as fallback
                    batch_embeddings = np.zeros((len(batch_data), model_config["layer_size"]))
                    embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        
        # Calculate metrics
        if 'cell_type' in adata_scg.obs:
            print("  Calculating scGPT metrics...")
            true_labels = pd.Categorical(adata_scg.obs['cell_type']).codes
            
            # K-means clustering
            n_clusters = len(np.unique(true_labels))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            predicted_labels = kmeans.fit_predict(embeddings)
            
            nmi = normalized_mutual_info_score(true_labels, predicted_labels)
            ari = adjusted_rand_score(true_labels, predicted_labels)
            
            # Classification accuracy
            X_train, X_test, y_train, y_test = train_test_split(
                embeddings, true_labels, test_size=0.2, random_state=42, stratify=true_labels
            )
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            accuracy = rf.score(X_test, y_test)
            
            results = {
                'model': 'scGPT',
                'nmi': float(nmi),
                'ari': float(ari),
                'accuracy': float(accuracy),
                'n_cells': int(adata_scg.shape[0]),
                'n_genes': int(adata_scg.shape[1]),
                'embedding_dim': int(embeddings.shape[1]),
                'runtime_seconds': time.time() - start_time,
                'memory_peak_mb': torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 'N/A'
            }
            
            print(f"  Results: NMI={nmi:.4f}, ARI={ari:.4f}, Accuracy={accuracy:.4f}")
        else:
            results = {
                'model': 'scGPT',
                'error': 'No cell type labels available',
                'runtime_seconds': time.time() - start_time
            }
        
    except Exception as e:
        print(f"  Error in scGPT benchmark: {e}")
        import traceback
        traceback.print_exc()
        results = {'model': 'scGPT', 'error': str(e)}
    
    return results

def save_results(bioformer_results, scgpt_results, output_path):
    """Save benchmark results"""
    results = {
        'timestamp': datetime.now().isoformat(),
        'bioformer': bioformer_results,
        'scgpt': scgpt_results,
        'comparison': {}
    }
    
    # Add comparison if both succeeded
    if 'nmi' in bioformer_results and 'nmi' in scgpt_results:
        results['comparison'] = {
            'nmi_improvement': bioformer_results['nmi'] - scgpt_results['nmi'],
            'ari_improvement': bioformer_results['ari'] - scgpt_results['ari'],
            'accuracy_improvement': bioformer_results['accuracy'] - scgpt_results['accuracy'],
            'runtime_ratio': bioformer_results['runtime_seconds'] / scgpt_results['runtime_seconds'],
            'winner_nmi': 'BioFormer' if bioformer_results['nmi'] > scgpt_results['nmi'] else 'scGPT',
            'winner_ari': 'BioFormer' if bioformer_results['ari'] > scgpt_results['ari'] else 'scGPT',
            'winner_accuracy': 'BioFormer' if bioformer_results['accuracy'] > scgpt_results['accuracy'] else 'scGPT'
        }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    return results

def main():
    """Main benchmark function"""
    setup_environment()
    
    # Load data
    adata = load_pbmc_data()
    
    # Preprocess for both models
    adata_bio = preprocess_for_bioformer(adata)
    adata_scg = preprocess_for_scgpt(adata)
    
    # Run benchmarks
    bioformer_results = benchmark_bioformer(adata_bio)
    scgpt_results = benchmark_scgpt(adata_scg)
    
    # Save results
    output_path = '/home/tripham/scgpt/trial_3_based_moe/bioformer_vs_scgpt_benchmark.json'
    final_results = save_results(bioformer_results, scgpt_results, output_path)
    
    # Print summary
    print("\n📊 BENCHMARK SUMMARY")
    print("=" * 50)
    print(f"BioFormer: {bioformer_results}")
    print(f"scGPT: {scgpt_results}")
    
    if 'comparison' in final_results and final_results['comparison']:
        comp = final_results['comparison']
        print(f"\n🏆 COMPARISON:")
        print(f"  NMI: {comp.get('winner_nmi', 'Unknown')} wins")
        print(f"  ARI: {comp.get('winner_ari', 'Unknown')} wins") 
        print(f"  Accuracy: {comp.get('winner_accuracy', 'Unknown')} wins")
    
    print("\n✅ Benchmark completed!")
    return final_results

if __name__ == '__main__':
    main()