#!/usr/bin/env python3
"""
Simple FFN BioFormer Cell Type Annotation Test
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scanpy as sc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import warnings
warnings.filterwarnings('ignore')

class SimpleFFN(nn.Module):
    """Simple FFN model for cell type annotation"""
    def __init__(self, vocab_size, d_model=256, num_layers=4):
        super(SimpleFFN, self).__init__()
        self.d_model = d_model
        
        self.gene_embedding = nn.Embedding(vocab_size, d_model)
        self.value_embedding = nn.Embedding(51, d_model)
        
        encoder_layer = TransformerEncoderLayer(d_model, nhead=8, dim_feedforward=d_model*2, dropout=0.1, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        
        self.norm = nn.LayerNorm(d_model)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.gene_embedding.weight)
        nn.init.xavier_uniform_(self.value_embedding.weight)

    def forward(self, gene_ids, values):
        gene_emb = self.gene_embedding(gene_ids)
        value_emb = self.value_embedding(values)
        
        x = gene_emb + value_emb
        x = self.norm(x)
        x = self.transformer(x)
        
        return x.mean(dim=1)  # Mean pooling

def load_model_and_vocab():
    """Load model and vocabulary"""
    vocab_file = "/mnt/nasdev2/dung/preprocessed/selected_genes.txt"
    with open(vocab_file, 'r') as f:
        genes = [line.strip() for line in f.readlines()]
    
    vocab = {gene: i for i, gene in enumerate(genes)}
    
    model = SimpleFFN(vocab_size=len(vocab))
    model.eval()
    
    return model, vocab, genes

def preprocess_data(adata, vocab, genes):
    """Simple preprocessing"""
    # Filter genes
    gene_order = [g for g in genes if g in adata.var_names]
    adata = adata[:, gene_order]
    
    # Simple quantile binning
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    X = np.log1p(X)  # Log transform
    
    # Quantile binning per gene
    X_binned = np.zeros_like(X, dtype=int)
    for j in range(X.shape[1]):
        x = X[:, j]
        if x.max() > 0:
            bins = np.quantile(x[x > 0], np.linspace(0, 1, 51))
            bins = np.unique(bins)
            X_binned[:, j] = np.digitize(x, bins, right=True)
            X_binned[:, j] = np.clip(X_binned[:, j], 0, 50)
    
    # Convert to model input
    gene_ids = []
    values = []
    
    for i in range(adata.n_obs):
        row_genes = []
        row_values = []
        
        for j, gene in enumerate(gene_order):
            if gene in vocab:
                row_genes.append(vocab[gene])
                row_values.append(X_binned[i, j])
        
        gene_ids.append(row_genes)
        values.append(row_values)
    
    return gene_ids, values

def extract_embeddings_simple(model, gene_ids, values, batch_size=64):
    """Extract embeddings with simple batching"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    embeddings = []
    
    for i in range(0, len(gene_ids), batch_size):
        batch_genes = gene_ids[i:i+batch_size]
        batch_values = values[i:i+batch_size]
        
        # Pad to same length
        max_len = max(len(seq) for seq in batch_genes)
        
        padded_genes = [seq + [0] * (max_len - len(seq)) for seq in batch_genes]
        padded_values = [seq + [0] * (max_len - len(seq)) for seq in batch_values]
        
        genes_tensor = torch.tensor(padded_genes, dtype=torch.long).to(device)
        values_tensor = torch.tensor(padded_values, dtype=torch.long).to(device)
        
        with torch.no_grad():
            emb = model(genes_tensor, values_tensor)
            embeddings.append(emb.cpu().numpy())
    
    return np.vstack(embeddings)

def test_dataset(dataset_name, train_path, test_path, model, vocab, genes):
    """Test a single dataset"""
    print(f"\n=== Testing {dataset_name} ===")
    
    # Load data
    train_adata = sc.read_h5ad(train_path)
    test_adata = sc.read_h5ad(test_path)
    
    # For MS, split data
    if dataset_name == 'MS':
        n_train = int(0.8 * train_adata.n_obs)
        indices = np.random.permutation(train_adata.n_obs)
        train_adata = train_adata[indices[:n_train]]
        test_adata = test_adata[indices[n_train:]]
    
    print(f"Train: {train_adata.n_obs} cells, Test: {test_adata.n_obs} cells")
    
    # Preprocess
    train_genes, train_vals = preprocess_data(train_adata, vocab, genes)
    test_genes, test_vals = preprocess_data(test_adata, vocab, genes)
    
    # Extract embeddings
    train_emb = extract_embeddings_simple(model, train_genes, train_vals)
    test_emb = extract_embeddings_simple(model, test_genes, test_vals)
    
    # Get labels
    cell_type_cols = ['celltype', 'cell_type', 'Celltype']
    train_col = None
    test_col = None
    
    for col in cell_type_cols:
        if col in train_adata.obs.columns:
            train_col = col
            break
    
    for col in cell_type_cols:
        if col in test_adata.obs.columns:
            test_col = col
            break
    
    train_labels = train_adata.obs[train_col].values
    test_labels = test_adata.obs[test_col].values
    
    # Encode labels
    le = LabelEncoder()
    train_labels_enc = le.fit_transform(train_labels)
    test_labels_enc = le.transform(test_labels)
    
    # Train classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_emb, train_labels_enc)
    
    # Predict
    pred = knn.predict(test_emb)
    
    # Metrics
    acc = accuracy_score(test_labels_enc, pred)
    f1_macro = f1_score(test_labels_enc, pred, average='macro')
    f1_weighted = f1_score(test_labels_enc, pred, average='weighted')
    
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (macro): {f1_macro:.4f}")
    print(f"F1 (weighted): {f1_weighted:.4f}")
    
    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'n_train': len(train_emb),
        'n_test': len(test_emb),
        'n_cell_types': len(le.classes_)
    }

def main():
    """Main function"""
    # Setup logging
    import sys
    import datetime
    
    log_file = '/home/tripham/scgpt/trial_3_based_moe/ffn_cell_annotation_log.txt'
    
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(log_file)
    
    print(f"=== FFN BioFormer Cell Type Annotation Testing ===")
    print(f"Started at: {datetime.datetime.now()}")
    
    # Set seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load model
    model, vocab, genes = load_model_and_vocab()
    
    # Test datasets
    datasets = [
        ('MS', 
         '/home/tripham/scgpt/trial_3_based_moe/data/external_datasets/MS/filtered_ms_adata.h5ad',
         '/home/tripham/scgpt/trial_3_based_moe/data/external_datasets/MS/filtered_ms_adata.h5ad'),
        ('Myeloid',
         '/home/tripham/scgpt/trial_3_based_moe/data/external_datasets/Myeloid/filtered_reference_adata.h5ad',
         '/home/tripham/scgpt/trial_3_based_moe/data/external_datasets/Myeloid/filtered_query_adata.h5ad'),
        ('hPancreas',
         '/home/tripham/scgpt/trial_3_based_moe/data/external_datasets/hPancreas/filtered_demo_train.h5ad',
         '/home/tripham/scgpt/trial_3_based_moe/data/external_datasets/hPancreas/filtered_demo_test.h5ad')
    ]
    
    results = {}
    
    for name, train_path, test_path in datasets:
        try:
            result = test_dataset(name, train_path, test_path, model, vocab, genes)
            results[name] = result
        except Exception as e:
            print(f"Error testing {name}: {e}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("FFN BioFormer Cell Type Annotation Results")
    print(f"{'='*60}")
    print(f"{'Dataset':<10} {'Accuracy':<10} {'F1_macro':<10} {'F1_weighted':<12} {'N_train':<8} {'N_test':<8} {'N_types':<8}")
    print("-" * 60)
    
    for name, result in results.items():
        print(f"{name:<10} {result['accuracy']:<10.4f} {result['f1_macro']:<10.4f} "
              f"{result['f1_weighted']:<12.4f} {result['n_train']:<8} {result['n_test']:<8} {result['n_cell_types']:<8}")
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv('/home/tripham/scgpt/trial_3_based_moe/ffn_cell_annotation_results.csv')
    print(f"\nResults saved to ffn_cell_annotation_results.csv")
    print(f"Completed at: {datetime.datetime.now()}")

if __name__ == "__main__":
    main()