#!/usr/bin/env python3
"""
Test FFN BioFormer model on cell type annotation datasets
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scanpy as sc
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import warnings
warnings.filterwarnings('ignore')

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

    def forward(self, gene_ids, values, cell_types=None):
        gene_emb = self.gene_embedding(gene_ids)
        value_emb = self.value_embedding(values)
        
        x = gene_emb + value_emb
        x = self.norm(x)
        
        x = self.transformer(x)
        
        return x

def load_model_and_vocab():
    """Load the FFN model and vocabulary"""
    print("Loading FFN model and vocabulary...")
    
    # Load vocabulary
    vocab_file = "/mnt/nasdev2/dung/preprocessed/selected_genes.txt"
    with open(vocab_file, 'r') as f:
        genes = [line.strip() for line in f.readlines()]
    
    # Create simple vocabulary mapping
    vocab = {gene: i for i, gene in enumerate(genes)}
    vocab["<pad>"] = len(genes)
    
    # Load model
    model = scGPT(
        vocab_size=len(vocab),
        num_cell_types=50,  # Placeholder
        d_model=512,
        nhead=8,
        num_layers=8,
        dropout=0.1
    )
    
    # Load checkpoint
    checkpoint_path = "/mnt/nasdev2/dung/training_bioformer/best_model.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # Handle potential state dict key mismatch
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except:
            try:
                model.load_state_dict(checkpoint)
            except:
                print("Warning: Could not load checkpoint weights")
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
    
    model.eval()
    return model, vocab, genes

def preprocess_for_model(adata, vocab, genes):
    """Preprocess data for model input"""
    # Ensure genes are in the correct order
    gene_order = [g for g in genes if g in adata.var_names]
    adata = adata[:, gene_order]
    
    # Quantile binning preprocessing
    def quantile_binning(x, n_bins=51):
        """Quantile binning for gene expression values"""
        if x.sum() == 0:
            return np.zeros_like(x, dtype=int)
        
        # Log1p transformation
        x = np.log1p(x)
        
        # Quantile binning
        quantiles = np.linspace(0, 1, n_bins)
        bin_edges = np.quantile(x[x > 0], quantiles) if (x > 0).sum() > 0 else np.array([0])
        
        # Ensure unique edges
        bin_edges = np.unique(bin_edges)
        
        if len(bin_edges) < 2:
            return np.zeros_like(x, dtype=int)
        
        # Bin the data
        binned = np.digitize(x, bin_edges, right=True)
        binned = np.clip(binned, 0, n_bins - 1)
        
        return binned
    
    # Apply preprocessing
    X_processed = np.zeros((adata.n_obs, len(gene_order)), dtype=int)
    
    for i in range(adata.n_obs):
        x = adata.X[i, :].toarray().flatten() if hasattr(adata.X, 'toarray') else adata.X[i, :]
        X_processed[i, :] = quantile_binning(x)
    
    # Convert to model input format
    input_gene_ids = []
    input_values = []
    
    for i in range(adata.n_obs):
        gene_ids = []
        values = []
        
        for j, gene in enumerate(gene_order):
            if gene in vocab:
                gene_ids.append(vocab[gene])
                values.append(X_processed[i, j])
        
        input_gene_ids.append(gene_ids)
        input_values.append(values)
    
    return input_gene_ids, input_values

def extract_embeddings(model, input_gene_ids, input_values, batch_size=32):
    """Extract embeddings using the model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    all_embeddings = []
    
    for i in range(0, len(input_gene_ids), batch_size):
        batch_gene_ids = input_gene_ids[i:i+batch_size]
        batch_values = input_values[i:i+batch_size]
        
        # Pad sequences
        max_len = max(len(seq) for seq in batch_gene_ids)
        
        padded_gene_ids = []
        padded_values = []
        
        for gene_ids, values in zip(batch_gene_ids, batch_values):
            padded_gene_ids.append(gene_ids + [0] * (max_len - len(gene_ids)))
            padded_values.append(values + [0] * (max_len - len(values)))
        
        # Convert to tensors
        gene_ids_tensor = torch.tensor(padded_gene_ids, dtype=torch.long).to(device)
        values_tensor = torch.tensor(padded_values, dtype=torch.float).to(device)
        
        with torch.no_grad():
            # Get embeddings
            output = model(gene_ids_tensor, values_tensor)
            # Use mean pooling of all gene embeddings
            embeddings = output.mean(dim=1)
            all_embeddings.append(embeddings.cpu())
    
    return torch.cat(all_embeddings, dim=0).numpy()

def evaluate_cell_type_annotation(train_adata, test_adata, model, vocab, genes, dataset_name):
    """Evaluate cell type annotation performance"""
    print(f"\n=== Evaluating {dataset_name} ===")
    
    # Preprocess data
    print("Preprocessing training data...")
    train_gene_ids, train_values = preprocess_for_model(train_adata, vocab, genes)
    print("Preprocessing test data...")
    test_gene_ids, test_values = preprocess_for_model(test_adata, vocab, genes)
    
    # Extract embeddings
    print("Extracting training embeddings...")
    train_embeddings = extract_embeddings(model, train_gene_ids, train_values)
    print("Extracting test embeddings...")
    test_embeddings = extract_embeddings(model, test_gene_ids, test_values)
    
    # Prepare labels - find the cell type column
    cell_type_cols = ['celltype', 'cell_type', 'Celltype']
    train_cell_type_col = None
    test_cell_type_col = None
    
    for col in cell_type_cols:
        if col in train_adata.obs.columns:
            train_cell_type_col = col
            break
    
    for col in cell_type_cols:
        if col in test_adata.obs.columns:
            test_cell_type_col = col
            break
    
    if train_cell_type_col is None or test_cell_type_col is None:
        raise ValueError("Could not find cell type column in data")
    
    train_labels = train_adata.obs[train_cell_type_col].values
    test_labels = test_adata.obs[test_cell_type_col].values
    
    # Encode labels
    le = LabelEncoder()
    train_labels_encoded = le.fit_transform(train_labels)
    test_labels_encoded = le.transform(test_labels)
    
    # Train kNN classifier
    print("Training kNN classifier...")
    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    knn.fit(train_embeddings, train_labels_encoded)
    
    # Predict
    pred_labels = knn.predict(test_embeddings)
    
    # Calculate metrics
    accuracy = accuracy_score(test_labels_encoded, pred_labels)
    f1_macro = f1_score(test_labels_encoded, pred_labels, average='macro')
    f1_weighted = f1_score(test_labels_encoded, pred_labels, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 (macro): {f1_macro:.4f}")
    print(f"F1 (weighted): {f1_weighted:.4f}")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(test_labels_encoded, pred_labels, target_names=le.classes_))
    
    # Create UMAP visualization
    print("Creating UMAP visualization...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    combined_embeddings = np.vstack([train_embeddings, test_embeddings])
    umap_embeddings = reducer.fit_transform(combined_embeddings)
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    # Training data
    plt.subplot(1, 2, 1)
    train_umap = umap_embeddings[:len(train_embeddings)]
    scatter = plt.scatter(train_umap[:, 0], train_umap[:, 1], 
                         c=train_labels_encoded, cmap='tab20', s=1, alpha=0.7)
    plt.title(f'{dataset_name} - Training Data')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    
    # Test data
    plt.subplot(1, 2, 2)
    test_umap = umap_embeddings[len(train_embeddings):]
    scatter = plt.scatter(test_umap[:, 0], test_umap[:, 1], 
                         c=test_labels_encoded, cmap='tab20', s=1, alpha=0.7)
    plt.title(f'{dataset_name} - Test Data')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    
    plt.tight_layout()
    plt.savefig(f'/home/tripham/scgpt/trial_3_based_moe/results_ffn_{dataset_name.lower()}_cell_annotation.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'n_train': len(train_embeddings),
        'n_test': len(test_embeddings),
        'n_cell_types': len(le.classes_)
    }

def main():
    """Main function"""
    print("=== FFN BioFormer Cell Type Annotation Testing ===")
    
    # Set random seed
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load model
    model, vocab, genes = load_model_and_vocab()
    
    # Define datasets
    datasets = [
        {
            'name': 'MS',
            'train': '/home/tripham/scgpt/trial_3_based_moe/data/external_datasets/MS/filtered_ms_adata.h5ad',
            'test': '/home/tripham/scgpt/trial_3_based_moe/data/external_datasets/MS/filtered_ms_adata.h5ad',  # Same file for train/test
            'cell_type_col': 'celltype'
        },
        {
            'name': 'Myeloid',
            'train': '/home/tripham/scgpt/trial_3_based_moe/data/external_datasets/Myeloid/filtered_reference_adata.h5ad',
            'test': '/home/tripham/scgpt/trial_3_based_moe/data/external_datasets/Myeloid/filtered_query_adata.h5ad',
            'cell_type_col': 'cell_type'
        },
        {
            'name': 'hPancreas',
            'train': '/home/tripham/scgpt/trial_3_based_moe/data/external_datasets/hPancreas/filtered_demo_train.h5ad',
            'test': '/home/tripham/scgpt/trial_3_based_moe/data/external_datasets/hPancreas/filtered_demo_test.h5ad',
            'cell_type_col': 'Celltype'
        }
    ]
    
    # Results storage
    results = {}
    
    # Test each dataset
    for dataset in datasets:
        try:
            print(f"\n{'='*50}")
            print(f"Testing {dataset['name']}")
            print(f"{'='*50}")
            
            # Load data
            train_adata = sc.read_h5ad(dataset['train'])
            test_adata = sc.read_h5ad(dataset['test'])
            
            # For MS dataset, split into train/test
            if dataset['name'] == 'MS':
                n_train = int(0.8 * train_adata.n_obs)
                indices = np.random.permutation(train_adata.n_obs)
                train_indices = indices[:n_train]
                test_indices = indices[n_train:]
                
                test_adata = train_adata[test_indices].copy()
                train_adata = train_adata[train_indices].copy()
            
            # Evaluate
            result = evaluate_cell_type_annotation(
                train_adata, test_adata, model, vocab, genes, dataset['name']
            )
            results[dataset['name']] = result
            
        except Exception as e:
            print(f"Error testing {dataset['name']}: {e}")
            continue
    
    # Print summary
    print(f"\n{'='*60}")
    print("FFN BioFormer Cell Type Annotation Results Summary")
    print(f"{'='*60}")
    print(f"{'Dataset':<12} {'Accuracy':<10} {'F1_macro':<10} {'F1_weighted':<12} {'N_train':<8} {'N_test':<8} {'N_types':<8}")
    print("-" * 60)
    
    for dataset_name, result in results.items():
        print(f"{dataset_name:<12} {result['accuracy']:<10.4f} {result['f1_macro']:<10.4f} "
              f"{result['f1_weighted']:<12.4f} {result['n_train']:<8} {result['n_test']:<8} {result['n_cell_types']:<8}")
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv('/home/tripham/scgpt/trial_3_based_moe/ffn_cell_annotation_results.csv')
    print(f"\nResults saved to ffn_cell_annotation_results.csv")

if __name__ == "__main__":
    main()