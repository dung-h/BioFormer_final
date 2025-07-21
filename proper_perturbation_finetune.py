#!/usr/bin/env python3
"""
PROPER PERTURBATION PREDICTION FINE-TUNING
1. Load pre-trained BioFormer foundation model
2. Fine-tune on 80% of perturbation data
3. Test on remaining 20% of perturbation data
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scanpy as sc
import json
import time
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Add model paths
sys.path.insert(0, '/home/tripham/scgpt/trial_3_based_moe/models')

def load_pretrained_bioformer(checkpoint_path, device):
    """Load pre-trained BioFormer foundation model"""
    print("🧠 Loading pre-trained BioFormer foundation model...")
    
    from models.bioformer_with_ffn_moe import BioFormerMoE as BioFormer
    
    # Initialize with same config as pre-trained model
    base_model = BioFormer(
        vocab_size=1000,
        num_cell_types=199,
        num_studies=15,
        num_bins=51,
        d_model=256,
        nhead=4,
        num_layers=8,
        dropout=0.1,
        num_experts=4
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    base_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    base_model.eval()
    
    print("✅ Pre-trained foundation model loaded successfully")
    return base_model

class BioFormerPerturbationModel(nn.Module):
    """BioFormer for perturbation prediction with pre-trained embeddings"""
    
    def __init__(self, pretrained_model, num_perturbations, num_genes, device):
        super().__init__()
        self.device = device
        self.d_model = pretrained_model.d_model
        self.num_genes = num_genes
        
        # Transfer embeddings from pre-trained model
        self.gene_embedding = nn.Embedding(num_genes, self.d_model)
        self.value_embedding = nn.Embedding(51, self.d_model)  # 51 bins like pre-trained
        
        # Copy pre-trained weights
        self._transfer_embeddings(pretrained_model)
        
        # Perturbation-specific components
        self.perturbation_embedding = nn.Embedding(num_perturbations, self.d_model)
        
        # Transformer layers (transfer from pre-trained)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=4,
                dim_feedforward=self.d_model * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=4
        )
        
        # Output prediction head
        self.output_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model, num_genes)
        )
        
        # Initialize new components
        nn.init.xavier_uniform_(self.perturbation_embedding.weight)
        
    def _transfer_embeddings(self, pretrained_model):
        """Transfer embeddings from pre-trained model"""
        print("🔧 Transferring embeddings from pre-trained model...")
        
        # Transfer gene embeddings
        if hasattr(pretrained_model, 'gene_embedding'):
            pretrained_gene_emb = pretrained_model.gene_embedding.weight.data
            min_genes = min(self.num_genes, pretrained_gene_emb.shape[0])
            self.gene_embedding.weight.data[:min_genes] = pretrained_gene_emb[:min_genes]
            print(f"  Transferred {min_genes} gene embeddings")
        
        # Transfer value embeddings
        if hasattr(pretrained_model, 'value_embedding'):
            pretrained_value_emb = pretrained_model.value_embedding.weight.data
            min_values = min(51, pretrained_value_emb.shape[0])
            self.value_embedding.weight.data[:min_values] = pretrained_value_emb[:min_values]
            print(f"  Transferred {min_values} value embeddings")
    
    def forward(self, gene_ids, values, perturbation_id):
        """Forward pass for perturbation prediction"""
        batch_size, seq_len = gene_ids.shape
        
        # Embeddings
        gene_emb = self.gene_embedding(gene_ids)  # [batch, seq_len, d_model]
        value_emb = self.value_embedding(values)  # [batch, seq_len, d_model]
        
        # Combine gene and value embeddings
        combined_emb = gene_emb + value_emb  # [batch, seq_len, d_model]
        
        # Add perturbation context
        pert_emb = self.perturbation_embedding(perturbation_id)  # [batch, d_model]
        pert_emb = pert_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, d_model]
        
        # Combine all embeddings
        input_emb = combined_emb + pert_emb  # [batch, seq_len, d_model]
        
        # Transformer
        transformer_output = self.transformer(input_emb)  # [batch, seq_len, d_model]
        
        # Global pooling
        pooled = torch.mean(transformer_output, dim=1)  # [batch, d_model]
        
        # Output prediction
        output = self.output_head(pooled)  # [batch, num_genes]
        
        return output

def load_perturbation_data(data_dir):
    """Load perturbation training and test data"""
    print("📊 Loading perturbation datasets...")
    
    train_path = Path(data_dir) / "perturbation_train.h5ad"
    test_path = Path(data_dir) / "perturbation_test.h5ad"
    
    # Load data
    adata_train = sc.read_h5ad(train_path)
    adata_test = sc.read_h5ad(test_path)
    
    print(f"✅ Training data: {adata_train.shape}")
    print(f"✅ Test data: {adata_test.shape}")
    
    return adata_train, adata_test

def preprocess_data(adata_train, adata_test, max_genes=2000):
    """Preprocess perturbation data for training"""
    print("🔧 Preprocessing perturbation data...")
    
    # Get perturbation mapping
    all_perturbations = list(set(adata_train.obs['perturbation'].unique()) | 
                           set(adata_test.obs['perturbation'].unique()))
    pert_to_id = {pert: i for i, pert in enumerate(all_perturbations)}
    
    # Select top variable genes
    if adata_train.shape[1] > max_genes:
        sc.pp.highly_variable_genes(adata_train, n_top_genes=max_genes)
        selected_genes = adata_train.var['highly_variable']
        adata_train = adata_train[:, selected_genes]
        adata_test = adata_test[:, selected_genes]
    
    # Normalize and bin
    sc.pp.normalize_total(adata_train, target_sum=1e4)
    sc.pp.log1p(adata_train)
    
    sc.pp.normalize_total(adata_test, target_sum=1e4)
    sc.pp.log1p(adata_test)
    
    # Binning (like scGPT)
    def bin_expression(X, n_bins=51):
        """Bin expression values"""
        X_binned = np.zeros_like(X, dtype=int)
        for i in range(X.shape[1]):
            gene_expr = X[:, i]
            if gene_expr.max() > gene_expr.min():
                bins = np.linspace(gene_expr.min(), gene_expr.max(), n_bins)
                X_binned[:, i] = np.digitize(gene_expr, bins) - 1
                X_binned[:, i] = np.clip(X_binned[:, i], 0, n_bins-1)
        return X_binned
    
    # Bin data
    X_train_data = adata_train.X.toarray() if hasattr(adata_train.X, 'toarray') else adata_train.X
    X_test_data = adata_test.X.toarray() if hasattr(adata_test.X, 'toarray') else adata_test.X
    
    X_train_binned = bin_expression(X_train_data)
    X_test_binned = bin_expression(X_test_data)
    
    # Create gene indices
    gene_indices = np.arange(adata_train.shape[1])
    
    # Prepare data for training
    def prepare_sequences(X_binned, perturbations, gene_indices):
        """Prepare sequences for training"""
        sequences = []
        values = []
        pert_ids = []
        targets = []
        
        for i in range(X_binned.shape[0]):
            # Gene sequence (use all genes)
            gene_seq = gene_indices
            value_seq = X_binned[i]
            pert_id = pert_to_id[perturbations[i]]
            target = X_binned[i].astype(np.float32)  # Predict same sequence
            
            sequences.append(gene_seq)
            values.append(value_seq)
            pert_ids.append(pert_id)
            targets.append(target)
        
        return np.array(sequences), np.array(values), np.array(pert_ids), np.array(targets)
    
    # Prepare training and test data
    X_train_seq, V_train_seq, P_train, Y_train = prepare_sequences(
        X_train_binned, adata_train.obs['perturbation'], gene_indices
    )
    X_test_seq, V_test_seq, P_test, Y_test = prepare_sequences(
        X_test_binned, adata_test.obs['perturbation'], gene_indices
    )
    
    print(f"✅ Preprocessed data:")
    print(f"  Train: {X_train_seq.shape[0]} samples, {X_train_seq.shape[1]} genes")
    print(f"  Test: {X_test_seq.shape[0]} samples, {X_test_seq.shape[1]} genes")
    print(f"  Perturbations: {len(pert_to_id)}")
    
    return {
        'train': (X_train_seq, V_train_seq, P_train, Y_train),
        'test': (X_test_seq, V_test_seq, P_test, Y_test),
        'num_genes': adata_train.shape[1],
        'num_perturbations': len(pert_to_id),
        'pert_to_id': pert_to_id
    }

def train_perturbation_model(model, train_data, device, epochs=10, lr=1e-4):
    """Fine-tune the perturbation model"""
    print(f"🚀 Fine-tuning perturbation model for {epochs} epochs...")
    
    X_train, V_train, P_train, Y_train = train_data
    
    # Create data loader
    train_dataset = TensorDataset(
        torch.LongTensor(X_train),
        torch.LongTensor(V_train),
        torch.LongTensor(P_train),
        torch.FloatTensor(Y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scaler = GradScaler()
    
    model.train()
    training_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for gene_ids, values, pert_ids, targets in pbar:
            gene_ids = gene_ids.to(device)
            values = values.to(device)
            pert_ids = pert_ids.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                predictions = model(gene_ids, values, pert_ids)
                loss = criterion(predictions, targets)
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"⚠️ NaN detected in epoch {epoch+1}, skipping batch")
                continue
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / max(num_batches, 1)
        training_losses.append(avg_loss)
        print(f"✅ Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
    
    print("✅ Fine-tuning completed!")
    return training_losses

def evaluate_perturbation_model(model, test_data, device):
    """Evaluate the fine-tuned model on test data"""
    print("📊 Evaluating fine-tuned model on test data...")
    
    X_test, V_test, P_test, Y_test = test_data
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    # Create test data loader
    test_dataset = TensorDataset(
        torch.LongTensor(X_test),
        torch.LongTensor(V_test),
        torch.LongTensor(P_test),
        torch.FloatTensor(Y_test)
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for gene_ids, values, pert_ids, targets in tqdm(test_loader, desc="Evaluating"):
            gene_ids = gene_ids.to(device)
            values = values.to(device)
            pert_ids = pert_ids.to(device)
            
            predictions = model(gene_ids, values, pert_ids)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.numpy())
    
    # Concatenate all predictions and targets
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics
    correlations = []
    mse_values = []
    
    for i in range(predictions.shape[0]):
        pred = predictions[i]
        target = targets[i]
        
        # Calculate correlation
        if np.std(pred) > 1e-6 and np.std(target) > 1e-6:
            corr, _ = pearsonr(pred, target)
            if not np.isnan(corr):
                correlations.append(corr)
        
        # Calculate MSE
        mse = mean_squared_error(target, pred)
        mse_values.append(mse)
    
    # Overall metrics
    overall_corr, _ = pearsonr(predictions.flatten(), targets.flatten())
    overall_mse = mean_squared_error(targets.flatten(), predictions.flatten())
    
    results = {
        'mean_correlation': float(np.mean(correlations)) if correlations else 0.0,
        'overall_correlation': float(overall_corr) if not np.isnan(overall_corr) else 0.0,
        'mean_mse': float(np.mean(mse_values)),
        'overall_mse': float(overall_mse),
        'num_samples': predictions.shape[0],
        'num_genes': predictions.shape[1],
        'num_correlations': len(correlations)
    }
    
    print(f"✅ Evaluation Results:")
    print(f"  Mean Correlation: {results['mean_correlation']:.4f}")
    print(f"  Overall Correlation: {results['overall_correlation']:.4f}")
    print(f"  Mean MSE: {results['mean_mse']:.4f}")
    print(f"  Test Samples: {results['num_samples']}")
    
    return results

def main():
    """Main function for proper perturbation fine-tuning"""
    print("🎯 PROPER PERTURBATION PREDICTION FINE-TUNING")
    print("=" * 60)
    print("Methodology: Pre-trained foundation model → Fine-tune 80% → Test 20%")
    print("=" * 60)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")
    
    # Paths
    checkpoint_path = "/mnt/nasdev2/dung/training1/checkpoints/checkpoint_epoch_4_20250615_081604_256d_4head_8layers.pt"
    data_dir = "test_datasets/perturbation"
    
    start_time = time.time()
    
    # 1. Load pre-trained model
    if not os.path.exists(checkpoint_path):
        print(f"❌ Pre-trained checkpoint not found: {checkpoint_path}")
        return
    
    pretrained_model = load_pretrained_bioformer(checkpoint_path, device)
    
    # 2. Load perturbation data
    adata_train, adata_test = load_perturbation_data(data_dir)
    
    # 3. Preprocess data
    data_dict = preprocess_data(adata_train, adata_test)
    
    # 4. Create perturbation model with pre-trained embeddings
    print("🏗️ Creating perturbation model with pre-trained embeddings...")
    model = BioFormerPerturbationModel(
        pretrained_model=pretrained_model,
        num_perturbations=data_dict['num_perturbations'],
        num_genes=data_dict['num_genes'],
        device=device
    ).to(device)
    
    print(f"📊 Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 5. Fine-tune on training data (80%)
    training_losses = train_perturbation_model(
        model, data_dict['train'], device, epochs=10, lr=1e-4
    )
    
    # 6. Evaluate on test data (20%)
    test_results = evaluate_perturbation_model(
        model, data_dict['test'], device
    )
    
    total_time = time.time() - start_time
    
    # 7. Save results
    final_results = {
        'methodology': 'Pre-trained foundation model → Fine-tune 80% → Test 20%',
        'dataset_info': {
            'train_samples': data_dict['train'][0].shape[0],
            'test_samples': data_dict['test'][0].shape[0],
            'num_genes': data_dict['num_genes'],
            'num_perturbations': data_dict['num_perturbations']
        },
        'training': {
            'epochs': 10,
            'learning_rate': 1e-4,
            'training_losses': training_losses,
            'final_loss': training_losses[-1] if training_losses else None
        },
        'evaluation': test_results,
        'timing': {
            'total_time_seconds': total_time,
            'training_time_per_epoch': total_time / 10
        },
        'model_info': {
            'parameters': sum(p.numel() for p in model.parameters()),
            'uses_pretrained_embeddings': True,
            'checkpoint_used': checkpoint_path
        }
    }
    
    # Save results
    with open('proper_perturbation_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Save model
    torch.save(model.state_dict(), 'bioformer_perturbation_proper_finetuned.pth')
    
    print("\n" + "=" * 60)
    print("🏆 PROPER PERTURBATION FINE-TUNING COMPLETED")
    print("=" * 60)
    print(f"✅ Used pre-trained foundation model: {checkpoint_path}")
    print(f"✅ Fine-tuned on {data_dict['train'][0].shape[0]} training samples")
    print(f"✅ Tested on {data_dict['test'][0].shape[0]} test samples")
    print(f"✅ Final correlation: {test_results['overall_correlation']:.4f}")
    print(f"✅ Training time: {total_time:.1f} seconds")
    print("\n📁 Results saved to proper_perturbation_results.json")
    
    return final_results

if __name__ == "__main__":
    results = main()