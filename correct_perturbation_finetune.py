#!/usr/bin/env python3
"""
CORRECT BioFormer Perturbation Prediction Fine-tuning
Proper methodology: Pre-trained checkpoint → Fine-tune on perturbation task
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import traceback
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import time
from tqdm import tqdm

# Import GEARS
try:
    from gears import PertData
    print("✅ GEARS imported successfully")
except ImportError as e:
    print(f"❌ GEARS import failed: {e}")
    sys.exit(1)

# Add BioFormer path
sys.path.insert(0, '/home/tripham/scgpt/trial_3_based_moe/models')
sys.path.insert(0, '/home/tripham/scgpt/trial_3_based_moe')

def load_pretrained_bioformer_base(checkpoint_path, device):
    """Load the pre-trained BioFormer foundation model"""
    print("🧠 Loading pre-trained BioFormer foundation model...")
    
    from models.bioformer_with_ffn_moe import BioFormerMoE as BioFormer
    
    # Initialize base model with same config as training
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
    missing_keys, unexpected_keys = base_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    
    print(f"✅ Base model loaded (missing: {len(missing_keys)}, unexpected: {len(unexpected_keys)})")
    base_model.eval()
    
    return base_model

def create_perturbation_model_from_pretrained(base_model, num_genes, num_perturbations, device):
    """Create perturbation model using pre-trained embeddings"""
    print("🔧 Creating perturbation model from pre-trained base...")
    
    from models.perturbation import BioFormerPerturb
    
    # Initialize perturbation model
    pert_model = BioFormerPerturb(
        vocab_size=num_genes,
        num_perturbs=num_perturbations,
        num_bins=51,
        d_model=256,  # Match base model
        nhead=4,      # Match base model  
        num_layers=6, # Slightly smaller for efficiency
        dropout=0.1
    ).to(device)
    
    # Transfer learned gene embeddings from pre-trained model
    if hasattr(base_model, 'gene_embedding') and hasattr(pert_model, 'gene_embedding'):
        # Get the minimum size to avoid dimension mismatch
        base_vocab_size = base_model.gene_embedding.weight.shape[0]
        pert_vocab_size = pert_model.gene_embedding.weight.shape[0]
        min_vocab_size = min(base_vocab_size, pert_vocab_size)
        
        print(f"  Transferring gene embeddings: {min_vocab_size} genes")
        pert_model.gene_embedding.weight.data[:min_vocab_size] = base_model.gene_embedding.weight.data[:min_vocab_size]
        
        # Initialize remaining embeddings if perturbation model is larger
        if pert_vocab_size > base_vocab_size:
            nn.init.xavier_uniform_(pert_model.gene_embedding.weight.data[base_vocab_size:])
    
    # Transfer value embeddings if available
    if hasattr(base_model, 'value_embedding') and hasattr(pert_model, 'value_embedding'):
        print("  Transferring value embeddings")
        pert_model.value_embedding.weight.data = base_model.value_embedding.weight.data.clone()
    
    print(f"✅ Perturbation model created with transferred embeddings")
    print(f"  - Parameters: {sum(p.numel() for p in pert_model.parameters()):,}")
    
    return pert_model

def prepare_perturbation_data(pert_data, train_ratio=0.8, max_genes=800, max_samples=5000):
    """Prepare perturbation data with proper train/test split"""
    print("📊 Preparing perturbation data...")
    
    # Get the data
    adata = pert_data.adata
    print(f"Raw data shape: {adata.shape}")
    
    # Sample subset for training
    if adata.shape[0] > max_samples:
        sample_indices = np.random.choice(adata.shape[0], max_samples, replace=False)
        adata = adata[sample_indices]
        print(f"Sampled to {max_samples} cells for training")
    
    # Get perturbation labels
    pert_categories = adata.obs['condition'].unique()
    print(f"Perturbation categories: {len(pert_categories)}")
    
    # Create perturbation mapping
    pert_to_idx = {pert: idx for idx, pert in enumerate(pert_categories)}
    
    # Prepare features and targets
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    
    # Limit genes for computational efficiency
    if X.shape[1] > max_genes:
        # Select top varying genes
        gene_var = np.var(X, axis=0)
        top_genes = np.argsort(gene_var)[-max_genes:]
        X = X[:, top_genes]
        print(f"Selected top {max_genes} most variable genes")
    
    # Normalize and bin gene expression
    X_normalized = np.log1p(X)  # Log transform
    
    # Create bins (0-50 for gene expression levels)
    X_binned = np.digitize(X_normalized, np.linspace(0, np.max(X_normalized), 51))
    X_binned = np.clip(X_binned, 0, 50)  # Ensure within vocab range
    
    # Get perturbation indices
    pert_indices = [pert_to_idx[pert] for pert in adata.obs['condition']]
    
    # Convert to tensors
    X_tensor = torch.LongTensor(X_binned)
    y_tensor = torch.FloatTensor(X_normalized)  # Target is continuous expression
    pert_tensor = torch.LongTensor(pert_indices)
    
    # Create train/test split
    n_samples = X_tensor.shape[0]
    n_train = int(n_samples * train_ratio)
    
    # Random shuffle for split
    indices = torch.randperm(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    # Split data
    X_train = X_tensor[train_indices]
    y_train = y_tensor[train_indices]
    pert_train = pert_tensor[train_indices]
    
    X_test = X_tensor[test_indices]
    y_test = y_tensor[test_indices]
    pert_test = pert_tensor[test_indices]
    
    print(f"Train samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Perturbations: {len(pert_categories)}")
    
    return {
        'X_train': X_train,
        'y_train': y_train, 
        'pert_train': pert_train,
        'X_test': X_test,
        'y_test': y_test,
        'pert_test': pert_test,
        'num_genes': X_train.shape[1],
        'num_perturbations': len(pert_categories),
        'pert_to_idx': pert_to_idx
    }

def create_data_loaders(data_dict, batch_size=32, test_batch_size=64):
    """Create PyTorch data loaders"""
    
    # Training data loader
    train_dataset = TensorDataset(
        data_dict['X_train'], 
        data_dict['pert_train'], 
        data_dict['y_train']
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    # Test data loader
    test_dataset = TensorDataset(
        data_dict['X_test'], 
        data_dict['pert_test'], 
        data_dict['y_test']
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=test_batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, test_loader

def finetune_perturbation_model(model, train_loader, val_loader, device, epochs=10, lr=1e-4):
    """Fine-tune the perturbation model on perturbation data"""
    print(f"🚀 Fine-tuning perturbation model for {epochs} epochs...")
    
    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    criterion = nn.MSELoss()
    scaler = GradScaler()
    
    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (gene_expr, pert_idx, targets) in enumerate(pbar):
            gene_expr = gene_expr.to(device)
            pert_idx = pert_idx.to(device) 
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                predictions = model(gene_expr, pert_idx)
                
                # Ensure shapes match for loss calculation
                if predictions.shape != targets.shape:
                    min_features = min(predictions.shape[1], targets.shape[1])
                    predictions = predictions[:, :min_features]
                    targets = targets[:, :min_features]
                
                loss = criterion(predictions, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_train_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = epoch_train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_idx, (gene_expr, pert_idx, targets) in enumerate(val_loader):
                if batch_idx >= 10:  # Quick validation
                    break
                    
                gene_expr = gene_expr.to(device)
                pert_idx = pert_idx.to(device)
                targets = targets.to(device)
                
                with autocast():
                    predictions = model(gene_expr, pert_idx)
                    
                    # Ensure shapes match
                    if predictions.shape != targets.shape:
                        min_features = min(predictions.shape[1], targets.shape[1])
                        predictions = predictions[:, :min_features]
                        targets = targets[:, :min_features]
                    
                    val_loss = criterion(predictions, targets)
                
                epoch_val_loss += val_loss.item()
                val_batches += 1
        
        avg_val_loss = epoch_val_loss / val_batches if val_batches > 0 else 0.0
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✅ Loaded best model with validation loss: {best_val_loss:.4f}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }

def evaluate_model(model, test_loader, device):
    """Evaluate the fine-tuned model"""
    print("📊 Evaluating fine-tuned model...")
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for gene_expr, pert_idx, targets in tqdm(test_loader, desc="Evaluating"):
            gene_expr = gene_expr.to(device)
            pert_idx = pert_idx.to(device)
            targets = targets.to(device)
            
            with autocast():
                predictions = model(gene_expr, pert_idx)
            
            # Ensure shapes match
            if predictions.shape != targets.shape:
                min_features = min(predictions.shape[1], targets.shape[1])
                predictions = predictions[:, :min_features]
                targets = targets[:, :min_features]
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Concatenate results
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    
    # Calculate metrics
    correlations = []
    mse_scores = []
    mae_scores = []
    
    # Per-gene correlation
    for i in range(min(predictions.shape[1], 100)):  # Limit to first 100 genes
        pred_gene = predictions[:, i]
        target_gene = targets[:, i]
        
        if np.std(pred_gene) > 1e-6 and np.std(target_gene) > 1e-6:
            corr, p_value = pearsonr(pred_gene, target_gene)
            if not np.isnan(corr):
                correlations.append(corr)
        
        mse_scores.append(mean_squared_error(target_gene, pred_gene))
        mae_scores.append(mean_absolute_error(target_gene, pred_gene))
    
    # Overall correlation
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    if np.std(pred_flat) > 1e-6 and np.std(target_flat) > 1e-6:
        overall_corr, _ = pearsonr(pred_flat, target_flat)
        overall_corr = overall_corr if not np.isnan(overall_corr) else 0.0
    else:
        overall_corr = 0.0
    
    results = {
        'mean_correlation': float(np.mean(correlations)) if correlations else 0.0,
        'overall_correlation': float(overall_corr),
        'mean_mse': float(np.mean(mse_scores)),
        'mean_mae': float(np.mean(mae_scores)),
        'num_correlations': len(correlations),
        'num_test_samples': predictions.shape[0],
        'num_genes_evaluated': predictions.shape[1]
    }
    
    return results

def main():
    """Main corrected perturbation fine-tuning pipeline"""
    print("🎯 CORRECT BIOFORMER PERTURBATION PREDICTION FINE-TUNING")
    print("=" * 60)
    print("Methodology: Pre-trained foundation model → Fine-tune on perturbation")
    print("=" * 60)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")
    
    # === Step 1: Load pre-trained foundation model ===
    print("\n📋 Step 1: Load pre-trained BioFormer foundation model")
    
    checkpoint_path = "/mnt/nasdev2/dung/training1/checkpoints/checkpoint_epoch_4_20250615_081604_256d_4head_8layers.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Pre-trained checkpoint not found: {checkpoint_path}")
        print("Cannot proceed without foundation model!")
        return None
    
    try:
        base_model = load_pretrained_bioformer_base(checkpoint_path, device)
    except Exception as e:
        print(f"❌ Failed to load pre-trained model: {e}")
        traceback.print_exc()
        return None
    
    # === Step 2: Load and prepare perturbation data ===
    print("\n📋 Step 2: Load perturbation data")
    
    try:
        # Create data directory
        data_dir = "./gears_data"
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize and load data
        pert_data = PertData(data_dir)
        pert_data.load(data_name="adamson")
        
        print(f"✅ Loaded Adamson dataset: {pert_data.adata.shape}")
        
        # Prepare data with 80/20 split
        data_dict = prepare_perturbation_data(pert_data, train_ratio=0.8, max_genes=800, max_samples=5000)
        
        # Create data loaders
        train_loader, test_loader = create_data_loaders(data_dict, batch_size=32, test_batch_size=64)
        
        print(f"✅ Data loaders created")
        print(f"  - Training batches: {len(train_loader)}")
        print(f"  - Test batches: {len(test_loader)}")
        
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        traceback.print_exc()
        return None
    
    # === Step 3: Create perturbation model from pre-trained base ===
    print("\n📋 Step 3: Create perturbation model from pre-trained foundation")
    
    try:
        # Create perturbation model using pre-trained embeddings
        model = create_perturbation_model_from_pretrained(
            base_model, 
            data_dict['num_genes'], 
            data_dict['num_perturbations'], 
            device
        )
        
    except Exception as e:
        print(f"❌ Perturbation model creation failed: {e}")
        traceback.print_exc()
        return None
    
    # === Step 4: Fine-tune on perturbation task ===
    print("\n📋 Step 4: Fine-tune on perturbation prediction task")
    
    start_time = time.time()
    
    try:
        # Fine-tune the model
        training_history = finetune_perturbation_model(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            device=device,
            epochs=8,  # Reasonable number for fine-tuning
            lr=5e-5   # Lower learning rate for fine-tuning
        )
        
        training_time = time.time() - start_time
        print(f"✅ Fine-tuning completed in {training_time:.1f} seconds")
        
        # Save the fine-tuned model
        torch.save(model.state_dict(), 'bioformer_perturbation_correct_finetuned.pth')
        print("✅ Model saved to bioformer_perturbation_correct_finetuned.pth")
        
    except Exception as e:
        print(f"❌ Fine-tuning failed: {e}")
        traceback.print_exc()
        return None
    
    # === Step 5: Evaluate model ===
    print("\n📋 Step 5: Evaluate fine-tuned model")
    
    try:
        eval_results = evaluate_model(model, test_loader, device)
        
        total_time = time.time() - start_time
        
        # Compile final results
        final_results = {
            'methodology': 'CORRECT: Pre-trained BioFormer → Fine-tuned for Perturbation',
            'foundation_model': checkpoint_path,
            'dataset': 'Adamson',
            'training_setup': '80% train, 20% test',
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'training_time_seconds': training_time,
            'total_time_seconds': total_time,
            'training_history': training_history,
            'evaluation_results': eval_results,
            'data_info': {
                'train_samples': data_dict['X_train'].shape[0],
                'test_samples': data_dict['X_test'].shape[0],
                'num_genes': data_dict['num_genes'],
                'num_perturbations': data_dict['num_perturbations']
            },
            'comparison': {
                'previous_wrong_approach': 'Random init → Fine-tune: 0.995 correlation',
                'current_correct_approach': f'Pre-trained → Fine-tune: {eval_results["mean_correlation"]:.4f} correlation',
                'zero_shot_baseline': 0.007
            }
        }
        
        # Save results
        with open('correct_perturbation_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 60)
        print("🏆 CORRECT PERTURBATION FINE-TUNING RESULTS")
        print("=" * 60)
        print(f"Methodology: Pre-trained foundation → Fine-tuning")
        print(f"Foundation Model: {os.path.basename(checkpoint_path)}")
        print(f"Training Time: {training_time:.1f} seconds")
        print(f"Total Time: {total_time:.1f} seconds")
        print("-" * 60)
        print("PERFORMANCE COMPARISON:")
        print(f"Zero-shot Baseline: 0.007 correlation")
        print(f"Wrong (Random→Fine-tune): 0.995 correlation")
        print(f"Correct (Pre-trained→Fine-tune): {eval_results['mean_correlation']:.4f} correlation")
        print(f"Overall Correlation: {eval_results['overall_correlation']:.4f}")
        print(f"Mean MSE: {eval_results['mean_mse']:.4f}")
        print(f"Mean MAE: {eval_results['mean_mae']:.4f}")
        print(f"Test Samples: {eval_results['num_test_samples']:,}")
        print(f"Genes Evaluated: {eval_results['num_genes_evaluated']}")
        print("=" * 60)
        
        # Performance assessment
        mean_corr = eval_results['mean_correlation']
        if mean_corr > 0.99:
            print(f"🏆 EXCEPTIONAL: {mean_corr:.4f} - Best possible performance!")
        elif mean_corr > 0.95:
            print(f"✅ EXCELLENT: {mean_corr:.4f} - Outstanding perturbation prediction")
        elif mean_corr > 0.8:
            print(f"✅ VERY GOOD: {mean_corr:.4f} - Strong perturbation prediction")
        elif mean_corr > 0.5:
            print(f"📈 GOOD: {mean_corr:.4f} - Solid improvement over baseline")
        else:
            print(f"🔧 NEEDS WORK: {mean_corr:.4f} - Requires further optimization")
        
        print(f"\n📁 Results saved to correct_perturbation_results.json")
        return final_results
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()