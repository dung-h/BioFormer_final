#!/usr/bin/env python3
"""
Run comprehensive comparison of MoE BioFormer across multiple checkpoints
Tests both PBMC and Immune Human datasets
"""
import os
import subprocess
import json
import pandas as pd
from datetime import datetime

def run_test_with_checkpoint(script_name, checkpoint_path, output_suffix):
    """Run a test script with a specific checkpoint and return the log file path."""
    log_file = f"{script_name.replace('.py', '')}_{output_suffix}.log"
    
    cmd = [
        'python3', script_name,
        '--checkpoint', checkpoint_path
    ]
    
    print(f"Running {script_name} with {os.path.basename(checkpoint_path)}...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Log file: {log_file}")
    
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
        print(f"✅ Completed: {log_file}")
        return log_file, True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {script_name} with {checkpoint_path}")
        return log_file, False

def extract_metrics_from_log(log_file):
    """Extract performance metrics from log file."""
    metrics = {}
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Extract metrics using string parsing
        lines = content.split('\n')
        for line in lines:
            if 'NMI =' in line or 'NMI (cell type coherence):' in line:
                metrics['NMI'] = float(line.split('=')[-1].strip() if '=' in line else line.split(':')[-1].strip())
            elif 'ARI =' in line or 'ARI (cell type coherence):' in line:
                metrics['ARI'] = float(line.split('=')[-1].strip() if '=' in line else line.split(':')[-1].strip())
            elif 'Silhouette =' in line or 'Silhouette score:' in line:
                metrics['Silhouette'] = float(line.split('=')[-1].strip() if '=' in line else line.split(':')[-1].strip())
            elif 'Graph connectivity =' in line or 'Graph connectivity:' in line:
                metrics['Graph_Connectivity'] = float(line.split('=')[-1].strip() if '=' in line else line.split(':')[-1].strip())
            elif 'AvgBIO =' in line or 'AvgBIO (biological):' in line:
                metrics['AvgBIO'] = float(line.split('=')[-1].strip() if '=' in line else line.split(':')[-1].strip())
            elif 'AvgBATCH =' in line or 'AvgBATCH (integration):' in line:
                metrics['AvgBATCH'] = float(line.split('=')[-1].strip() if '=' in line else line.split(':')[-1].strip())
    
    except Exception as e:
        print(f"Error extracting metrics from {log_file}: {e}")
    
    return metrics

def main():
    # Define checkpoints to test
    checkpoint_base = "/mnt/nasdev2/dung/training_1k/checkpoints"
    checkpoints = [
        f"{checkpoint_base}/checkpoint_epoch_1_20250708_000207.pt",
        f"{checkpoint_base}/checkpoint_epoch_2_20250708_221741.pt", 
        f"{checkpoint_base}/checkpoint_epoch_3_20250709_210643.pt",
        f"{checkpoint_base}/checkpoint_epoch_4_20250711_232234.pt"
    ]
    
    # Test scripts
    scripts = [
        "test_moe_integration.py",
        "test_immune_human_integration.py"
    ]
    
    results = {}
    
    print(f"{'='*80}")
    print(f"MoE BioFormer Checkpoint Comparison")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # Run all combinations
    for i, checkpoint in enumerate(checkpoints):
        epoch_num = i + 1
        epoch_name = f"epoch_{epoch_num}"
        results[epoch_name] = {}
        
        print(f"\n{'='*60}")
        print(f"Testing Epoch {epoch_num}: {os.path.basename(checkpoint)}")
        print(f"{'='*60}")
        
        for script in scripts:
            dataset_name = "PBMC" if "moe_integration" in script else "Immune_Human"
            output_suffix = f"epoch{epoch_num}_{dataset_name.lower()}"
            
            log_file, success = run_test_with_checkpoint(script, checkpoint, output_suffix)
            
            if success:
                metrics = extract_metrics_from_log(log_file)
                results[epoch_name][dataset_name] = {
                    'log_file': log_file,
                    'metrics': metrics,
                    'success': True
                }
            else:
                results[epoch_name][dataset_name] = {
                    'log_file': log_file,
                    'success': False
                }
    
    # Save raw results
    with open('checkpoint_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary tables
    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*80}")
    
    # PBMC Results Table
    print(f"\n📊 PBMC Dataset Results:")
    pbmc_data = []
    for epoch, data in results.items():
        if 'PBMC' in data and data['PBMC']['success']:
            metrics = data['PBMC']['metrics']
            row = {'Epoch': epoch.replace('epoch_', '')}
            row.update(metrics)
            pbmc_data.append(row)
    
    if pbmc_data:
        pbmc_df = pd.DataFrame(pbmc_data)
        print(pbmc_df.to_string(index=False, float_format='%.4f'))
    
    # Immune Human Results Table
    print(f"\n📊 Immune Human Dataset Results:")
    immune_data = []
    for epoch, data in results.items():
        if 'Immune_Human' in data and data['Immune_Human']['success']:
            metrics = data['Immune_Human']['metrics']
            row = {'Epoch': epoch.replace('epoch_', '')}
            row.update(metrics)
            immune_data.append(row)
    
    if immune_data:
        immune_df = pd.DataFrame(immune_data)
        print(immune_df.to_string(index=False, float_format='%.4f'))
    
    print(f"\n{'='*80}")
    print(f"Comparison completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: checkpoint_comparison_results.json")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()