#!/usr/bin/env python3
"""
Download Adamson and Norman perturbation datasets via GEARS
Save in .h5ad format and analyze common genes
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path

# Import GEARS
try:
    from gears import PertData
    print("✅ GEARS imported successfully")
except ImportError as e:
    print(f"❌ GEARS import failed: {e}")
    print("Please install GEARS: pip install gears-sc")
    sys.exit(1)

def download_and_save_datasets():
    """Download Adamson and Norman datasets and save as .h5ad"""
    print("🔄 Downloading perturbation datasets via GEARS...")
    
    # Create directories
    base_dir = Path("test_datasets")
    perturbation_dir = base_dir / "perturbation"
    raw_dir = base_dir / "raw_perturbation"
    
    for dir_path in [base_dir, perturbation_dir, raw_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")
    
    # Data directory for GEARS
    gears_data_dir = "./gears_data"
    os.makedirs(gears_data_dir, exist_ok=True)
    
    datasets = {}
    
    # Download datasets
    for dataset_name in ["adamson", "norman"]:
        print(f"\n📥 Downloading {dataset_name.upper()} dataset...")
        
        try:
            # Initialize PertData and load dataset
            pert_data = PertData(gears_data_dir)
            pert_data.load(data_name=dataset_name)
            
            # Get the AnnData object
            adata = pert_data.adata.copy()
            
            # Add dataset info
            adata.uns['dataset_name'] = dataset_name
            adata.uns['source'] = 'GEARS'
            
            print(f"✅ {dataset_name.upper()} dataset loaded:")
            print(f"  Shape: {adata.shape}")
            print(f"  Cells: {adata.shape[0]:,}")
            print(f"  Genes: {adata.shape[1]:,}")
            print(f"  Perturbations: {len(adata.obs['condition'].unique())}")
            print(f"  Data type: {type(adata.X)}")
            print(f"  Obs columns: {list(adata.obs.columns)}")
            print(f"  Var columns: {list(adata.var.columns)}")
            
            # Check unique conditions
            conditions = adata.obs['condition'].unique()
            print(f"  Sample conditions: {list(conditions[:5])}")
            
            # Store dataset
            datasets[dataset_name] = adata
            
            # Save raw dataset
            raw_path = raw_dir / f"{dataset_name}_raw.h5ad"
            adata.write(raw_path)
            print(f"💾 Saved raw dataset to: {raw_path}")
            
        except Exception as e:
            print(f"❌ Failed to download {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return datasets

def analyze_common_genes(datasets):
    """Analyze common genes across datasets"""
    print("\n🔬 Analyzing common genes across datasets...")
    
    if len(datasets) < 2:
        print("❌ Need at least 2 datasets for comparison")
        return None
    
    # Get gene lists
    gene_sets = {}
    for dataset_name, adata in datasets.items():
        if 'gene_name' in adata.var.columns:
            genes = set(adata.var['gene_name'].values)
        else:
            genes = set(adata.var_names.values)
        gene_sets[dataset_name] = genes
        print(f"📊 {dataset_name.upper()}: {len(genes)} genes")
    
    # Find intersections
    dataset_names = list(gene_sets.keys())
    common_genes = gene_sets[dataset_names[0]]
    
    for name in dataset_names[1:]:
        common_genes = common_genes.intersection(gene_sets[name])
    
    print(f"\n🎯 GENE OVERLAP ANALYSIS:")
    print(f"Common genes across all datasets: {len(common_genes)}")
    
    # Pairwise comparisons
    for i, name1 in enumerate(dataset_names):
        for name2 in dataset_names[i+1:]:
            overlap = gene_sets[name1].intersection(gene_sets[name2])
            union = gene_sets[name1].union(gene_sets[name2])
            jaccard = len(overlap) / len(union) if union else 0
            
            print(f"  {name1} ∩ {name2}: {len(overlap)} genes (Jaccard: {jaccard:.3f})")
    
    # Save common genes
    if common_genes:
        common_genes_list = sorted(list(common_genes))
        common_genes_df = pd.DataFrame({
            'gene_name': common_genes_list,
            'gene_index': range(len(common_genes_list))
        })
        
        common_genes_path = Path("test_datasets/perturbation/common_genes.csv")
        common_genes_df.to_csv(common_genes_path, index=False)
        print(f"💾 Saved common genes to: {common_genes_path}")
        
        # Show sample of common genes
        print(f"\n📋 Sample common genes: {common_genes_list[:10]}")
    
    return common_genes

def create_standardized_datasets(datasets, common_genes):
    """Create standardized datasets with common genes"""
    print("\n🔧 Creating standardized datasets with common genes...")
    
    if not common_genes:
        print("❌ No common genes found, using all genes from each dataset")
        
        # Just save individual datasets
        for dataset_name, adata in datasets.items():
            output_path = Path(f"test_datasets/perturbation/{dataset_name}.h5ad")
            adata.write(output_path)
            print(f"💾 Saved {dataset_name} to: {output_path}")
        return
    
    common_genes_list = sorted(list(common_genes))
    standardized_datasets = {}
    
    for dataset_name, adata in datasets.items():
        print(f"🔄 Processing {dataset_name.upper()}...")
        
        # Get gene column
        if 'gene_name' in adata.var.columns:
            gene_col = 'gene_name'
        else:
            gene_col = adata.var_names
            
        # Filter to common genes
        if gene_col == 'gene_name':
            gene_mask = adata.var['gene_name'].isin(common_genes_list)
        else:
            gene_mask = adata.var_names.isin(common_genes_list)
            
        adata_filtered = adata[:, gene_mask].copy()
        
        print(f"  Original shape: {adata.shape}")
        print(f"  Filtered shape: {adata_filtered.shape}")
        
        # Reorder genes to match common gene order
        if gene_col == 'gene_name':
            current_genes = adata_filtered.var['gene_name'].values
        else:
            current_genes = adata_filtered.var_names.values
            
        # Create mapping to reorder
        gene_order = []
        for gene in common_genes_list:
            if gene in current_genes:
                if gene_col == 'gene_name':
                    idx = np.where(adata_filtered.var['gene_name'] == gene)[0][0]
                else:
                    idx = np.where(adata_filtered.var_names == gene)[0][0]
                gene_order.append(idx)
        
        if len(gene_order) > 0:
            adata_reordered = adata_filtered[:, gene_order].copy()
            print(f"  Reordered shape: {adata_reordered.shape}")
            
            # Add standardization info
            adata_reordered.uns['standardized'] = True
            adata_reordered.uns['common_genes_count'] = len(common_genes_list)
            adata_reordered.uns['original_dataset'] = dataset_name
            
            standardized_datasets[dataset_name] = adata_reordered
            
            # Save standardized dataset
            output_path = Path(f"test_datasets/perturbation/{dataset_name}.h5ad")
            adata_reordered.write(output_path)
            print(f"💾 Saved standardized {dataset_name} to: {output_path}")
        else:
            print(f"⚠️  No genes found for {dataset_name}")
    
    return standardized_datasets

def analyze_perturbations(datasets):
    """Analyze perturbation conditions across datasets"""
    print("\n🧪 Analyzing perturbation conditions...")
    
    all_conditions = {}
    
    for dataset_name, adata in datasets.items():
        conditions = adata.obs['condition'].value_counts()
        all_conditions[dataset_name] = conditions
        
        print(f"\n📊 {dataset_name.upper()} perturbations:")
        print(f"  Total conditions: {len(conditions)}")
        print(f"  Total cells: {conditions.sum():,}")
        print(f"  Top 5 conditions:")
        for condition, count in conditions.head().items():
            print(f"    {condition}: {count:,} cells")
        
        # Check for control conditions
        control_conditions = [c for c in conditions.index if 'ctrl' in c.lower() or 'control' in c.lower() or c.lower() == 'non-targeting']
        if control_conditions:
            print(f"  Control conditions found: {control_conditions}")
    
    # Save perturbation summary
    summary_path = Path("test_datasets/perturbation/perturbation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("PERTURBATION DATASET SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        for dataset_name, conditions in all_conditions.items():
            f.write(f"{dataset_name.upper()} Dataset:\n")
            f.write(f"  Total conditions: {len(conditions)}\n")
            f.write(f"  Total cells: {conditions.sum():,}\n")
            f.write(f"  Conditions:\n")
            for condition, count in conditions.items():
                f.write(f"    {condition}: {count:,} cells\n")
            f.write("\n")
    
    print(f"💾 Saved perturbation summary to: {summary_path}")

def main():
    """Main function to download and process datasets"""
    print("🎯 PERTURBATION DATASET DOWNLOADER")
    print("=" * 50)
    print("Downloading Adamson and Norman datasets via GEARS")
    print("Saving in .h5ad format with common gene analysis")
    print("=" * 50)
    
    # Download datasets
    datasets = download_and_save_datasets()
    
    if not datasets:
        print("❌ No datasets downloaded successfully")
        return
    
    # Analyze common genes
    common_genes = analyze_common_genes(datasets)
    
    # Create standardized datasets
    standardized_datasets = create_standardized_datasets(datasets, common_genes)
    
    # Analyze perturbations
    analyze_perturbations(datasets)
    
    # Final summary
    print("\n" + "=" * 50)
    print("🏆 DOWNLOAD COMPLETED SUCCESSFULLY")
    print("=" * 50)
    print("📁 Files created in test_datasets/:")
    
    test_dir = Path("test_datasets")
    if test_dir.exists():
        for file_path in test_dir.rglob("*"):
            if file_path.is_file():
                size = file_path.stat().st_size / (1024*1024)  # MB
                print(f"  {file_path.relative_to(test_dir)}: {size:.1f} MB")
    
    print("\n✅ Ready for perturbation prediction experiments!")

if __name__ == "__main__":
    main()