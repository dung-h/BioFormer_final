#!/usr/bin/env python3
"""
Explore CZ CELLxGENE Discover preprocessed datasets to understand data sources.
"""
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def explore_study_file(file_path):
    """Explore a single preprocessed study file."""
    study_info = {}
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Get basic dataset info
            study_info['file_path'] = str(file_path)
            study_info['file_size_mb'] = file_path.stat().st_size / (1024 * 1024)
            
            # Extract available keys
            study_info['keys'] = list(f.keys())
            
            # Get dimensions
            if 'binned_expr' in f:
                shape = f['binned_expr'].shape
                study_info['n_cells'] = shape[0]
                study_info['n_genes'] = shape[1]
            
            # Extract cell types
            if 'cell_type' in f:
                cell_types = f['cell_type'][:]
                if isinstance(cell_types[0], bytes):
                    cell_types = [ct.decode('utf-8') for ct in cell_types]
                study_info['cell_types'] = sorted(list(set(cell_types)))
                study_info['n_cell_types'] = len(study_info['cell_types'])
                
                # Count cells per type
                cell_type_counts = pd.Series(cell_types).value_counts()
                study_info['cell_type_counts'] = cell_type_counts.to_dict()
            
            # Extract tissue information if available
            if 'tissue' in f:
                tissues = f['tissue'][:]
                if isinstance(tissues[0], bytes):
                    tissues = [t.decode('utf-8') for t in tissues]
                study_info['tissues'] = sorted(list(set(tissues)))
                study_info['n_tissues'] = len(study_info['tissues'])
            
            # Extract study metadata if available
            if 'study_id' in f:
                study_ids = f['study_id'][:]
                study_info['study_ids'] = sorted(list(set(study_ids)))
                study_info['n_studies'] = len(study_info['study_ids'])
            
            # Extract other metadata
            for key in ['disease', 'organ', 'organism', 'sex', 'ethnicity', 'development_stage']:
                if key in f:
                    values = f[key][:]
                    if isinstance(values[0], bytes):
                        values = [v.decode('utf-8') for v in values]
                    study_info[key] = sorted(list(set(values)))
            
            # Check for gene names
            if 'gene_names' in f:
                gene_names = f['gene_names'][:]
                if isinstance(gene_names[0], bytes):
                    gene_names = [g.decode('utf-8') for g in gene_names[:10]]  # Just first 10
                study_info['sample_genes'] = gene_names
            
            print(f"✅ Successfully processed {file_path.name}")
            
    except Exception as e:
        print(f"❌ Error processing {file_path.name}: {e}")
        study_info['error'] = str(e)
    
    return study_info

def explore_all_cellxgene_data():
    """Explore all CZ CELLxGENE preprocessed files."""
    data_dir = Path("/mnt/nasdev2/dung/preprocessed")
    
    print("🔍 Exploring CZ CELLxGENE Discover Preprocessed Datasets")
    print("=" * 60)
    
    # Find all preprocessed study files
    study_files = sorted(data_dir.glob("preprocessed_study*.h5"))
    
    print(f"Found {len(study_files)} preprocessed study files")
    print()
    
    all_studies = {}
    total_cells = 0
    all_cell_types = set()
    all_tissues = set()
    
    # Process each study file
    for study_file in study_files:
        study_num = study_file.stem.replace('preprocessed_study', '')
        print(f"📊 Processing Study {study_num}...")
        
        study_info = explore_study_file(study_file)
        all_studies[study_num] = study_info
        
        if 'n_cells' in study_info:
            total_cells += study_info['n_cells']
        
        if 'cell_types' in study_info:
            all_cell_types.update(study_info['cell_types'])
        
        if 'tissues' in study_info:
            all_tissues.update(study_info['tissues'])
        
        print()
    
    # Generate summary report
    print("📋 SUMMARY REPORT")
    print("=" * 60)
    print(f"Total Studies: {len(all_studies)}")
    print(f"Total Cells: {total_cells:,}")
    print(f"Unique Cell Types: {len(all_cell_types)}")
    print(f"Unique Tissues: {len(all_tissues)}")
    print()
    
    # Detailed study breakdown
    print("📊 STUDY BREAKDOWN")
    print("=" * 60)
    
    for study_num in sorted(all_studies.keys(), key=int):
        study = all_studies[study_num]
        if 'error' in study:
            print(f"Study {study_num}: ERROR - {study['error']}")
            continue
            
        print(f"Study {study_num}:")
        print(f"  📁 File: {study['file_path']}")
        print(f"  💾 Size: {study['file_size_mb']:.1f} MB")
        
        if 'n_cells' in study:
            print(f"  🔬 Cells: {study['n_cells']:,}")
        if 'n_genes' in study:
            print(f"  🧬 Genes: {study['n_genes']:,}")
        if 'n_cell_types' in study:
            print(f"  📊 Cell Types: {study['n_cell_types']}")
        if 'n_tissues' in study:
            print(f"  🧪 Tissues: {study['n_tissues']}")
        
        # Show top cell types
        if 'cell_type_counts' in study:
            print("  🏆 Top Cell Types:")
            top_types = sorted(study['cell_type_counts'].items(), key=lambda x: x[1], reverse=True)[:5]
            for cell_type, count in top_types:
                print(f"    - {cell_type}: {count:,} cells")
        
        # Show tissues
        if 'tissues' in study:
            print(f"  🧪 Tissues: {', '.join(study['tissues'][:5])}")
            if len(study['tissues']) > 5:
                print(f"    ... and {len(study['tissues']) - 5} more")
        
        print()
    
    # Show all unique cell types
    print("🧬 ALL UNIQUE CELL TYPES")
    print("=" * 60)
    sorted_cell_types = sorted(all_cell_types)
    for i, cell_type in enumerate(sorted_cell_types):
        print(f"{i+1:2d}. {cell_type}")
    
    print()
    
    # Show all unique tissues
    if all_tissues:
        print("🧪 ALL UNIQUE TISSUES")
        print("=" * 60)
        sorted_tissues = sorted(all_tissues)
        for i, tissue in enumerate(sorted_tissues):
            print(f"{i+1:2d}. {tissue}")
    
    # Generate data source summary for paper
    print("\n📝 DATA SOURCE SUMMARY FOR PAPER")
    print("=" * 60)
    print("Data Source: CZ CELLxGENE Discover (https://cellxgene.cziscience.com/datasets)")
    print(f"Number of Studies: {len(all_studies)}")
    print(f"Total Cells: ~{total_cells/1e6:.1f}M cells")
    print(f"Cell Type Diversity: {len(all_cell_types)} unique cell types")
    if all_tissues:
        print(f"Tissue Diversity: {len(all_tissues)} unique tissues")
    print(f"Gene Vocabulary: 1,000 highly variable genes")
    
    return all_studies

if __name__ == "__main__":
    studies = explore_all_cellxgene_data()
    
    # Save summary to file
    print("\n💾 Saving detailed summary to cellxgene_data_summary.txt")
    
    with open("/home/tripham/scgpt/trial_3_based_moe/cellxgene_data_summary.txt", "w") as f:
        f.write("CZ CELLxGENE Discover Dataset Analysis\n")
        f.write("=" * 50 + "\n\n")
        
        for study_num in sorted(studies.keys(), key=int):
            study = studies[study_num]
            f.write(f"Study {study_num}:\n")
            
            if 'error' in study:
                f.write(f"  ERROR: {study['error']}\n\n")
                continue
            
            for key, value in study.items():
                if key not in ['cell_type_counts', 'file_path']:
                    f.write(f"  {key}: {value}\n")
            f.write("\n")
    
    print("✅ Analysis complete!")