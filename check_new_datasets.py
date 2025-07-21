#!/usr/bin/env python3
"""
Check if new datasets have been added and analyze the enrichment
"""
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def check_dataset_count():
    """Check how many datasets are currently available"""
    data_dir = Path('/mnt/nasdev2/dung/data')
    
    print("="*80)
    print("DATASET COUNT CHECK")
    print("="*80)
    
    h5ad_files = list(data_dir.glob("*.h5ad"))
    print(f"Total h5ad files found: {len(h5ad_files)}")
    
    # List all files
    for file_path in sorted(h5ad_files):
        try:
            adata = sc.read_h5ad(file_path)
            print(f"{file_path.name}: {adata.n_obs:,} cells, {adata.n_vars:,} genes")
        except Exception as e:
            print(f"{file_path.name}: Error reading - {e}")
    
    return len(h5ad_files)

def analyze_new_datasets():
    """Analyze datasets beyond data20.h5ad"""
    data_dir = Path('/mnt/nasdev2/dung/data')
    
    print("\n" + "="*80)
    print("NEW DATASETS ANALYSIS (data21+)")
    print("="*80)
    
    new_datasets = []
    new_cell_types = set()
    new_tissues = set()
    
    # Check for datasets beyond data20
    for i in range(21, 50):  # Check up to data50
        file_path = data_dir / f"data{i}.h5ad"
        if file_path.exists():
            try:
                print(f"\nAnalyzing NEW dataset {file_path.name}...")
                adata = sc.read_h5ad(file_path)
                
                info = {
                    'filename': file_path.name,
                    'n_cells': adata.n_obs,
                    'n_genes': adata.n_vars,
                    'cell_types': [],
                    'tissues': [],
                    'species': 'unknown'
                }
                
                # Check cell types
                cell_type_cols = ['cell_type', 'celltype', 'cell_ontology_class', 'annotation']
                for col in cell_type_cols:
                    if col in adata.obs.columns:
                        unique_types = adata.obs[col].unique()
                        info['cell_types'] = [str(x) for x in unique_types if pd.notna(x)]
                        new_cell_types.update(info['cell_types'])
                        break
                
                # Check tissues
                tissue_cols = ['tissue', 'organ', 'tissue_type']
                for col in tissue_cols:
                    if col in adata.obs.columns:
                        unique_tissues = adata.obs[col].unique()
                        info['tissues'] = [str(x) for x in unique_tissues if pd.notna(x)]
                        new_tissues.update(info['tissues'])
                        break
                
                # Check species
                species_cols = ['organism', 'species']
                for col in species_cols:
                    if col in adata.obs.columns:
                        info['species'] = str(adata.obs[col].iloc[0])
                        break
                
                print(f"  Cells: {info['n_cells']:,}, Genes: {info['n_genes']:,}")
                print(f"  Species: {info['species']}")
                print(f"  Cell types ({len(info['cell_types'])}): {info['cell_types'][:10]}")
                print(f"  Tissues ({len(info['tissues'])}): {info['tissues']}")
                
                new_datasets.append(info)
                
            except Exception as e:
                print(f"Error analyzing {file_path.name}: {e}")
    
    print(f"\n" + "="*80)
    print(f"NEW DATASETS SUMMARY")
    print("="*80)
    print(f"New datasets found: {len(new_datasets)}")
    print(f"New unique cell types: {len(new_cell_types)}")
    print(f"New unique tissues: {len(new_tissues)}")
    
    if new_cell_types:
        print(f"\nNEW CELL TYPES ADDED:")
        for i, ct in enumerate(sorted(new_cell_types), 1):
            print(f"  {i:3d}. {ct}")
    
    if new_tissues:
        print(f"\nNEW TISSUES ADDED:")
        for i, tissue in enumerate(sorted(new_tissues), 1):
            print(f"  {i:2d}. {tissue}")
    
    return new_datasets, new_cell_types, new_tissues

def assess_enrichment():
    """Assess if the new datasets address the gaps identified before"""
    
    print(f"\n" + "="*80)
    print("ENRICHMENT ASSESSMENT")
    print("="*80)
    
    # Get current analysis
    new_datasets, new_cell_types, new_tissues = analyze_new_datasets()
    
    # Previous gaps identified
    missing_categories = {
        'Neural cells': ['astrocyte', 'oligodendrocyte', 'microglia', 'ependymal cell', 'choroid plexus cell'],
        'Immune diversity': ['regulatory T cell', 'gamma delta T cell', 'innate lymphoid cell'],
        'Developmental': ['embryonic', 'fetal', 'stem cell'],
        'Specific organs': ['heart', 'liver', 'muscle', 'adipose']
    }
    
    print("CHECKING IF GAPS WERE FILLED:")
    
    for category, missing_types in missing_categories.items():
        found_types = []
        for missing_type in missing_types:
            for new_type in new_cell_types:
                if missing_type.lower() in new_type.lower():
                    found_types.append(new_type)
        
        if found_types:
            print(f"\n✅ {category}: IMPROVED")
            print(f"   Found: {found_types}")
        else:
            print(f"\n❌ {category}: STILL MISSING")
            print(f"   Still need: {missing_types}")
    
    # Overall assessment
    total_new_cells = sum(d['n_cells'] for d in new_datasets)
    
    print(f"\n" + "="*80)
    print("OVERALL ENRICHMENT ASSESSMENT")
    print("="*80)
    print(f"Added {len(new_datasets)} new datasets")
    print(f"Added {total_new_cells:,} new cells")
    print(f"Added {len(new_cell_types)} unique cell types")
    print(f"Added {len(new_tissues)} unique tissues")
    
    if len(new_datasets) >= 5 and len(new_cell_types) >= 20:
        print("\n✅ GOOD ENRICHMENT: Substantial diversity added")
    elif len(new_datasets) >= 3 and len(new_cell_types) >= 10:
        print("\n⚠️  MODERATE ENRICHMENT: Some diversity added")
    else:
        print("\n❌ LIMITED ENRICHMENT: More diversity needed")
    
    print("\nRECOMMENDATIONS:")
    print("1. If neural diversity is still limited, add brain atlas data")
    print("2. If immune diversity is still limited, add immunology-focused datasets")
    print("3. Consider developmental datasets (embryonic/fetal)")
    print("4. Consider disease vs healthy comparisons")

if __name__ == "__main__":
    total_files = check_dataset_count()
    if total_files > 20:
        analyze_new_datasets()
        assess_enrichment()
    else:
        print("No new datasets found beyond the original 20")