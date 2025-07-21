#!/usr/bin/env python3
"""
Create gene mapping between ENSEMBL IDs and gene symbols
"""
import pandas as pd
import requests
import json
from pathlib import Path
import time

def get_ensembl_to_symbol_mapping(ensembl_ids):
    """
    Get gene symbol mapping from ENSEMBL IDs using BioMart API
    """
    print(f"Fetching gene symbols for {len(ensembl_ids)} ENSEMBL IDs...")
    
    # Split into batches to avoid API limits
    batch_size = 100
    mapping = {}
    
    for i in range(0, len(ensembl_ids), batch_size):
        batch = ensembl_ids[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(ensembl_ids) + batch_size - 1)//batch_size}")
        
        # Create XML query for BioMart
        xml_query = f'''<?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE Query>
        <Query  virtualSchemaName = "default" formatter = "TSV" header = "0" uniqueRows = "0" count = "" datasetConfigVersion = "0.6" >
            <Dataset name = "hsapiens_gene_ensembl" interface = "default" >
                <Filter name = "ensembl_gene_id" value = "{','.join(batch)}"/>
                <Attribute name = "ensembl_gene_id" />
                <Attribute name = "external_gene_name" />
            </Dataset>
        </Query>'''
        
        try:
            # Submit query to BioMart
            response = requests.post(
                'http://www.ensembl.org/biomart/martservice',
                data={'query': xml_query},
                timeout=30
            )
            
            if response.status_code == 200:
                lines = response.text.strip().split('\n')
                for line in lines:
                    if line.strip():
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            ensembl_id = parts[0]
                            gene_symbol = parts[1]
                            if gene_symbol:  # Only add if gene symbol exists
                                mapping[ensembl_id] = gene_symbol
            else:
                print(f"API request failed with status {response.status_code}")
                
        except Exception as e:
            print(f"Error fetching batch: {e}")
        
        # Rate limiting
        time.sleep(1)
    
    print(f"Found mappings for {len(mapping)}/{len(ensembl_ids)} genes")
    return mapping

def create_mapping_file():
    """Create gene mapping file"""
    # Load model vocabulary (ENSEMBL IDs)
    vocab_file = "/mnt/nasdev2/dung/preprocessed/selected_genes.txt"
    with open(vocab_file, 'r') as f:
        ensembl_ids = [line.strip() for line in f.readlines()]
    
    print(f"Loaded {len(ensembl_ids)} ENSEMBL IDs from model vocabulary")
    
    # Get gene symbol mapping
    mapping = get_ensembl_to_symbol_mapping(ensembl_ids)
    
    # Create DataFrame
    df = pd.DataFrame([
        {'ensembl_id': ensembl_id, 'gene_symbol': mapping.get(ensembl_id, '')}
        for ensembl_id in ensembl_ids
    ])
    
    # Save mapping
    output_file = Path("/home/tripham/scgpt/trial_3_based_moe/gene_mapping.csv")
    df.to_csv(output_file, index=False)
    print(f"Gene mapping saved to {output_file}")
    
    # Print summary
    mapped_count = df[df['gene_symbol'] != ''].shape[0]
    print(f"Mapping summary: {mapped_count}/{len(ensembl_ids)} genes mapped")
    
    return df

if __name__ == "__main__":
    create_mapping_file()