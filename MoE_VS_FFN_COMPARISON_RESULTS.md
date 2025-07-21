# BioFormer MoE vs FFN Comparison Results

## Gene Vocabulary Analysis
- **Data folder vocabulary**: `/home/tripham/scgpt/trial_3_based_moe/data/selected_genes.txt` (1,000 genes)
- **Preprocessed vocabulary**: `/mnt/nasdev2/dung/preprocessed/selected_genes.txt` (1,000 genes)
- **Gene overlap between vocabularies**: 185 genes (18.0%)

This explains why using the correct preprocessed vocabulary is crucial for accurate model evaluation.

## Model Comparison Results

### PBMC10k Dataset
**Gene Overlap**: 24.2% (242/1,000 genes)

| Model | Checkpoint | NMI | ARI | Silhouette | Graph Connectivity | AvgBIO | AvgBATCH |
|-------|------------|-----|-----|------------|-------------------|--------|----------|
| **FFN** | `/mnt/nasdev2/dung/preprocessed/training1/checkpoints/checkpoint_epoch_3_20250705_014106.pt` | 0.7811 | 0.4086 | 0.9230 | 1.0000 | **0.7042** | 1.0000 |
| **MoE** | `/mnt/nasdev2/dung/training1/checkpoints/checkpoint_epoch_4_20250615_081604_256d_4head_8layers.pt` | 0.6002 | 0.1277 | 0.9378 | 1.0000 | **0.5553** | 1.0000 |

### COVID-19 Dataset  
**Gene Overlap**: 46.2% (462/1,000 genes using ENSEMBL IDs)

| Model | Checkpoint | NMI | ARI | Silhouette | Graph Connectivity | AvgBIO | AvgBATCH |
|-------|------------|-----|-----|------------|-------------------|--------|----------|
| **FFN** | `/mnt/nasdev2/dung/preprocessed/training1/checkpoints/checkpoint_epoch_3_20250705_014106.pt` | 0.8185 | 0.3759 | 0.5636 | 0.6742 | **0.5860** | 0.6742 |
| **MoE** | `/mnt/nasdev2/dung/training1/checkpoints/checkpoint_epoch_4_20250615_081604_256d_4head_8layers.pt` | 0.7260 | 0.1507 | 0.9149 | 0.7298 | **0.5972** | 0.7298 |

## Key Findings

### Performance Analysis
1. **PBMC10k Results**:
   - **FFN outperforms MoE** significantly in biological preservation (AvgBIO: 0.7042 vs 0.5553)
   - Both models achieve **perfect batch integration** (AvgBATCH = 1.0000)
   - FFN shows better clustering quality (NMI: 0.7811 vs 0.6002, ARI: 0.4086 vs 0.1277)

2. **COVID-19 Results**:
   - **MoE slightly outperforms FFN** in overall biological preservation (AvgBIO: 0.5972 vs 0.5860)
   - **MoE shows better batch integration** (AvgBATCH: 0.7298 vs 0.6742)
   - Trade-off between clustering metrics: FFN has better NMI/ARI, MoE has better Silhouette

### Model Architecture Comparison
- **FFN Model**: 512d, 8 heads, 8 layers (simpler architecture)
- **MoE Model**: 256d, 4 heads, 8 layers with Mixture of Experts (more complex architecture)

### Dataset-Dependent Performance
- **Low gene overlap (24.2% - PBMC10k)**: FFN performs significantly better
- **Higher gene overlap (46.2% - COVID-19)**: Performance is more comparable, with slight MoE advantage

## Conclusions

1. **FFN Model Advantages**:
   - Better performance on datasets with lower gene vocabulary overlap
   - Superior clustering quality on PBMC10k benchmark
   - Simpler architecture, potentially more robust

2. **MoE Model Advantages**:
   - Slightly better performance on datasets with higher gene overlap
   - Better batch integration on COVID-19 dataset
   - More sophisticated architecture for handling diverse data patterns

3. **Gene Vocabulary Impact**:
   - The 18% overlap between different gene vocabularies significantly affects model performance
   - Using the correct preprocessing vocabulary is crucial for accurate evaluation
   - Higher gene overlap correlates with more comparable model performance

4. **Practical Recommendations**:
   - For general use with diverse datasets: **FFN model** appears more robust
   - For datasets well-aligned with training vocabulary: **MoE model** may provide slight advantages
   - Always ensure gene vocabulary alignment between training and testing datasets