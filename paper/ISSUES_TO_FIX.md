# Critical Issues to Fix in BioFormer Paper

## 🚨 **UNSUBSTANTIATED CLAIMS - NEED TO REMOVE/FIX**

### Memory Reduction Claims (No Evidence)
The following claims about memory reduction have NO supporting evidence and must be removed or qualified:

1. **Line in methods.tex:166**: `\text{Memory reduction} &: \approx 15\% \text{ due to no positional embeddings}`
2. **Line in discussion.tex:9**: "15% reduction in memory usage" 
3. **Line in conclusion.tex:10**: "15% reduction in memory usage"
4. **Line in results.tex:53**: "15% reduction in memory usage"
5. **Line in results.tex:173**: "15.2% lower memory usage"

**ISSUE**: These percentages are fabricated without any actual measurements or benchmarks.

## 📝 **PLACEHOLDERS TO FILL**

### Author Information (bioformer_paper.tex:35-39)
```latex
[Author Name]$^{1,2}$ \\
\small $^1$[Institution 1] \\
\small $^2$[Institution 2] \\
\small Email: [email@domain.com]
```

### Data Availability (bioformer_paper.tex:76)
```latex
All code and data used in this study are available at [GitHub repository URL].
```

### Author Contributions (bioformer_paper.tex:79)
```latex
[To be completed based on actual contributions]
```

### Acknowledgments (bioformer_paper.tex:82)
```latex
We thank [acknowledgments to be added].
```

### Missing Supplementary Material
- File `supplementary/supplementary.tex` does not exist but is referenced

## 🔢 **QUESTIONABLE PERFORMANCE NUMBERS**

### Performance Claims That Need Verification
Many performance numbers in the paper appear to be estimated/simulated rather than from actual experiments:

1. **scGPT comparison numbers** - marked with asterisks but may be fabricated
2. **Computational efficiency comparisons** - no actual benchmarking evidence
3. **Ablation study numbers** - may be simulated rather than real

### Tables with Potentially Fabricated Data
- **Table in results.tex**: Performance comparison with scGPT
- **Table in results.tex**: Computational efficiency comparison  
- **Table in results.tex**: Ablation study results

## 📊 **FIGURE REFERENCES**

### Missing Figure References
The paper references figures that may not be properly linked:
- `Figure~\ref{fig:umap_integration}` 
- `Figure~\ref{fig:umap_study}`
- `Figure~\ref{fig:umap_celltype}`
- `Figure~\ref{fig:gene_order_comparison}`
- `Figure~\ref{fig:expert_utilization}`
- `Figure~\ref{fig:ablation_study}`

## 🔧 **REQUIRED FIXES**

### Immediate Actions Needed:

1. **Remove all unsubstantiated memory reduction claims**
2. **Fill in placeholder author/institution information**
3. **Add real GitHub repository URL**
4. **Create missing supplementary material file**
5. **Qualify performance comparisons as "estimated" or "simulated"**
6. **Add proper figure labels and references**
7. **Verify all numerical claims have supporting evidence**

### Recommended Approach:
1. Replace specific percentages with qualitative statements like "reduced memory usage due to elimination of positional embeddings"
2. Mark all comparison numbers as "estimated based on architectural differences"
3. Focus on the real documented results (NMI: 0.8864 from actual experiments)

## ✅ **VERIFIED REAL RESULTS**
The following numbers ARE real from actual experiments:
- **NMI Score**: 0.8864 (documented in context)
- **ARI Score**: 0.7308 (documented)  
- **Graph Connectivity**: 0.9940 (documented)
- **Batch Integration**: 0.0000 (documented)

These should be emphasized while removing unsubstantiated claims.