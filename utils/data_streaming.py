import os
import glob
import logging
import time
import re
import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from tqdm import tqdm
import scanpy as sc
import scipy.sparse
from sklearn.preprocessing import LabelEncoder

from utils.preprocessing import cpu_quantile_binning

def custom_collate_fn(batch):
    """Custom collate function to handle scalar and tensor data."""
    elem = batch[0]
    elem_type = type(elem)
    return elem_type({
        key: torch.stack([torch.as_tensor(d[key]) for d in batch])
        if isinstance(elem[key], torch.Tensor) and elem[key].dim() > 0
        else torch.tensor([d[key] for d in batch], dtype=torch.long if key in ['study_id', 'cell_type', 'global_idx', 'perturb_idx'] else elem[key].dtype)
        for key in elem
    })

class StreamingSingleCellDataset(Dataset):
    """Memory-efficient streaming dataset for large-scale single-cell data."""
    
    def __init__(self, data_dir, rank=0, num_cell_types=300):
        self.num_cell_types = num_cell_types
        self.data_files = sorted(glob.glob(os.path.join(data_dir, 'preprocessed_study*.h5')))
        if not self.data_files:
            raise ValueError(f"No HDF5 files found in {data_dir}")
        
        self.rank = rank
        self.cell_type_encoder = LabelEncoder()
        self.merged_cell_types = {}
        
        # Build file index and cell type mapping without loading data
        self._build_file_index()
        self._setup_cell_types(num_cell_types)
        
        logging.info(f"[Rank {rank}] StreamingDataset initialized: {self.total_cells:,} cells across {len(self.data_files)} files")
        logging.info(f"[Rank {rank}] Memory footprint: ~{self._estimate_memory():.2f} GB (streaming)")
    
    def _build_file_index(self):
        """Build index of files and cell counts without loading data."""
        self.file_info = []
        self.total_cells = 0
        cumulative_cells = 0
        
        for file_idx, f in enumerate(self.data_files):
            with h5py.File(f, 'r') as h5:
                n_cells = h5['binned_expr'].shape[0]
                n_genes = h5['binned_expr'].shape[1]
                
                self.file_info.append({
                    'file_path': f,
                    'file_idx': file_idx,
                    'n_cells': n_cells,
                    'n_genes': n_genes,
                    'start_idx': cumulative_cells,
                    'end_idx': cumulative_cells + n_cells
                })
                
                cumulative_cells += n_cells
                self.total_cells += n_cells
        
        # Create global index to file mapping
        self.index_to_file = {}
        for info in self.file_info:
            for i in range(info['start_idx'], info['end_idx']):
                local_idx = i - info['start_idx']
                self.index_to_file[i] = (info['file_path'], local_idx, info['file_idx'])
    
    def _setup_cell_types(self, num_cell_types):
        """Setup cell type encoding by scanning all files."""
        unique_cell_types = set()
        
        # Collect all unique cell types
        for file_info in self.file_info:
            with h5py.File(file_info['file_path'], 'r') as h5:
                cell_types = [x.decode('utf-8') for x in h5['cell_type'][:]]
                unique_cell_types.update(cell_types)
        
        # Process and normalize cell types
        self._process_cell_types(unique_cell_types, num_cell_types)
        
        logging.info(f"[Rank {self.rank}] Found {len(unique_cell_types)} unique cell types")
    
    def _process_cell_types(self, unique_cell_types, num_cell_types):
        """Process and normalize cell types."""
        norm_map = defaultdict(list)
        for ct in unique_cell_types:
            norm_map[self._normalize_cell_type(ct)].append(ct)
            
        self.merged_cell_types = {norm: max(names, key=len) for norm, names in norm_map.items()}
        canonical_cell_types = sorted(set(self.merged_cell_types.values()))
        self.cell_type_labels = canonical_cell_types + ['unknown'] * (num_cell_types - len(canonical_cell_types))
        self.cell_type_encoder.fit(self.cell_type_labels)
        self.cell_type_mapping = {ct: i for i, ct in enumerate(canonical_cell_types)}
    
    def _normalize_cell_type(self, ct):
        """Normalize cell type names."""
        ct = ct.lower()
        ct = re.sub(r'[^\w\s]', '', ct)
        ct = re.sub(r'\s+', ' ', ct)
        return ct.replace('human', '').strip()
    
    def _estimate_memory(self):
        """Estimate memory usage for streaming (much lower than loading all)."""
        # Only metadata and file handles, not the actual data
        return len(self.file_info) * 0.001  # ~1MB per file for metadata
    
    # Enhanced cache for frequently accessed files to improve performance
    _file_cache = {}
    _data_cache = {}  # Cache for actual data chunks
    _cache_size = 4  # Keep 4 files open at most
    _data_cache_size = 1000  # Cache up to 1000 data items
    
    def _get_file_handle(self, file_path):
        """Get file handle with enhanced LRU cache."""
        if file_path in self._file_cache:
            # Move to end (most recently used)
            handle = self._file_cache.pop(file_path)
            self._file_cache[file_path] = handle
            return handle
        
        # Clean cache if too large
        if len(self._file_cache) >= self._cache_size:
            # Remove oldest entry (first item)
            oldest_file = next(iter(self._file_cache))
            self._file_cache[oldest_file].close()
            del self._file_cache[oldest_file]
        
        # Open new file
        self._file_cache[file_path] = h5py.File(file_path, 'r')
        return self._file_cache[file_path]
    
    def _cache_data_item(self, idx, data):
        """Cache data item with LRU eviction."""
        if len(self._data_cache) >= self._data_cache_size:
            # Remove oldest cached item
            oldest_idx = next(iter(self._data_cache))
            del self._data_cache[oldest_idx]
        
        self._data_cache[idx] = data
    
    def _get_cached_data(self, idx):
        """Get cached data if available."""
        if idx in self._data_cache:
            # Move to end (most recently used)
            data = self._data_cache.pop(idx)
            self._data_cache[idx] = data
            return data
        return None
    
    def __len__(self):
        return self.total_cells
    
    def __getitem__(self, idx):
        """Load single cell data on-demand with caching."""
        if idx >= self.total_cells:
            raise IndexError(f"Index {idx} out of range (total: {self.total_cells})")
        
        # Check cache first
        cached_data = self._get_cached_data(idx)
        if cached_data is not None:
            return cached_data
        
        # Get file and local index
        file_path, local_idx, file_idx = self.index_to_file[idx]
        
        # Load data from file
        h5 = self._get_file_handle(file_path)
        
        try:
            # Load single cell data
            binned_expr = h5['binned_expr'][local_idx].astype(np.int32)
            expr_cont = h5['expr_cont'][local_idx].astype(np.float32)
            non_zero_mask = h5['non_zero_mask'][local_idx].astype(np.uint8)
            study_id = h5['study_ids'][local_idx].astype(np.int32)
            
            # Process cell type
            raw_cell_type = h5['cell_type'][local_idx].decode('utf-8')
            normed_cell_type = self._normalize_cell_type(raw_cell_type)
            mapped_cell_type = self.merged_cell_types.get(normed_cell_type, 'unknown')
            cell_type_idx = self.cell_type_mapping.get(mapped_cell_type, len(self.cell_type_mapping))
            
            data = {
                'binned_expr': torch.from_numpy(binned_expr).long(),
                'expr_cont': torch.from_numpy(expr_cont).float(),
                'non_zero_mask': torch.from_numpy(non_zero_mask).to(torch.uint8),
                'study_id': torch.tensor(study_id, dtype=torch.long),
                'cell_type': torch.tensor(cell_type_idx, dtype=torch.long),
                'global_idx': torch.tensor(idx, dtype=torch.long)
            }
            
            # Cache the data for future use
            self._cache_data_item(idx, data)
            return data
            
        except Exception as e:
            logging.error(f"Error loading cell {idx} from {file_path}[{local_idx}]: {e}")
            raise
    
    def __del__(self):
        """Clean up file handles."""
        for f in self._file_cache.values():
            try:
                f.close()
            except:
                pass

class BatchedStreamingDataset(Dataset):
    """Batched streaming dataset for even better memory efficiency."""
    
    def __init__(self, data_dir, batch_size=1000, rank=0, num_cell_types=300):
        self.base_dataset = StreamingSingleCellDataset(data_dir, rank, num_cell_types)
        self.batch_size = batch_size
        self.num_batches = (len(self.base_dataset) + batch_size - 1) // batch_size
        
        logging.info(f"[Rank {rank}] BatchedStreaming: {self.num_batches} batches of size {batch_size}")
    
    def __len__(self):
        return self.num_batches
    
    def __getitem__(self, batch_idx):
        """Return a batch of cells."""
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.base_dataset))
        
        batch_data = []
        for i in range(start_idx, end_idx):
            batch_data.append(self.base_dataset[i])
        
        return custom_collate_fn(batch_data)

# Compatibility class that chooses the right dataset based on data size
class SingleCellDataset(Dataset):
    """Adaptive dataset that chooses streaming vs in-memory based on data size."""
    
    def __init__(self, data_dir, rank=0, num_cell_types=300, streaming_threshold_gb=50):
        self.data_files = sorted(glob.glob(os.path.join(data_dir, 'preprocessed_study*.h5')))
        if not self.data_files:
            raise ValueError(f"No HDF5 files found in {data_dir}")
        
        # Estimate dataset size
        total_size_gb = self._estimate_dataset_size()
        
        logging.info(f"[Rank {rank}] Estimated dataset size: {total_size_gb:.2f} GB")
        
        if total_size_gb > streaming_threshold_gb:
            logging.info(f"[Rank {rank}] Using streaming dataset (size > {streaming_threshold_gb}GB)")
            self.dataset = StreamingSingleCellDataset(data_dir, rank, num_cell_types)
            self.is_streaming = True
        else:
            logging.info(f"[Rank {rank}] Using in-memory dataset (size <= {streaming_threshold_gb}GB)")
            # Import original dataset
            from utils.data import SingleCellDataset as OriginalDataset
            self.dataset = OriginalDataset(data_dir, rank=rank, num_cell_types=num_cell_types)
            self.is_streaming = False
        
        # Delegate attributes
        self.num_cell_types = self.dataset.num_cell_types
    
    def _estimate_dataset_size(self):
        """Estimate dataset size in GB."""
        total_cells = 0
        total_genes = 0
        
        for f in self.data_files:
            with h5py.File(f, 'r') as h5:
                n_cells = h5['binned_expr'].shape[0]
                n_genes = h5['binned_expr'].shape[1]
                total_cells += n_cells
                total_genes = max(total_genes, n_genes)
        
        # Estimate: int32 binned_expr + float32 expr_cont + uint8 non_zero_mask + metadata
        bytes_per_cell = total_genes * (4 + 4 + 1) + 20  # ~9 bytes per gene + metadata
        total_bytes = total_cells * bytes_per_cell
        return total_bytes / (1024**3)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]