import argparse
import os
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
import numpy as np
import scanpy as sc
import anndata
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from datetime import datetime
from tqdm import tqdm

from models.bioformer_with_ffn_moe import BioFormerMoE as BioFormer

from utils.data import SingleCellTestDataset, custom_collate_fn
from utils.preprocessing import setup_logging
from utils.visualization import plot_umap
from utils.metrics import compute_graph_connectivity


def evaluate(rank, world_size, args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29400"
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    setup_logging(rank, args.output_dir)
    dataset = SingleCellTestDataset(args.data_dir, args.selected_genes_file, rank, num_cell_types=199)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False, collate_fn=custom_collate_fn)

    model = BioFormer(
        vocab_size=args.vocab_size,
        num_cell_types=199,
        num_studies=args.num_studies,
        num_bins=args.num_bins,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_experts=args.num_experts
    ).to(device)

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    model.eval()

    embeddings, cell_types, study_ids = [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            binned_expr = batch["binned_expr"].to(device)
            non_zero_mask = batch["non_zero_mask"].to(device)
            cell_type = batch["cell_type"].to(device)
            study_id = batch["study_id"].to(device)

            with autocast():
                output_tuple = model(binned_expr, cell_type, study_id, non_zero_mask)
                if len(output_tuple) == 4:
                    _, _, output, _ = output_tuple
                else:
                    _, _, output = output_tuple

            embeddings.append(output[:, 0].cpu())  # CLS token
            cell_types.append(cell_type.cpu())
            study_ids.append(study_id.cpu())

    embeddings = torch.cat(embeddings)
    cell_types = torch.cat(cell_types)
    study_ids = torch.cat(study_ids)

    if rank == 0:
        adata = anndata.AnnData(X=embeddings.numpy())
        adata.obs["cell_type"] = LabelEncoder().fit_transform(cell_types.numpy())
        adata.obs["study_id"] = study_ids.numpy().astype(str)
        sc.pp.neighbors(adata, use_rep="X")
        sc.tl.leiden(adata, resolution=1.0)
        result_dir = os.path.join(args.output_dir, "test_results")
        os.makedirs(result_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_umap(embeddings.numpy(), adata.obs["cell_type"], adata.obs["study_id"], 'study_ids', 'cell_types', result_dir, timestamp, 'MoE UMAP')
        plot_umap(embeddings.numpy(), adata.obs["cell_type"], adata.obs["study_id"], 'cell_types', 'study_ids', result_dir, timestamp, 'MoE UMAP')

        nmi = normalized_mutual_info_score(adata.obs["cell_type"], adata.obs["leiden"])
        ari = adjusted_rand_score(adata.obs["cell_type"], adata.obs["leiden"])
        sil = silhouette_score(adata.X, adata.obs["cell_type"], metric="cosine")
        conn = compute_graph_connectivity(adata, adata.obs["cell_type"], adata.obs["study_id"])

        with open(os.path.join(result_dir, "metrics.txt"), "w") as f:
            f.write(f"NMI: {nmi:.4f}\n")
            f.write(f"ARI: {ari:.4f}\n")
            f.write(f"Silhouette: {sil:.4f}\n")
            f.write(f"GraphConn: {conn:.4f}\n")

    if world_size > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/home/tripham/scgpt/test_clustering/data")
    parser.add_argument('--selected_genes_file', type=str, default="/home/tripham/scgpt/preprocessed/selected_genes.txt")
    parser.add_argument('--checkpoint_path', type=str, default="/home/tripham/scgpt/training1/checkpoints/checkpoint_epoch_3_20250616_014544.pt")
    parser.add_argument('--output_dir', type=str, default="result")
    parser.add_argument('--vocab_size', type=int, default=1000)
    parser.add_argument('--num_studies', type=int, default=15)
    parser.add_argument('--num_bins', type=int, default=51)
    parser.add_argument('--d_model', type=int, default=384)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--no-ddp', action='store_true')
    args = parser.parse_args()

    if args.no_ddp:
        evaluate(0, 1, args)
    else:
        mp.spawn(evaluate, args=(torch.cuda.device_count(), args), nprocs=torch.cuda.device_count())


if __name__ == "__main__":
    main()
