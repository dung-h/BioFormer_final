import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging
import scanpy as sc
from models.bioformer_prompt import BioFormerPrompt
from utils.data import SingleCellDataset, custom_collate_fn
from utils.losses import ecs_loss
from utils.preprocessing import setup_logging, cpu_quantile_binning

def load_model_from_checkpoint(model, checkpoint_path):
    """Loads model state_dict from a checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model checkpoint from {checkpoint_path}")
    return model

def train_perturbation(args):
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(0, args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize BioFormerPrompt model
    model = BioFormerPrompt(
        vocab_size=args.vocab_size,
        num_cell_types=args.num_cell_types,
        num_bins=args.num_bins,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_perturbations=args.num_perturbations
    ).to(device)

    # Load pretrained weights from the base model
    if args.pretrained_checkpoint:
        model = load_model_from_checkpoint(model, args.pretrained_checkpoint)

    dataset = SingleCellDataset(
        args.data_dir,
        selected_genes_file=args.selected_genes_file,
        rank=0,
        num_cell_types=args.num_cell_types
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=2,
        pin_memory=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            binned_expr = batch['binned_expr'].to(device)
            non_zero_mask = batch['non_zero_mask'].to(device)
            cell_type = batch['cell_type'].to(device)
            perturb_idx = batch['perturb_idx'].to(device)  # Ensure perturbation token is included

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                _, _, embeddings = model(binned_expr, cell_type, perturb_idx, non_zero_mask)
                loss = ecs_loss(embeddings, cell_type, margin=args.ecs_margin)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

        ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, ckpt_path)
        logging.info(f"Saved checkpoint to {ckpt_path}")
