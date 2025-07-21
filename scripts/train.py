
import os
import argparse
import logging
import time
from datetime import datetime, timedelta

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

class ResumableDistributedSampler(DistributedSampler):
    """DistributedSampler that can resume from a specific step without iteration."""
    
    def __init__(self, *args, start_step=0, gradient_accumulation_steps=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_step = start_step
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
    def __iter__(self):
        indices = list(super().__iter__())
        
        # Calculate how many samples to skip based on start_step
        if self.start_step > 0:
            batches_per_step = self.gradient_accumulation_steps
            samples_per_step = batches_per_step * self.num_replicas  # account for all GPUs
            skip_samples = (self.start_step * samples_per_step) % len(indices)
            
            # Skip the samples that were already processed
            indices = indices[skip_samples:]
            
        return iter(indices)
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import math

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bioformer_with_ffn_moe import BioFormerMoE as BioFormer
from utils.data_streaming import SingleCellDataset, custom_collate_fn
from utils.losses import ecs_loss
from utils.preprocessing import setup_logging

def setup_distributed_env(rank):
    os.environ["NCCL_TIMEOUT"] = "172800000"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29700"
    os.environ["NCCL_SOCKET_IFNAME"] = "lo"
    os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def get_warmup_cosine_schedule(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    """Warmup + Cosine decay scheduler"""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)

def initialize_model(args, device, num_cell_types):
    model = BioFormer(
        vocab_size=args.vocab_size,
        num_studies=args.num_studies,
        num_cell_types=args.num_cell_types,
        num_bins=args.num_bins,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_experts=args.num_experts
    ).to(device)
    return model

def load_checkpoint(model, optimizer, scheduler, args, device, rank):
    start_epoch = 0
    start_step = 0
    
    # Try to load from specified checkpoint or latest
    checkpoint_path = args.resume_from_checkpoint
    if not checkpoint_path:
        latest_path = os.path.join(args.output_dir, 'checkpoints', 'latest_checkpoint.pt')
        if os.path.exists(latest_path):
            checkpoint_path = latest_path
    
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return start_epoch, start_step
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Handle both DDP and non-DDP models
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0)
    start_step = checkpoint.get('step', 0)
    
    # If checkpoint was saved at end of epoch (no step info), start from next epoch
    if start_step is None:
        start_epoch += 1
        start_step = 0
    
    # Restore scheduler state
    if scheduler and start_step > 0:
        for _ in range(start_step):
            scheduler.step()
    
    logging.info(f"[Rank {rank}] Resumed from epoch {start_epoch}, step {start_step}")
    return start_epoch, start_step

def train_epoch(model, dataloader, optimizer, scheduler, scaler, args, device, rank, epoch, start_step=0):
    model.train()
    total_mlm_loss, total_cont_loss, total_ecs_loss, total_batch_time, total_steps = 0, 0, 0, 0, 0
    accumulation_steps = 0
    global_step = start_step if start_step is not None else 0
    
    # Sampler handles resuming, so we can iterate normally
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"[Rank {rank}] Epoch {epoch+1}/{args.epochs}", disable=(rank != 0))):
        start_time = time.time()

        binned_expr = batch['binned_expr'].to(device, non_blocking=True)
        expr_cont = batch['expr_cont'].to(device, non_blocking=True)
        non_zero_mask = batch['non_zero_mask'].to(device, non_blocking=True)
        cell_type = batch['cell_type'].to(device, non_blocking=True)

        mask = torch.rand_like(binned_expr.float()) < 0.15
        effective_mask = mask & (binned_expr > 0)
        masked_input = binned_expr.clone()
        masked_input[effective_mask] = 0

        with autocast():
            mlm_logits, cont_pred, output, route_weights = model(
                masked_input, cell_type=cell_type, study_id=None,
                non_zero_mask=non_zero_mask, return_attention=False
            )

            mlm_loss = F.cross_entropy(
                mlm_logits.view(-1, mlm_logits.size(-1))[effective_mask.view(-1)],
                binned_expr.view(-1)[effective_mask.view(-1)]
            )
            cont_loss = F.mse_loss(cont_pred, expr_cont, reduction='none')
            cont_loss = (cont_loss * non_zero_mask).sum() / (non_zero_mask.sum() + 1e-8)
            ec_loss = ecs_loss(output, cell_type, margin=args.ecs_margin)

            total_loss = (
                args.mlm_weight * mlm_loss +
                args.cont_weight * cont_loss +
                args.ecs_weight * ec_loss
            ) / args.gradient_accumulation_steps

        scaler.scale(total_loss).backward()
        accumulation_steps += 1
        
        if accumulation_steps >= args.gradient_accumulation_steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accumulation_steps = 0
            global_step += 1
            
            # Save checkpoint every N steps
            if global_step % args.checkpoint_every == 0:
                metrics = {
                    'mlm_loss': total_mlm_loss / total_steps if total_steps > 0 else 0,
                    'cont_loss': total_cont_loss / total_steps if total_steps > 0 else 0,
                    'ecs_loss': total_ecs_loss / total_steps if total_steps > 0 else 0,
                    'batch_time': total_batch_time / total_steps if total_steps > 0 else 0
                }
                save_checkpoint(model, optimizer, epoch, args, rank, metrics, step=global_step)

        total_mlm_loss += mlm_loss.item()
        total_cont_loss += cont_loss.item()
        total_ecs_loss += ec_loss.item()
        total_steps += 1
        total_batch_time += time.time() - start_time

        if rank == 0 and total_steps % 50 == 0:
            entropy = -(route_weights * torch.log(route_weights + 1e-8)).sum(dim=-1).mean().item()
            usage = route_weights.mean(dim=(0, 1)).detach().cpu().numpy()
            usage_str = ", ".join([f"E{i}: {u:.3f}" for i, u in enumerate(usage)])
            current_lr = optimizer.param_groups[0]['lr']
            gpu_memory_gb = torch.cuda.memory_allocated() / 1024**3
            import psutil
            ram_memory_gb = psutil.virtual_memory().used / 1024**3
            logging.info(
                f"[Rank {rank}] Global Step {global_step} (Local {total_steps}) | "
                f"MLM: {mlm_loss.item():.4f}, Cont: {cont_loss.item():.4f}, "
                f"ECS: {ec_loss.item():.4f}, LR: {current_lr:.2e}, GPU: {gpu_memory_gb:.1f}GB, RAM: {ram_memory_gb:.1f}GB, "
                f"Time: {total_batch_time/total_steps:.2f}s | "
                f"Routing Entropy: {entropy:.3f} | {usage_str}"
            )

    return {
        'mlm_loss': total_mlm_loss / total_steps,
        'cont_loss': total_cont_loss / total_steps,
        'ecs_loss': total_ecs_loss / total_steps,
        'batch_time': total_batch_time / total_steps
    }

def save_checkpoint(model, optimizer, epoch, args, rank, metrics, step=None):
    if rank != 0:
        return
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if step is not None:
        path = os.path.join(args.output_dir, 'checkpoints', f'checkpoint_step_{step}_{timestamp}.pt')
    else:
        path = os.path.join(args.output_dir, 'checkpoints', f'checkpoint_epoch_{epoch+1}_{timestamp}.pt')
    
    # Handle both DDP and non-DDP models
    if hasattr(model, 'module'):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    checkpoint_data = {
        'epoch': epoch,  # Save current epoch, not epoch+1 for mid-epoch checkpoints
        'step': step,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        **metrics
    }
    torch.save(checkpoint_data, path)
    logging.info(f"[Rank {rank}] Checkpoint saved to {path}")
    
    # Keep latest checkpoint symlink
    latest_path = os.path.join(args.output_dir, 'checkpoints', 'latest_checkpoint.pt')
    if os.path.exists(latest_path):
        os.remove(latest_path)
    os.symlink(os.path.basename(path), latest_path)

def train(rank, world_size, args):
    setup_distributed_env(rank)
    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=172800))
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    setup_logging(rank, args.output_dir)

    dataset = SingleCellDataset(args.data_dir, rank=rank, num_cell_types=args.num_cell_types)

    model = initialize_model(args, device, dataset.num_cell_types)
    
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    
    scaler = GradScaler()
    start_epoch, start_step = load_checkpoint(model, optimizer, None, args, device, rank)
    
    # Disable torch.compile due to CUDA graph conflicts with gradient accumulation
    logging.info(f"[Rank {rank}] torch.compile disabled due to CUDA graph compatibility issues")
    
    # Create sampler and dataloader after loading checkpoint to know start_step
    if world_size > 1:
        sampler = ResumableDistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True,
            start_step=start_step, gradient_accumulation_steps=args.gradient_accumulation_steps
        )
    else:
        sampler = None
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        collate_fn=custom_collate_fn, num_workers=16, pin_memory=True,
        persistent_workers=True, prefetch_factor=16, drop_last=True
    )
    
    # Setup scheduler after creating dataloader
    total_steps = len(dataloader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * 0.1)  # 10% warmup
    scheduler = get_warmup_cosine_schedule(optimizer, warmup_steps, total_steps)
    
    # Restore scheduler state if resuming
    if start_step is not None and start_step > 0:
        for _ in range(start_step):
            scheduler.step()

    for epoch in range(start_epoch, args.epochs):
        # Set sampler epoch for proper distributed shuffling
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        # Skip batches if resuming mid-epoch
        epoch_start_step = start_step if epoch == start_epoch and start_step is not None else 0
        metrics = train_epoch(model, dataloader, optimizer, scheduler, scaler, args, device, rank, epoch, epoch_start_step)
        logging.info(
            f"[Rank {rank}] Epoch {epoch+1}/{args.epochs}: "
            f"MLM Loss={metrics['mlm_loss']:.4f}, "
            f"Cont Loss={metrics['cont_loss']:.4f}, "
            f"ECS Loss={metrics['ecs_loss']:.4f}, "
            f"Batch Time={metrics['batch_time']:.2f}s"
        )
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save_checkpoint(model, optimizer, epoch, args, rank, metrics)

    if world_size > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/mnt/nasdev2/dung/preprocessed_2000")
    parser.add_argument('--output_dir', type=str, default="/mnt/nasdev2/dung/training_2k_256d_8expert")
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--vocab_size', type=int, default=2000)
    parser.add_argument('--num_studies', type=int, default=30)
    parser.add_argument('--num_cell_types', type=int, default=300)
    parser.add_argument('--num_bins', type=int, default=51)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--mlm_weight', type=float, default=1.0)
    parser.add_argument('--cont_weight', type=float, default=0.1)
    parser.add_argument('--ecs_weight', type=float, default=0.1)
    parser.add_argument('--ecs_margin', type=float, default=1.0)
    parser.add_argument('--num_experts', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=10)
    parser.add_argument('--checkpoint_every', type=int, default=5000)
    parser.add_argument('--no-ddp', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )

    if args.no_ddp:
        # Set GPU 2 for single GPU training
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        train(0, 1, args)
    else:
        # Use single GPU 2 for 384d 8-expert training with torch.compile
        available_gpus = [1]  # Use GPU 2 only
        world_size = len(available_gpus)
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, available_gpus))
        mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
