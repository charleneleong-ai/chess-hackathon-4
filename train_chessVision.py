from cycling_utils import TimestampedTimer
from torch.utils.tensorboard import SummaryWriter

timer = TimestampedTimer("Imported TimestampedTimer")

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import ConcatDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import argparse
import os
import socket
import yaml
import time
import math

from cycling_utils import (
    InterruptableDistributedSampler,
    MetricsTracker,
    AtomicDirectory,
    atomic_torch_save,
)

from utils.optimizers import Lamb, Lookahead
from utils.datasets import EVAL_HDF_Dataset
from model import Model
import numpy as np
import random
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler

torch.backends.cudnn.benchmark = True ## input dims not changing; set for faster conv

scaler = GradScaler()

SEED = 42
MAX_EPOCHS = 10_000

timer.report("Completed imports")

def set_all_seeds(seed: int = 42) -> None:
    """
    Set the random seed for reproducibility.
    """
    print(f"Random seed set as {seed}")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", help="model config path", type=Path, default="/root/chess-hackathon-4/model_config.yaml")
    parser.add_argument("--save-dir", help="save checkpoint path", type=Path, default=os.getenv("OUTPUT_PATH", ".logs"))
    parser.add_argument("--load-path", help="path to checkpoint.pt file to resume from", type=Path, default='checkpoint.pt')
    parser.add_argument("--bs", help="batch size", type=int, default=32)
    parser.add_argument("--lr", help="learning rate", type=float, default=0.001)
    parser.add_argument("--max-lr", help="max learning rate", type=float, default=0.01)
    parser.add_argument("--wd", help="weight decay", type=float, default=0.01)
    parser.add_argument("--ws", help="learning rate warm up steps", type=int, default=1000)
    parser.add_argument("--grad-accum", help="gradient accumulation steps", type=int, default=4)
    parser.add_argument("--save-steps", help="saving interval steps", type=int, default=500)
    parser.add_argument("--distributed", action="store_false", help="whether to use distributed training") ## store_false=True
    parser.add_argument("--patience", help="patience of early stopping", type=int, default=10)
    parser.add_argument("--amp", action="store_true", help="Mixed precision") ## store_false=True
    return parser

def logish_transform(data):
    '''Zero-symmetric log-transformation.'''
    reflector = -1 * (data < 0).to(torch.int8)
    return reflector * torch.log(torch.abs(data) + 1)

def spearmans_rho(seq_a, seq_b):
    '''Spearman's rank correlation coefficient'''
    assert len(seq_a) == len(seq_b), "ERROR: Sortables must be equal length."
    rank_a = torch.argsort(torch.argsort(seq_a))
    rank_b = torch.argsort(torch.argsort(seq_b))
    d = rank_a - rank_b    
    n = seq_a.size(0)
    return 1 - (6 * torch.sum(d**2)) / (n * (n**2 - 1))


def early_stopping(val_losses, patience):
    """Checks for early stopping"""
    if len(val_losses) > patience and min(val_losses[-patience:]) >= min(val_losses):
        return True
    return False

def main(args, timer):
    set_all_seeds(SEED)
    # dist.init_process_group("nccl")  # Expects RANK set in environment variable
    # rank = int(os.getenv("RANK", 0))  # Rank of this GPU in cluster
    # world_size = int(os.getenv("WORLD_SIZE", 1)) # Total number of GPUs in the cluster
    # args.device_id = int(os.getenv("LOCAL_RANK", 1))  # Rank on local node
    # args.is_master = rank == 0  # Master node for saving / reporting
    
    print(f"Running in distributed: {args.distributed}")
    if args.distributed:
        dist.init_process_group("nccl")  # or "gloo" for CPU
        # rank = int(os.getenv("RANK", 0))# Rank of this GPU in cluster
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        # world_size = int(os.getenv("WORLD_SIZE", 1)) # Total number of GPUs in the cluster
        
        args.device_id = int(os.getenv("LOCAL_RANK", 0)) # Rank on local node
        args.is_master = rank == 0  # Master node for saving / reporting
    else:
        args.device_id = 0 if torch.cuda.is_available() else -1  # Use the first available GPU or CPU
        args.is_master = True
    
    device = torch.device(f"cuda:{args.device_id}" if args.device_id >= 0 else "cpu")
    
    if args.is_master and args.save_dir:
        writer= SummaryWriter(log_dir=args.save_dir / "logs")
        
    if args.device_id >= 0:
        torch.cuda.set_device(args.device_id)  # Enables calling 'cuda'
        torch.autograd.set_detect_anomaly(True) 

    if args.device_id >= 0:
        hostname = socket.gethostname()
        print("Hostname:", hostname)
        print(f"TrainConfig: {args}")
    timer.report("Setup for distributed training")

    saver = AtomicDirectory(args.save_dir)
    timer.report("Validated checkpoint path")

    data_path = "/data"
    # dataset = EVAL_HDF_Dataset(f"{data_path}/gm")
    gm_dataset = EVAL_HDF_Dataset(f"{data_path}/gm")
    lc0 = EVAL_HDF_Dataset(f"{data_path}/lc0")
    dataset = ConcatDataset([gm_dataset, lc0])
    
    # Determine number of workers dynamically based on available CPU cores
    num_gpus = world_size
    cpu_count = mp.cpu_count()  # Total number of available CPU cores
    ## num_workers_per_gpu = cpu_count // num_gpus
    num_workers_per_gpu = 32

    print(f"Number of workers per GPU: {num_workers_per_gpu}, Total workers: {num_workers_per_gpu * num_gpus}")

    timer.report(f"Intitialized dataset with {len(dataset):,} Board Evaluations.")

    random_generator = torch.Generator().manual_seed(SEED)
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2], generator=random_generator)

    if args.distributed:
        train_sampler = InterruptableDistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=SEED, )
        test_sampler = InterruptableDistributedSampler(test_dataset,  num_replicas=world_size, rank=rank, shuffle=False,)
    else:
        train_sampler = None
        test_sampler = None

    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, sampler=train_sampler, num_workers=num_workers_per_gpu, pin_memory=True, prefetch_factor=30, persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.bs, sampler=test_sampler, num_workers=num_workers_per_gpu, pin_memory=True, prefetch_factor=30, persistent_workers=True)
    timer.report("Prepared dataloaders")

    model_config = yaml.safe_load(open(args.model_config))
    if args.device_id >= 0:
        print(f"ModelConfig: {model_config}")
    model_config["device"] = 'cuda'
    model = Model(**model_config)
    model = model.to(args.device_id)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
    timer.report(f"Initialized model with {params:,} params, moved to device")

    if args.distributed:
        model = DDP(model, device_ids=[args.device_id], find_unused_parameters=False)
    timer.report("Prepared model for distributed training")

    loss_fn = nn.MSELoss()
    optimizer = Lamb(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=args.patience)
    optimizer = Lookahead(optimizer, k=10, alpha=0.3)  # Lookahead wrapper
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-2, step_size_up=2000, mode='triangular2')
    scheduler =  torch.optim.lr_scheduler.OneCycleLR(
        optimizer.optimizer if isinstance(optimizer, Lookahead) else optimizer, 
        max_lr=args.max_lr, 
        steps_per_epoch=len(train_dataloader), 
        epochs=MAX_EPOCHS,
        cycle_momentum=False ## Disable momentum cycling for Lamb
    )

    metrics = {"train": MetricsTracker(), "test": MetricsTracker()}
    val_losses = []
    best_val_loss = float('inf')
    no_improve_count = 0


    checkpoint_path = None
    local_resume_path = os.path.join(args.save_dir, saver.symlink_name)
    print(local_resume_path)
    if os.path.islink(local_resume_path):
        checkpoint_path = os.path.join(os.readlink(local_resume_path), "checkpoint.pt")
    elif args.load_path:
        if os.path.isfile(args.load_path):
            checkpoint_path = args.load_path
    if checkpoint_path:
        if args.is_master:
            timer.report(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{args.device_id}")
        model.module.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        ## Adding scheduler after checkpoint
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            print("Scheduler state not found in checkpoint. Creating new scheduler.")
            for param_group in optimizer.optimizer.param_groups:
                if 'initial_lr' not in param_group:
                    param_group['initial_lr'] = param_group['lr']
                if 'max_lr' not in param_group:
                    param_group['max_lr'] = args.max_lr

        train_dataloader.sampler.load_state_dict(checkpoint["train_sampler"])
        test_dataloader.sampler.load_state_dict(checkpoint["test_sampler"])
        metrics = checkpoint["metrics"]
        timer = checkpoint["timer"]
        timer.start_time = time.time()
        timer.report("Retrieved saved checkpoint")
    
    for epoch in range(train_dataloader.sampler.epoch, MAX_EPOCHS):
        with train_dataloader.sampler.in_epoch(epoch):
            timer.report(f"Training epoch {epoch}")
            train_batches_per_epoch = len(train_dataloader)
            train_steps_per_epoch = math.ceil(train_batches_per_epoch / args.grad_accum)
            optimizer.zero_grad()
            model.train()

            for boards, scores in train_dataloader:

                # Determine the current step
                batch = train_dataloader.sampler.progress // train_dataloader.batch_size
                is_save_batch = (batch + 1) % args.save_steps == 0
                is_accum_batch = (batch + 1) % args.grad_accum == 0
                is_last_batch = (batch + 1) == train_batches_per_epoch

                # Prepare checkpoint directory
                if (is_save_batch or is_last_batch) and args.is_master:
                    checkpoint_directory = saver.prepare_checkpoint_directory()

                scores = logish_transform(scores) # suspect this might help
                boards, scores = boards.to(args.device_id, non_blocking=True), scores.to(args.device_id, non_blocking=True)

                ## Mixed precision
                if args.amp:
                    with autocast():
                        logits = model(boards)
                        loss = loss_fn(logits, scores)
                        loss = loss / args.grad_accum

                    scaler.scale(loss).backward()
                else:
                    logits = model(boards)
                    loss = loss_fn(logits, scores)
                    loss = loss / args.grad_accum
                    loss.backward()
                
                ## Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2.0)
                
                train_dataloader.sampler.advance(len(scores))

                # How accurately do our model scores rank the batch of moves? 
                rank_corr = spearmans_rho(logits, scores)
                
                metrics["train"].update({
                    "examples_seen": len(scores),
                    "accum_loss": loss.item() * args.grad_accum, 
                    "rank_corr": rank_corr
                })

                if is_accum_batch or is_last_batch:
                    if args.amp:
                        scaler.step(optimizer) ## Scaler step optimizer
                        scaler.update()
                    else:
                        optimizer.step()
                        
                    scheduler.step()
                    optimizer.zero_grad()
                    step = batch // args.grad_accum
                    
                    # Log memory utilization
                    # print(torch.cuda.memory_summary(device=args.device_id))
                    
                    # learning rate warmup
                    lr_factor = min((epoch + 1) * step / args.ws, 1)
                    for g in optimizer.param_groups:
                        g['lr'] = lr_factor * args.lr
                        
                    
                    metrics["train"].reduce()
                    rpt = metrics["train"].local
                    avg_loss = rpt["accum_loss"] / rpt["examples_seen"]
                    rpt_rank_corr = 100 * rpt["rank_corr"] / rpt["examples_seen"]
                    current_lr = optimizer.param_groups[0]['lr']
                    report = f"""\
Epoch [{epoch:,}] Step [{step:,} / {train_steps_per_epoch:,}] Batch [{batch:,} / {train_batches_per_epoch:,}] Lr: [{current_lr:,.3}], \
Avg Loss [{avg_loss:,.3f}], Rank Corr.: [{rpt_rank_corr:,.3f}%], Examples: {rpt['examples_seen']:,.0f}"""
                    timer.report(report)
                    metrics["train"].reset_local()
                
                    if args.is_master:
                        global_step = epoch * train_steps_per_epoch + step
                        writer.add_scalar('Train/Loss', avg_loss, global_step)
                        writer.add_scalar('Train/Rank_Corr', rpt_rank_corr, global_step)
                        
                        # current_lr = optimizer.param_groups[0]['lr']
                        
                
                del boards, scores, logits
                torch.cuda.empty_cache()  ## Clear GPU memory

                # Saving
                if (is_save_batch or is_last_batch) and args.is_master:
                    # Save checkpoint
                    atomic_torch_save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), 
                            "train_sampler": train_dataloader.sampler.state_dict(),
                            "test_sampler": test_dataloader.sampler.state_dict(),
                            "metrics": metrics,
                            "timer": timer
                        },
                        os.path.join(checkpoint_directory, "checkpoint.pt"),
                    )
                    saver.atomic_symlink(checkpoint_directory)

            val_loss = 0.0
            with test_dataloader.sampler.in_epoch(epoch):
                timer.report(f"Testing epoch {epoch}")
                test_batches_per_epoch = len(test_dataloader)
                model.eval()

                with torch.no_grad():
                    for boards, scores in test_dataloader:

                        # Determine the current step
                        batch = test_dataloader.sampler.progress // test_dataloader.batch_size
                        is_save_batch = (batch + 1) % args.save_steps == 0
                        is_last_batch = (batch + 1) == test_batches_per_epoch

                        # Prepare checkpoint directory
                        if (is_save_batch or is_last_batch) and args.is_master:
                            checkpoint_directory = saver.prepare_checkpoint_directory()

                        scores = logish_transform(scores) # suspect this might help
                        boards, scores = boards.to(args.device_id, non_blocking=True), scores.to(args.device_id, non_blocking=True)

                        if args.amp:
                            with autocast():
                                logits = model(boards)
                                loss = loss_fn(logits, scores)
                        else:
                            logits = model(boards)
                            loss = loss_fn(logits, scores)
                            
                        val_loss += loss.item()
                            
                        test_dataloader.sampler.advance(len(scores))

                        # How accurately do our model scores rank the batch of moves? 
                        rank_corr = spearmans_rho(logits, scores)

                        metrics["test"].update({
                            "examples_seen": len(scores),
                            "accum_loss": loss.item() * args.grad_accum, 
                            "rank_corr": rank_corr
                        })
                        
                        # Reporting
                        if is_last_batch:
                            metrics["test"].reduce()
                            rpt = metrics["test"].local
                            avg_loss = rpt["accum_loss"] / rpt["examples_seen"]
                            rpt_rank_corr = 100 * rpt["rank_corr"] / rpt["examples_seen"]
                            report = f"Epoch [{epoch}] Evaluation, Avg Loss [{avg_loss:,.3f}], Rank Corr. [{rpt_rank_corr:,.3f}%]"
                            timer.report(report)
                            

                            # Step the learning rate scheduler based on validation loss
                            # scheduler.step(val_loss)
                            
                            val_loss /= len(test_dataloader)
                            val_losses.append(val_loss)
                            
                            if args.is_master:
                                writer.add_scalar('Validation/Loss', val_loss, epoch)
                                writer.add_scalar('Validation/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
                                writer.add_scalar('Validation/Rank_Corr', rpt_rank_corr, epoch)
                            
                            ## Early Stopping
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                no_improve_count = 0
                            else:
                                no_improve_count += 1
                            if early_stopping(val_losses, args.patience):
                                print(f"Early stopping at epoch {epoch} step {global_step}")
                                break
                        del boards, scores, logits
                        torch.cuda.empty_cache()  ## Clear GPU memory

                                
                        # Saving
                        if (is_save_batch or is_last_batch) and args.is_master:
                            # Save checkpoint
                            atomic_torch_save(
                                {
                                    "model": model.module.state_dict(),
                                    "optimizer": optimizer.state_dict(),
                                    "scheduler": scheduler.state_dict(), 
                                    "train_sampler": train_dataloader.sampler.state_dict(),
                                    "test_sampler": test_dataloader.sampler.state_dict(),
                                    "metrics": metrics,
                                    "timer": timer
                                },
                                os.path.join(checkpoint_directory, "checkpoint.pt"),
                            )
                            saver.atomic_symlink(checkpoint_directory)
                            
                    if args.is_master:
                        writer.flush()
        if args.is_master:
            writer.close()
                
timer.report("Defined functions")
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args, timer)
