"""Distributed training for fine-tuned BERT reranker using PyTorch DDP.

This script demonstrates multi-GPU training with DistributedDataParallel.
Single-GPU training: use train_finetuned.py instead.

Usage (single-node, 2 GPUs):
    torchrun --nproc_per_node=2 -m helix.train_distributed --epochs 3

Usage (multi-node, 2 nodes x 2 GPUs each):
    # Node 0
    torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 \
        --master_addr=NODE0_IP --master_port=29500 \
        -m helix.train_distributed --epochs 3
    
    # Node 1
    torchrun --nproc_per_node=2 --nnodes=2 --node_rank=1 \
        --master_addr=NODE0_IP --master_port=29500 \
        -m helix.train_distributed --epochs 3

Output:
    helix/finetuned_reranker_ddp.pt (model weights, rank 0 only, not committed)
    helix/finetuned_reranker_ddp.meta.json (metadata + metrics, committed)

Requirements:
    pip install torch transformers datasets

Hardware:
    2+ GPUs recommended. RTX 2060 (6GB) works but single-GPU is faster for small datasets.
    Training time: ~3-5 minutes on 2x RTX 3090, ~5-10 minutes on 2x RTX 2060.

DDP notes:
    - Each process loads the full dataset, but the DataLoader sampler shards it.
    - Gradients are averaged across GPUs via all-reduce.
    - Only rank 0 saves the model and metadata.
    - Use torchrun (not python) to launch. It sets up the process group.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from helix.corpus import CORPUS
from helix.reranker import QRELS_PATH

MODEL_PATH = Path(__file__).resolve().parent / "helix" / "finetuned_reranker_ddp.pt"
META_PATH = Path(__file__).resolve().parent / "helix" / "finetuned_reranker_ddp.meta.json"


def load_qrels() -> list[dict]:
    """Load qrels and join with corpus to get (query, title, body, rel) tuples."""
    qrels = []
    for line in QRELS_PATH.read_text().strip().splitlines():
        qrels.append(json.loads(line))
    
    doc_map = {d.id: d for d in CORPUS}
    rows = []
    for q in qrels:
        doc = doc_map.get(q["docId"])
        if doc is None:
            continue
        rows.append({
            "query": q["query"],
            "title": doc.title,
            "body": doc.body,
            "rel": q["rel"],
            "split": q["split"],
        })
    return rows


def train(args: argparse.Namespace) -> dict:
    """Fine-tune BERT with DDP. Returns metrics dict."""
    try:
        import torch
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.utils.data import Dataset, DataLoader
        from torch.utils.data.distributed import DistributedSampler
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as e:
        raise RuntimeError("PyTorch and transformers required: pip install torch transformers") from e
    
    # Initialize DDP
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    if rank == 0:
        print(f"DDP initialized: world_size={world_size}, backend=nccl")
    
    rows = load_qrels()
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "test"]
    
    if rank == 0:
        print(f"Train: {len(train_rows)}, Val: {len(val_rows)}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=1)
    model.to(device)
    model = DDP(model, device_ids=[local_rank])
    
    class RerankDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows
        
        def __len__(self):
            return len(self.rows)
        
        def __getitem__(self, idx):
            r = self.rows[idx]
            text = f"{r['query']} [SEP] {r['title']} {r['body']}"
            enc = tokenizer(text, truncation=True, max_length=256, return_tensors="pt")
            return {
                "input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "label": torch.tensor(float(r["rel"]) / 3.0, dtype=torch.float32),
            }
    
    train_ds = RerankDataset(train_rows)
    val_ds = RerankDataset(val_rows)
    
    # DistributedSampler shards the dataset across GPUs
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    
    # Batch size is per-GPU. Effective batch size = batch_size * world_size
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fn = torch.nn.MSELoss()
    
    start = time.time()
    best_val_loss = float("inf")
    
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)  # Shuffle differently each epoch
        
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze()
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze()
                loss = loss_fn(logits, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save model state dict (not the DDP wrapper)
                torch.save(model.module.state_dict(), MODEL_PATH)
                print(f"  Saved best model (val_loss={val_loss:.4f})")
    
    elapsed = time.time() - start
    dist.destroy_process_group()
    
    return {
        "base_model": args.base_model,
        "epochs": args.epochs,
        "batch_size_per_gpu": args.batch_size,
        "effective_batch_size": args.batch_size * world_size,
        "lr": args.lr,
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "best_val_loss": float(best_val_loss),
        "training_time_sec": float(elapsed),
        "world_size": world_size,
        "backend": "nccl",
        "seed": args.seed,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BERT reranker with DDP")
    parser.add_argument("--base-model", default="distilbert-base-uncased", help="HF model name")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
    except ImportError:
        pass
    
    metrics = train(args)
    
    # Only rank 0 saves metadata
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        META_PATH.write_text(json.dumps(metrics, indent=2))
        print(f"\nMetadata saved to {META_PATH}")
        print(f"Model saved to {MODEL_PATH}")
        print(f"Best val loss: {metrics['best_val_loss']:.4f}")
        print(f"Training time: {metrics['training_time_sec']:.1f}s")
        print(f"World size: {metrics['world_size']} GPUs")


if __name__ == "__main__":
    main()
