"""Train a fine-tuned BERT reranker on the domain qrels.

This is the third tier in the multi-tier reranking strategy:
    Tier 1: sklearn LogisticRegression (fast, interpretable, CPU)
    Tier 2: HF CrossEncoder pre-trained (accurate, black-box)
    Tier 3: PyTorch BERT fine-tuned on domain data (this script)

Usage:
    cd python
    python -m helix.train_finetuned --base-model distilbert-base-uncased --epochs 3

Output:
    helix/finetuned_reranker.pt (model weights, not committed)
    helix/finetuned_reranker.meta.json (metadata + metrics, committed)

Requirements:
    pip install torch transformers datasets

Hardware:
    RTX 2060 (6GB VRAM) is sufficient for DistilBERT with batch_size=8.
    Training time: ~5-10 minutes on GPU, ~1-2 hours on CPU.

The training data is the same as the sklearn tier (rerank_qrels.jsonl), but the
model learns end-to-end from raw text instead of hand-crafted features. This
captures domain-specific patterns (ticker symbols, financial jargon) that the
feature-based model might miss.

The model artifact is NOT committed to GitHub (too large). Only metadata and
metrics are committed. To reproduce: run this script with the same seed and
hyperparameters.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from helix.corpus import CORPUS
from helix.reranker import QRELS_PATH

MODEL_PATH = Path(__file__).resolve().parent / "finetuned_reranker.pt"
META_PATH = Path(__file__).resolve().parent / "finetuned_reranker.meta.json"


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
    """Fine-tune BERT on the qrels. Returns metrics dict."""
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as e:
        raise RuntimeError("PyTorch and transformers required: pip install torch transformers") from e
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    rows = load_qrels()
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "holdout"]
    
    print(f"Train: {len(train_rows)}, Val: {len(val_rows)}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=1)
    model.to(device)
    
    class RerankDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows
        
        def __len__(self):
            return len(self.rows)
        
        def __getitem__(self, idx):
            r = self.rows[idx]
            text = f"{r['query']} [SEP] {r['title']} {r['body']}"
            enc = tokenizer(text, truncation=True, max_length=256, padding="max_length", return_tensors="pt")
            return {
                "input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "label": torch.tensor(float(r["rel"]) / 3.0, dtype=torch.float32),
            }
    
    train_ds = RerankDataset(train_rows)
    val_ds = RerankDataset(val_rows)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fn = torch.nn.MSELoss()
    
    start = time.time()
    best_val_loss = float("inf")
    
    for epoch in range(args.epochs):
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
        
        print(f"Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  Saved best model (val_loss={val_loss:.4f})")
    
    elapsed = time.time() - start
    
    return {
        "base_model": args.base_model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "best_val_loss": float(best_val_loss),
        "training_time_sec": float(elapsed),
        "device": str(device),
        "seed": args.seed,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BERT reranker")
    parser.add_argument("--base-model", default="distilbert-base-uncased", help="HF model name")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
    except ImportError:
        pass
    
    print("Training fine-tuned reranker...")
    metrics = train(args)
    
    META_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\nMetadata saved to {META_PATH}")
    print(f"Model saved to {MODEL_PATH}")
    print(f"Best val loss: {metrics['best_val_loss']:.4f}")
    print(f"Training time: {metrics['training_time_sec']:.1f}s")


if __name__ == "__main__":
    main()
