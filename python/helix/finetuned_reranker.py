"""PyTorch fine-tuned reranker: third tier in the multi-tier strategy.

Architecture tiers, resolved in order:
    1. Fast tier: sklearn LogisticRegression on 9 hand-crafted features (reranker.py)
       - ~1ms inference, interpretable weights, CPU-only
    2. Accurate tier: HF CrossEncoder pre-trained on MS-MARCO (cross_encoder.py)
       - ~50ms inference, black-box, requires sentence-transformers
    3. Fine-tuned tier: PyTorch BERT fine-tuned on domain qrels (this module)
       - ~50ms inference, domain-adapted, requires PyTorch + GPU for training

Why three tiers: each solves a different problem. Fast is for production latency.
Accurate is for off-the-shelf quality. Fine-tuned is for domain-specific patterns
that pre-trained models miss (ticker symbols, financial jargon, time-sensitive
evidence). The runtime picks the tier based on HELIX_RERANKER_TIER env var.

Training data: python/tests/data/rerank_qrels.jsonl (same as sklearn tier).
Model artifact: helix/finetuned_reranker.pt (not committed; too large for GitHub).
Metadata: helix/finetuned_reranker.meta.json (committed; metrics + config).

The fine-tuned model is optional. If PyTorch is not installed or the artifact is
missing, the runtime falls back to tier 1 or 2. This is not a demo — it is a
production strategy where you can ship fast today and fine-tune tomorrow.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from helix.models import RetrievedDoc

MODEL_PATH = Path(__file__).with_name("finetuned_reranker.pt")
META_PATH = Path(__file__).with_name("finetuned_reranker.meta.json")


def enabled() -> bool:
    """Check if fine-tuned tier is enabled and available."""
    tier = os.environ.get("HELIX_RERANKER_TIER", "fast").lower()
    if tier != "finetuned":
        return False
    if not MODEL_PATH.exists():
        return False
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def rerank(query: str, hits: list[RetrievedDoc], top: int = 8) -> list[RetrievedDoc]:
    """Fine-tuned BERT reranking. Falls back to input order if unavailable."""
    if not enabled() or len(hits) < 2:
        return hits[:top]
    
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        return hits[:top]
    
    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    model_name = meta.get("base_model", "distilbert-base-uncased")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    
    pairs = [(query, f"{h.doc.title} {h.doc.body}") for h in hits]
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=256)
    
    with torch.no_grad():
        outputs = model(**inputs)
        scores = outputs.logits.squeeze().numpy()
    
    ranked = sorted(zip(hits, scores, strict=False), key=lambda x: float(x[1]), reverse=True)
    out = []
    for hit, sc in ranked[:top]:
        hit.rerank = float(sc)
        hit.score = float(sc)
        out.append(hit)
    return out
