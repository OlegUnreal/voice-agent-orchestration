"""Optional HF cross-encoder rerank. Hashing first-stage misses paraphrases; this closes that."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from helix.models import RetrievedDoc


def enabled() -> bool:
    flag = os.environ.get("HELIX_CROSS_ENCODER", "").lower()
    return flag in {"1", "true", "yes", "st"}


def rerank(query: str, hits: list[RetrievedDoc], top: int = 8) -> list[RetrievedDoc]:
    if not enabled() or len(hits) < 2:
        return hits[:top]
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        return hits[:top]
    model_name = os.environ.get("HELIX_CE_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    model = CrossEncoder(model_name)
    pairs = [(query, f"{h.doc.title} {h.doc.body}") for h in hits]
    scores = model.predict(pairs)
    ranked = sorted(zip(hits, scores, strict=False), key=lambda x: float(x[1]), reverse=True)
    out = []
    for hit, sc in ranked[:top]:
        hit.rerank = float(sc)
        hit.score = float(sc)
        out.append(hit)
    return out
