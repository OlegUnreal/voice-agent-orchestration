"""TrainingOps: SFT-style pairs from golden qrels, LoRA r8 / QLoRA r4, model registry."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from helix.evals import GOLDEN, gate_check, run_eval_suite
from helix.models import Checkpoint, EvalMetrics, PromotionStage
from helix.rag import retrieve
from helix.reranker import BASELINE_WEIGHTS, EMBED_WEIGHTS, TrainPoint, feat, sklearn_check, train_logistic
from helix.settings import AS_OF

_lora_w = EMBED_WEIGHTS.copy()
_qlora_w = EMBED_WEIGHTS.copy()
_lora_loss: list[float] = []
_qlora_loss: list[float] = []
_last_train_ms = 0.0
_sklearn_w = EMBED_WEIGHTS.copy()


def get_adapter_weights(kind: str) -> np.ndarray | None:
    if kind == "lora":
        return _lora_w
    if kind == "embed":
        return EMBED_WEIGHTS
    if kind == "qlora":
        return _qlora_w
    return BASELINE_WEIGHTS


def _build_pairs() -> list[TrainPoint]:
    data: list[TrainPoint] = []
    for case in GOLDEN:
        ranked = retrieve(case.query, k=12, as_of=case.asOf, weights=EMBED_WEIGHTS)
        rel = set(case.relevant)
        q_up = case.query.upper()
        for r in ranked:
            title_hit = 1.0 if any(w in r.doc.title.lower() for w in case.query.lower().split()) else 0.0
            ticker_hit = 1.0 if r.doc.ticker and r.doc.ticker in q_up else 0.0
            data.append(
                TrainPoint(
                    features=feat(r.cosine, r.recency, ticker_hit, title_hit),
                    y=1 if r.doc.id in rel else 0,
                )
            )
    return data


def train_adapters(epochs_lora: int = 56, epochs_qlora: int = 40) -> dict:
    global _lora_w, _qlora_w, _lora_loss, _qlora_loss, _last_train_ms, _sklearn_w
    data = _build_pairs()
    t0 = datetime.now(timezone.utc)
    import time

    t_perf = time.perf_counter()
    _lora_w, _lora_loss = train_logistic(data, epochs_lora, 0.4, 0.008, 0)
    _qlora_w, _qlora_loss = train_logistic(data, epochs_qlora, 0.45, 0.02, 1)
    _sklearn_w = sklearn_check(data)
    _last_train_ms = (time.perf_counter() - t_perf) * 1000
    return {
        "loraLoss": _lora_loss,
        "qloraLoss": _qlora_loss,
        "lastTrainMs": _last_train_ms,
        "pairs": len(data),
        "sklearnIntercept": float(_sklearn_w[0]),
        "trainedAt": t0.isoformat(),
    }


def adapter_loss() -> dict:
    return {"loraLoss": _lora_loss, "qloraLoss": _qlora_loss, "lastTrainMs": _last_train_ms}


def default_checkpoints() -> list[Checkpoint]:
    lexical = run_eval_suite("lexical", BASELINE_WEIGHTS)
    embed = run_eval_suite("embed", EMBED_WEIGHTS)
    lora = run_eval_suite("lora", _lora_w)
    qlora = run_eval_suite("lora", _qlora_w)
    return [
        Checkpoint(
            id="ckpt-lex-01",
            name="lexical-bm25",
            kind="baseline",
            createdAt=int(datetime(2026, 7, 12, tzinfo=timezone.utc).timestamp() * 1000),
            metrics=lexical,
            stage="staging",
            lineage=[],
        ),
        Checkpoint(
            id="ckpt-emb-02",
            name="embed-hash64",
            kind="embed",
            createdAt=int(datetime(2026, 8, 2, tzinfo=timezone.utc).timestamp() * 1000),
            metrics=embed,
            stage="shadow",
            lineage=["ckpt-lex-01"],
        ),
        Checkpoint(
            id="ckpt-qlora-r4",
            name="rerank-qlora-r4",
            kind="qlora",
            rank=4,
            createdAt=int(datetime(2026, 8, 22, tzinfo=timezone.utc).timestamp() * 1000),
            metrics=qlora,
            stage="canary",
            lineage=["ckpt-lex-01", "ckpt-emb-02"],
        ),
        Checkpoint(
            id="ckpt-lora-r8",
            name="rerank-lora-r8",
            kind="lora",
            rank=8,
            createdAt=int(datetime(2026, 9, 4, tzinfo=timezone.utc).timestamp() * 1000),
            metrics=lora,
            stage="production",
            lineage=["ckpt-lex-01", "ckpt-emb-02", "ckpt-qlora-r4"],
        ),
    ]


def next_stage(stage: PromotionStage) -> PromotionStage | None:
    order: list[PromotionStage] = ["staging", "shadow", "canary", "production"]
    try:
        i = order.index(stage)
    except ValueError:
        return None
    return order[i + 1] if i + 1 < len(order) else None


def can_promote(m: EvalMetrics) -> bool:
    return all(g["ok"] for g in gate_check(m))


def promote(checkpoints: list[Checkpoint], checkpoint_id: str) -> list[Checkpoint]:
    out = [c.model_copy(deep=True) for c in checkpoints]
    for c in out:
        if c.id != checkpoint_id:
            continue
        nxt = next_stage(c.stage)
        if nxt and can_promote(c.metrics):
            c.stage = nxt
    return out
