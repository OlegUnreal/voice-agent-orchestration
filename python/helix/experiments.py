"""MLflow-style experiment log without a tracking server. Params + metrics + dataset hash."""

from __future__ import annotations

import os
import time
from typing import Any

from helix.store import append_jsonl, read_jsonl

EXPS = "experiments.jsonl"


def log_experiment(name: str, params: dict[str, Any], metrics: dict[str, Any], dataset: str | None = None) -> dict[str, Any]:
    rec = {
        "id": f"exp-{int(time.time() * 1000)}",
        "ts": int(time.time() * 1000),
        "name": name,
        "params": params,
        "metrics": metrics,
        "dataset": dataset,
        "git": os.environ.get("HELIX_GIT_SHA") or os.environ.get("GITHUB_SHA"),
        "embeddings": os.environ.get("HELIX_EMBEDDINGS", "hash"),
    }
    append_jsonl(EXPS, rec)
    return rec


def list_experiments(limit: int = 40) -> list[dict[str, Any]]:
    return list(reversed(read_jsonl(EXPS)[-limit:]))
