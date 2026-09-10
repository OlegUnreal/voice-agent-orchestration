"""Serving telemetry: TTFT, tokens, cost, failures. Chat products hide this; EvalForge does not."""

from __future__ import annotations

import os
import time
from typing import Any

from helix.store import append_jsonl, read_jsonl

SERVING = "serving.jsonl"


def estimate_tokens(text: str) -> int:
    return max(1, (len(text or "") + 3) // 4)


def estimate_cost(tokens_in: int, tokens_out: int, model: str) -> float:
    if model.startswith("local") or model.startswith("vllm") or model.endswith("-cache"):
        in_rate = float(os.environ.get("HELIX_VLLM_USD_PER_IN", "0"))
        out_rate = float(os.environ.get("HELIX_VLLM_USD_PER_OUT", "0"))
    else:
        in_rate = float(os.environ.get("HELIX_XAI_USD_PER_IN", "0.000003"))
        out_rate = float(os.environ.get("HELIX_XAI_USD_PER_OUT", "0.000015"))
    return tokens_in * in_rate + tokens_out * out_rate


def log_serving(row: dict[str, Any]) -> dict[str, Any]:
    rec = {
        "ts": int(time.time() * 1000),
        "traceId": row.get("traceId"),
        "model": row.get("model") or "local-tools",
        "ttftMs": float(row.get("ttftMs") or 0),
        "totalMs": float(row.get("totalMs") or 0),
        "tokensIn": int(row.get("tokensIn") or 0),
        "tokensOut": int(row.get("tokensOut") or 0),
        "costUsd": float(row.get("costUsd") or 0),
        "failed": bool(row.get("failed")),
        "streaming": bool(row.get("streaming")),
    }
    append_jsonl(SERVING, rec)
    return rec


def serving_report(limit: int = 200) -> dict[str, Any]:
    rows = read_jsonl(SERVING)[-limit:]
    n = len(rows) or 1
    fails = sum(1 for r in rows if r.get("failed"))
    ttfts = sorted(float(r.get("ttftMs") or 0) for r in rows)
    totals = sorted(float(r.get("totalMs") or 0) for r in rows)

    def pct(xs: list[float], p: float) -> float:
        if not xs:
            return 0.0
        return xs[min(int(p * (len(xs) - 1)), len(xs) - 1)]

    return {
        "n": len(rows),
        "failureRate": fails / n,
        "ttftP50Ms": pct(ttfts, 0.5),
        "ttftP95Ms": pct(ttfts, 0.95),
        "totalP95Ms": pct(totals, 0.95),
        "tokens": sum(int(r.get("tokensIn") or 0) + int(r.get("tokensOut") or 0) for r in rows),
        "costUsd": sum(float(r.get("costUsd") or 0) for r in rows),
        "rows": list(reversed(rows[-40:])),
    }
