"""A/B testing framework for reranker variants.

Splits golden-set eval between two reranker configurations, collects per-variant
metrics, and tests for statistically significant difference. No external A/B
server — runs as a one-shot evaluation over the golden set with deterministic
assignment so results are reproducible.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np

from helix.evals import GOLDEN, retrieval_metrics, _run_retriever
from helix.experiments import log_experiment
from helix.reranker import BASELINE_WEIGHTS


def _assign_variant(case_id: str, seed: int = 0) -> str:
    """Deterministic assignment: hash(case_id + seed) → A or B."""
    h = hashlib.md5(f"{case_id}:{seed}".encode()).hexdigest()
    return "A" if int(h[:8], 16) % 2 == 0 else "B"


def run_ab_test(
    kind_a: str = "lora",
    kind_b: str = "hybrid",
    weights_a: np.ndarray | None = None,
    weights_b: np.ndarray | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Run A/B evaluation over the golden set.

    Each golden case is assigned to variant A or B by deterministic hash.
    Both variants are evaluated on their assigned cases. Returns per-variant
    metrics, per-case assignment, and a significance test.
    """
    assignments: list[dict[str, Any]] = []
    results_a: list[dict[str, float]] = []
    results_b: list[dict[str, float]] = []

    for case in GOLDEN:
        variant = _assign_variant(case.id, seed)
        kind = kind_a if variant == "A" else kind_b
        w = weights_a if variant == "A" else weights_b

        t0 = time.perf_counter()
        ranked = _run_retriever(kind, case.query, case.asOf, w)
        elapsed = (time.perf_counter() - t0) * 1000
        ids = [r.doc.id for r in ranked]
        metrics = retrieval_metrics(ids, case.relevant)

        row = {
            "caseId": case.id,
            "query": case.query,
            "variant": variant,
            "kind": kind,
            "latencyMs": round(elapsed, 2),
            **{k: round(v, 4) for k, v in metrics.items()},
        }
        assignments.append(row)

        if variant == "A":
            results_a.append(metrics)
        else:
            results_b.append(metrics)

    def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
        if not rows:
            return {"n": 0, "recallAt5": 0, "mrr": 0, "ndcgAt10": 0}
        n = len(rows)
        return {
            "n": n,
            "recallAt5": round(sum(r["recall5"] for r in rows) / n, 4),
            "recallAt10": round(sum(r["recall10"] for r in rows) / n, 4),
            "mrr": round(sum(r["mrr"] for r in rows) / n, 4),
            "ndcgAt10": round(sum(r["ndcg10"] for r in rows) / n, 4),
        }

    agg_a = _aggregate(results_a)
    agg_b = _aggregate(results_b)

    significance = _paired_test(results_a, results_b, assignments, seed)

    report = {
        "ts": int(time.time() * 1000),
        "variantA": {"kind": kind_a, **agg_a},
        "variantB": {"kind": kind_b, **agg_b},
        "significance": significance,
        "assignments": assignments,
    }

    log_experiment(
        "ab_test",
        {"kindA": kind_a, "kindB": kind_b, "seed": seed},
        {
            "nA": agg_a["n"],
            "nB": agg_b["n"],
            "mrrA": agg_a["mrr"],
            "mrrB": agg_b["mrr"],
            "ndcgA": agg_a["ndcgAt10"],
            "ndcgB": agg_b["ndcgAt10"],
            "significant": significance.get("significant", False),
        },
    )

    return report


def _paired_test(
    results_a: list[dict[str, float]],
    results_b: list[dict[str, float]],
    assignments: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    """Paired permutation test on MRR.

    For each case, compute the MRR under both variants (not just the assigned
    one). Then test whether the observed mean difference is significant by
    comparing against the permutation distribution.
    """
    if len(results_a) < 3 or len(results_b) < 3:
        return {"significant": False, "reason": "insufficientSamples", "pValue": 1.0}

    case_mrr_pairs: list[tuple[float, float]] = []
    for row in assignments:
        case_id = row["caseId"]
        case = next(c for c in GOLDEN if c.id == case_id)
        kind_a = row["kind"] if row["variant"] == "A" else "lora"
        kind_b = row["kind"] if row["variant"] == "B" else "hybrid"

        ranked_a = _run_retriever("lora", case.query, case.asOf, BASELINE_WEIGHTS)
        ids_a = [r.doc.id for r in ranked_a]
        mrr_a = retrieval_metrics(ids_a, case.relevant)["mrr"]

        ranked_b = _run_retriever("hybrid", case.query, case.asOf, None)
        ids_b = [r.doc.id for r in ranked_b]
        mrr_b = retrieval_metrics(ids_b, case.relevant)["mrr"]

        case_mrr_pairs.append((mrr_a, mrr_b))

    diffs = [a - b for a, b in case_mrr_pairs]
    observed_diff = np.mean(diffs)

    n_perm = 200
    perm_diffs = []
    rng = np.random.RandomState(seed)
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(diffs))
        perm_diffs.append(np.mean(np.array(diffs) * signs))

    p_value = float(np.mean(np.abs(perm_diffs) >= abs(observed_diff)))

    return {
        "significant": p_value < 0.05,
        "pValue": round(p_value, 4),
        "observedDiff": round(float(observed_diff), 4),
        "nPermutations": n_perm,
        "metric": "mrr",
    }
