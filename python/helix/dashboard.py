"""Performance dashboard: single endpoint aggregating serving, evals, drift, experiments.

Production ops need one view, not five tabs. This module pulls from the existing
modules (serving, evals, drift, experiments, traces) and returns a unified report
suitable for a Grafana panel, a Slack webhook, or a CLI summary.
"""

from __future__ import annotations

from typing import Any

from helix.drift import drift_report
from helix.evals import gate_check, run_eval_suite
from helix.experiments import list_experiments
from helix.serving import serving_report


def dashboard_report(eval_kind: str = "lora") -> dict[str, Any]:
    """Aggregate serving, eval, drift, and experiment data into one report.

    This is the "single pane of glass" for ops. Each section is a dict so the
    frontend can render independently; the top-level `health` field is a quick
    green/yellow/red for the whole system.
    """
    serving = serving_report()
    eval_metrics = run_eval_suite(eval_kind)
    eval_gates = gate_check(eval_metrics)
    drift = drift_report()
    experiments = list_experiments(limit=10)

    gates_ok = all(g["ok"] for g in eval_gates)
    any_drift = drift.get("anyDrift", False)
    failure_rate = serving.get("failureRate", 0)

    if not gates_ok or failure_rate > 0.1:
        health = "red"
    elif any_drift or failure_rate > 0.02:
        health = "yellow"
    else:
        health = "green"

    return {
        "health": health,
        "serving": {
            "n": serving.get("n", 0),
            "failureRate": round(failure_rate, 4),
            "ttftP50Ms": round(serving.get("ttftP50Ms", 0), 2),
            "ttftP95Ms": round(serving.get("ttftP95Ms", 0), 2),
            "totalP95Ms": round(serving.get("totalP95Ms", 0), 2),
            "tokens": serving.get("tokens", 0),
            "costUsd": round(serving.get("costUsd", 0), 6),
        },
        "eval": {
            "kind": eval_kind,
            "recallAt5": round(eval_metrics.recallAt5, 4),
            "recallAt10": round(eval_metrics.recallAt10, 4),
            "mrr": round(eval_metrics.mrr, 4),
            "ndcgAt10": round(eval_metrics.ndcgAt10, 4),
            "groundedness": round(eval_metrics.groundedness, 4),
            "p50Ms": round(eval_metrics.p50Ms, 2),
            "p95Ms": round(eval_metrics.p95Ms, 2),
            "gatesOk": gates_ok,
            "gates": eval_gates,
        },
        "drift": {
            "anyDrift": any_drift,
            "metric": drift.get("metric", {}),
            "serving": drift.get("serving", {}),
            "input": drift.get("input", {}),
        },
        "experiments": experiments,
    }
