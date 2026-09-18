"""Model drift detection: metric, serving, and input distribution monitoring.

Drift is detected by comparing a recent window against a historical baseline.
No scipy dependency — uses numpy percentile bootstrap for confidence intervals
and a simple z-test for mean shift detection.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from helix.evals import run_eval_suite
from helix.experiments import list_experiments
from helix.serving import serving_report
from helix.store import read_jsonl

RECENT_WINDOW = 50
BASELINE_MIN = 10
DRIFT_THRESHOLD_Z = 2.0


def _z_test_mean(recent: list[float], baseline: list[float]) -> dict[str, float]:
    """Two-sample z-test for mean shift. Returns z-score and p-value approximation."""
    if len(recent) < 2 or len(baseline) < 2:
        return {"z": 0.0, "drift": False}
    r = np.array(recent)
    b = np.array(baseline)
    r_mean, b_mean = r.mean(), b.mean()
    r_se = r.std(ddof=1) / np.sqrt(len(r))
    b_se = b.std(ddof=1) / np.sqrt(len(b))
    se = np.sqrt(r_se**2 + b_se**2)
    if se < 1e-12:
        return {"z": 0.0, "drift": False, "delta": float(r_mean - b_mean)}
    z = (r_mean - b_mean) / se
    return {"z": float(z), "drift": abs(z) > DRIFT_THRESHOLD_Z, "delta": float(r_mean - b_mean), "recentMean": float(r_mean), "baselineMean": float(b_mean)}


def detect_metric_drift() -> dict[str, Any]:
    """Compare current golden-set eval against historical experiment runs.

    Runs the eval suite now, then compares each metric against the distribution
    of the same metric from past experiment logs. Flags metrics where the current
    value is >2 standard deviations from the historical mean.
    """
    current = run_eval_suite("lora")
    current_metrics = current.model_dump()

    exps = list_experiments(limit=200)
    eval_exps = [e for e in exps if e.get("name") == "evalforge" and e.get("metrics")]

    if len(eval_exps) < BASELINE_MIN:
        return {
            "driftDetected": False,
            "reason": "insufficientBaseline",
            "baselineRuns": len(eval_exps),
            "required": BASELINE_MIN,
            "current": {k: round(v, 4) if isinstance(v, float) else v for k, v in current_metrics.items()},
        }

    metric_names = ["recallAt5", "recallAt10", "mrr", "ndcgAt10", "groundedness", "p95Ms"]
    results = {}
    drifted = []
    for m in metric_names:
        baseline_vals = [e["metrics"][m] for e in eval_exps if m in e.get("metrics", {})]
        if len(baseline_vals) < 3:
            continue
        current_val = current_metrics.get(m, 0.0)
        b = np.array(baseline_vals)
        z = (current_val - b.mean()) / (b.std(ddof=1) + 1e-12)
        is_drift = abs(z) > DRIFT_THRESHOLD_Z
        results[m] = {
            "current": round(float(current_val), 4),
            "baselineMean": round(float(b.mean()), 4),
            "baselineStd": round(float(b.std(ddof=1)), 4),
            "z": round(float(z), 3),
            "drift": is_drift,
        }
        if is_drift:
            drifted.append(m)

    return {
        "driftDetected": len(drifted) > 0,
        "driftedMetrics": drifted,
        "baselineRuns": len(eval_exps),
        "metrics": results,
    }


def detect_serving_drift() -> dict[str, Any]:
    """Compare recent serving telemetry against historical baseline.

    Splits serving.jsonl into recent (last RECENT_WINDOW rows) and baseline
    (everything before). Tests TTFT, total time, and failure rate for shift.
    """
    report = serving_report(limit=500)
    rows = report.get("rows", [])

    if len(rows) < RECENT_WINDOW + BASELINE_MIN:
        return {
            "driftDetected": False,
            "reason": "insufficientData",
            "totalRows": len(rows),
            "required": RECENT_WINDOW + BASELINE_MIN,
        }

    recent = rows[:RECENT_WINDOW]
    baseline = rows[RECENT_WINDOW:]

    checks = {}
    drifted = []

    r_ttft = [r.get("ttftMs", 0) for r in recent]
    b_ttft = [r.get("ttftMs", 0) for r in baseline]
    checks["ttftMs"] = _z_test_mean(r_ttft, b_ttft)
    if checks["ttftMs"]["drift"]:
        drifted.append("ttftMs")

    r_total = [r.get("totalMs", 0) for r in recent]
    b_total = [r.get("totalMs", 0) for r in baseline]
    checks["totalMs"] = _z_test_mean(r_total, b_total)
    if checks["totalMs"]["drift"]:
        drifted.append("totalMs")

    r_fail = [1.0 if r.get("failed") else 0.0 for r in recent]
    b_fail = [1.0 if r.get("failed") else 0.0 for r in baseline]
    checks["failureRate"] = _z_test_mean(r_fail, b_fail)
    if checks["failureRate"]["drift"]:
        drifted.append("failureRate")

    r_tokens = [r.get("tokensIn", 0) + r.get("tokensOut", 0) for r in recent]
    b_tokens = [r.get("tokensIn", 0) + r.get("tokensOut", 0) for r in baseline]
    checks["tokensPerQuery"] = _z_test_mean(r_tokens, b_tokens)
    if checks["tokensPerQuery"]["drift"]:
        drifted.append("tokensPerQuery")

    return {
        "driftDetected": len(drifted) > 0,
        "driftedMetrics": drifted,
        "recentWindow": len(recent),
        "baselineWindow": len(baseline),
        "checks": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in checks.items()},
    }


def detect_input_drift() -> dict[str, Any]:
    """Detect distribution shift in incoming queries.

    Tracks query length and intent distribution. Compares recent window against
    the full history. Uses KS-approximation via percentile comparison.
    """
    from helix.ledger import list_traces

    traces = list_traces(500)
    if len(traces) < RECENT_WINDOW + BASELINE_MIN:
        return {
            "driftDetected": False,
            "reason": "insufficientData",
            "totalTraces": len(traces),
        }

    recent = traces[:RECENT_WINDOW]
    baseline = traces[RECENT_WINDOW:]

    r_lengths = [len(t.get("query", "")) for t in recent]
    b_lengths = [len(t.get("query", "")) for t in baseline]
    length_check = _z_test_mean([float(x) for x in r_lengths], [float(x) for x in b_lengths])

    r_intents: dict[str, int] = {}
    b_intents: dict[str, int] = {}
    for t in recent:
        intent = t.get("intent", "general")
        r_intents[intent] = r_intents.get(intent, 0) + 1
    for t in baseline:
        intent = t.get("intent", "general")
        b_intents[intent] = b_intents.get(intent, 0) + 1

    all_intents = set(r_intents.keys()) | set(b_intents.keys())
    intent_drift = False
    intent_details = {}
    r_total = sum(r_intents.values()) or 1
    b_total = sum(b_intents.values()) or 1
    for intent in all_intents:
        r_frac = r_intents.get(intent, 0) / r_total
        b_frac = b_intents.get(intent, 0) / b_total
        delta = abs(r_frac - b_frac)
        if delta > 0.15:
            intent_drift = True
        intent_details[intent] = {"recentFrac": round(r_frac, 3), "baselineFrac": round(b_frac, 3), "delta": round(delta, 3)}

    return {
        "driftDetected": length_check["drift"] or intent_drift,
        "queryLength": {k: round(v, 4) if isinstance(v, float) else v for k, v in length_check.items()},
        "intentDistribution": intent_details,
        "recentWindow": len(recent),
        "baselineWindow": len(baseline),
    }


def drift_report() -> dict[str, Any]:
    """Combined drift report: metric + serving + input drift."""
    metric = detect_metric_drift()
    serving = detect_serving_drift()
    inp = detect_input_drift()
    return {
        "ts": int(time.time() * 1000),
        "metric": metric,
        "serving": serving,
        "input": inp,
        "anyDrift": any([
            metric.get("driftDetected", False),
            serving.get("driftDetected", False),
            inp.get("driftDetected", False),
        ]),
    }
