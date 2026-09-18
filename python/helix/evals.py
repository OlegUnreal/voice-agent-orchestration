"""EvalForge: golden / regression set, Recall@K, MRR, nDCG, groundedness, promotion gates."""

from __future__ import annotations

import hashlib
import time

import numpy as np

from helix.models import EvalCase, EvalMetrics, Intent
from helix.rag import blend_retrieve, dense_retrieve, lexical_retrieve, overlap_retrieve, retrieve
from helix.reranker import BASELINE_WEIGHTS
from helix.router import classify_intent
from helix.settings import AS_OF, DAY_MS as D, GATES

GOLDEN: list[EvalCase] = [
    EvalCase(id="g1", query="What is BTC's current regime?", asOf=AS_OF, relevant=["R-BTC-REGIME"], expectedIntent="regime", expectedTools=["detect_regime", "get_market_snapshot"]),
    EvalCase(id="g2", query="Any BTC exchange inflow anomalies?", asOf=AS_OF, relevant=["R-BTC-ONCHAIN", "R-ANOM-WINDOW"], expectedIntent="anomaly", expectedTools=["detect_anomalies"]),
    EvalCase(id="g3", query="ETH staking flows and liquid staking", asOf=AS_OF, relevant=["R-ETH-STAKING"], expectedIntent="research", expectedTools=["retrieve_evidence"]),
    EvalCase(id="g4", query="Backtest SMA crossover on SOL", asOf=AS_OF, relevant=["R-SOL-MOM", "R-STRAT-SMA"], expectedIntent="backtest", expectedTools=["run_backtest"]),
    EvalCase(id="g5", query="Portfolio risk if BTC drops 12 percent", asOf=AS_OF, relevant=["R-PORT-VAR", "R-PORT-REVIEW"], expectedIntent="risk", expectedTools=["estimate_risk"]),
    EvalCase(id="g6", query="Cite evidence for the SOL momentum call", asOf=AS_OF, relevant=["R-SOL-MOM", "R-SOL-ONCHAIN"], expectedIntent="research", expectedTools=["retrieve_evidence"]),
    EvalCase(id="g7", query="How do promotion gates work for rerankers?", asOf=AS_OF, relevant=["R-PROMOTE", "R-GOLDEN"], expectedIntent="eval", expectedTools=["get_eval_report"]),
    EvalCase(id="g8", query="NVDA implied vol after earnings", asOf=AS_OF, relevant=["R-NVDA-EARN"], expectedIntent="research", expectedTools=["retrieve_evidence"]),
    EvalCase(id="g9", query="Is AAPL useful ballast in a crypto book?", asOf=AS_OF, relevant=["R-AAPL-DEF", "R-PORT-VAR"], expectedIntent="portfolio", expectedTools=["estimate_risk", "retrieve_evidence"]),
    EvalCase(id="g10", query="What happened in the late-sample stress window?", asOf=AS_OF, relevant=["R-ANOM-WINDOW", "R-BTC-STRAT", "R-BTC-ONCHAIN"], expectedIntent="anomaly", expectedTools=["detect_anomalies", "retrieve_evidence"]),
    EvalCase(id="g11", query="ETH regime — trend or high vol?", asOf=AS_OF, relevant=["R-ETH-REGIME"], expectedIntent="regime", expectedTools=["detect_regime"]),
    EvalCase(id="g12", query="Explain the groundedness contract for spoken answers", asOf=AS_OF, relevant=["R-CITATION"], expectedIntent="research", expectedTools=["retrieve_evidence"]),
    EvalCase(id="g13", query="Which MCP tools need approval?", asOf=AS_OF, relevant=["R-MCP"], expectedIntent="general", expectedTools=["retrieve_evidence"]),
    EvalCase(id="g14", query="SMA playbook and trading costs", asOf=AS_OF, relevant=["R-STRAT-SMA", "R-BTC-STRAT"], expectedIntent="backtest", expectedTools=["run_backtest", "retrieve_evidence"]),
    EvalCase(id="g15", query="Review the default 40 25 15 10 10 book", asOf=AS_OF, relevant=["R-PORT-VAR", "R-PORT-REVIEW"], expectedIntent="portfolio", expectedTools=["estimate_risk"]),
    EvalCase(id="g16", query="Retrieval policy recency and no leakage", asOf=AS_OF - D, relevant=["R-RAG-POLICY"], expectedIntent="research", expectedTools=["retrieve_evidence"]),
]

INTENT_TOOLS: dict[Intent, list[str]] = {
    "regime": ["detect_regime", "get_market_snapshot"],
    "anomaly": ["detect_anomalies"],
    "backtest": ["run_backtest"],
    "risk": ["estimate_risk"],
    "research": ["retrieve_evidence"],
    "portfolio": ["estimate_risk", "retrieve_evidence"],
    "eval": ["get_eval_report"],
    "training": ["get_checkpoint_status"],
    "memory": ["memory_recall"],
    "general": ["retrieve_evidence"],
}


def dataset_version() -> str:
    payload = "|".join(f"{c.id}:{c.query}:{','.join(c.relevant)}" for c in GOLDEN)
    return "ds-golden-" + hashlib.sha1(payload.encode()).hexdigest()[:10]


def _dcg(rels: list[float]) -> float:
    return sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(rels))


def retrieval_metrics(ranked_ids: list[str], relevant: list[str]) -> dict[str, float]:
    rel = set(relevant)
    hits5 = sum(1 for i in ranked_ids[:5] if i in rel)
    hits10 = sum(1 for i in ranked_ids[:10] if i in rel)
    mrr = 0.0
    for i, rid in enumerate(ranked_ids):
        if rid in rel:
            mrr = 1.0 / (i + 1)
            break
    gains = [1.0 if i in rel else 0.0 for i in ranked_ids[:10]]
    idcg = _dcg([1.0] * min(len(relevant), 10))
    ndcg = (_dcg(gains) / idcg) if idcg else 0.0
    nrel = len(relevant) or 1
    return {
        "recall5": hits5 / nrel,
        "recall10": hits10 / nrel,
        "mrr": mrr,
        "ndcg10": ndcg,
    }


RETRIEVERS: dict[str, str] = {
    "overlap": "v1 lexical: raw token overlap, no idf, no saturation",
    "lexical": "BM25 (k1=1.4, b=0.4, title x2)",
    "dense": "hashing-64 cosine only, no lexical signal",
    "hybrid": "BM25 + dense cosine fused with RRF, first stage only",
    "blend": "v1 scorer: 0.4*rerank + 0.4*overlap + 0.12*ticker + 0.08*recency",
    "lora": "hybrid + learned logistic rerank (production path)",
    # The three rows below are *not* different pipelines: they are the product
    # pipeline scored with a caller-supplied coefficient vector, which is what
    # the model registry needs in order to compare adapters fairly.
    "embed": "hybrid + v1 EMBED_WEIGHTS rerank vector (5-d, projected by name)",
    "qlora": "hybrid + rank-limited rerank adapter",
    "baseline": "hybrid + v1 BASELINE_WEIGHTS rerank vector",
}
# Kinds that mean "product hybrid, reranker on, weights from the caller".
_RERANKED_KINDS = ("lora", "embed", "qlora", "baseline")
ABLATION_ORDER = ("overlap", "lexical", "dense", "hybrid", "blend", "lora")


def _run_retriever(kind: str, query: str, as_of: int, weights: np.ndarray | None) -> list:
    if kind == "overlap":
        return overlap_retrieve(query, 10, as_of)
    if kind == "lexical":
        return lexical_retrieve(query, 10, as_of)
    if kind == "dense":
        return dense_retrieve(query, 10, as_of)
    if kind == "hybrid":
        return retrieve(query, k=10, as_of=as_of, fusion="rrf", rerank=False)
    if kind == "blend":
        return blend_retrieve(query, k=10, as_of=as_of, weights=weights)
    if kind in _RERANKED_KINDS:
        return retrieve(query, k=10, as_of=as_of, weights=weights)
    raise ValueError(f"unknown eval kind {kind!r}; expected one of {sorted(RETRIEVERS)}")


def run_eval_suite(kind: str = "lora", weights: np.ndarray | None = None) -> EvalMetrics:
    times: list[float] = []
    rows: list[dict[str, float]] = []
    for case in GOLDEN:
        t0 = time.perf_counter()
        ranked = _run_retriever(kind, case.query, case.asOf, weights)
        ids = [r.doc.id for r in ranked]
        ir = retrieval_metrics(ids, case.relevant)
        intent = classify_intent(case.query)
        tools = INTENT_TOOLS[intent]
        tool_hit = sum(1 for t in case.expectedTools if t in tools) / max(len(case.expectedTools), 1)
        cite_ok = 1.0 if any(i in case.relevant for i in ids[:5]) else 0.0
        needs_cite = "retrieve_evidence" in case.expectedTools
        grounded = cite_ok if needs_cite else 1.0
        rows.append(
            {
                **ir,
                "intentOk": 1.0 if intent == case.expectedIntent else 0.0,
                "toolHit": tool_hit,
                "citeOk": cite_ok,
                "grounded": grounded,
                "structured": 1.0 if intent == case.expectedIntent else 0.0,
            }
        )
        times.append((time.perf_counter() - t0) * 1000)
    n = len(rows) or 1
    pick = lambda k: sum(r[k] for r in rows) / n
    times.sort()
    p50 = times[int(n * 0.5)] if times else 0.0
    p95 = times[min(int(n * 0.95), n - 1)] if times else 0.0
    return EvalMetrics(
        recallAt5=pick("recall5"),
        recallAt10=pick("recall10"),
        mrr=pick("mrr"),
        ndcgAt10=pick("ndcg10"),
        intentAcc=pick("intentOk"),
        toolAcc=pick("toolHit"),
        groundedness=pick("grounded"),
        citationCorrect=pick("citeOk"),
        structuredOk=pick("structured"),
        p50Ms=p50,
        p95Ms=p95,
        # Placeholder cost model, not a measurement: every stage here runs in
        # process. The number exists so the cost gate has something to bind to
        # and so swapping in a paid embedder/reranker trips it deliberately.
        costUsd=0.0 if kind in {"overlap", "lexical", "dense", "hybrid"} else 0.0012,
    )


def gate_check(m: EvalMetrics) -> list[dict]:
    return [
        {"name": "nDCG@10 ≥ 0.72", "ok": m.ndcgAt10 >= GATES["ndcgAt10"], "value": m.ndcgAt10},
        {"name": "Recall@5 ≥ 0.80", "ok": m.recallAt5 >= GATES["recallAt5"], "value": m.recallAt5},
        {"name": "Groundedness ≥ 0.85", "ok": m.groundedness >= GATES["groundedness"], "value": m.groundedness},
        {"name": "p95 < 800ms", "ok": m.p95Ms < GATES["p95Ms"], "value": m.p95Ms},
        {"name": "Cost/query < $0.02", "ok": m.costUsd < GATES["costUsd"], "value": m.costUsd},
    ]


def ablation(kinds: tuple[str, ...] = ABLATION_ORDER) -> list[dict]:
    """Retriever ablation over the golden set: one row per stage of the ladder.

    Deltas are against the BM25 row (`lexical`), which is the strongest single
    retriever here and therefore the honest control: beating the *hybrid* row
    would flatter the reranker, beating lexical shows what fusion adds.
    """
    scores = {kind: run_eval_suite(kind) for kind in kinds}
    control = scores.get("lexical")
    rows: list[dict] = []
    for kind in kinds:
        m = scores[kind]
        row = {
            "kind": kind,
            "what": RETRIEVERS[kind],
            "recallAt5": round(m.recallAt5, 4),
            "recallAt10": round(m.recallAt10, 4),
            "mrr": round(m.mrr, 4),
            "ndcgAt10": round(m.ndcgAt10, 4),
            "p95Ms": round(m.p95Ms, 2),
            "gatesOk": all(g["ok"] for g in gate_check(m)),
        }
        if control is not None and kind != "lexical":
            row["deltaMrr"] = round(m.mrr - control.mrr, 4)
            row["deltaNdcgAt10"] = round(m.ndcgAt10 - control.ndcgAt10, 4)
            row["deltaRecallAt5"] = round(m.recallAt5 - control.recallAt5, 4)
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse
    import json

    parser = argparse.ArgumentParser(prog="helix.evals")
    parser.add_argument("--ablation", action="store_true", help="run every retriever row over GOLDEN")
    parser.add_argument("--kind", default="lora")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.ablation:
        rows = ablation()
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            head = f"{'kind':<9}{'R@5':>8}{'R@10':>8}{'MRR':>8}{'nDCG@10':>9}{'p95ms':>8}  delta vs BM25 control"
            print(head)
            print("-" * len(head))
            for r in rows:
                delta = "  " if "deltaMrr" not in r else f"  MRR{r['deltaMrr']:+.4f}  nDCG{r['deltaNdcgAt10']:+.4f}  R@5{r['deltaRecallAt5']:+.4f}"
                print(f"{r['kind']:<9}{r['recallAt5']:>8.4f}{r['recallAt10']:>8.4f}{r['mrr']:>8.4f}{r['ndcgAt10']:>9.4f}{r['p95Ms']:>8.2f}{delta}")
        return 0
    m = run_eval_suite(args.kind)
    print(json.dumps({"kind": args.kind, "metrics": m.model_dump(), "gates": gate_check(m)}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
