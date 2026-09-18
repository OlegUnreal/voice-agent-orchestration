# Changelog

## 0.4.0 — 2026-09-18

- Hybrid retrieval: hand-rolled Okapi BM25 (`bm25.py`) — saturation, IDF, explicit length normalisation — with the old token-overlap scorer kept as a documented baseline; dense cosine + BM25 fused by reciprocal rank fusion (Cormack et al., SIGIR 2009) instead of incomparable raw-score blending.
- RankNet reranker (`reranker.py`): within-query feature differences mirrored for two classes, bias refitted back into slot 0; C selected by query-grouped 4-fold CV inside the train split, holdout read only by the acceptance gate.
- Acceptance-gated artifacts: `python -m helix.reranker --fit` writes `rerank_model.json`; the runtime ships a fitted rerank **only if it beats the hand-set prior on holdout nDCG@5 and MRR**. A rejected artifact degrades to the prior with the reason and metrics on record (current fit: 0.936 vs 0.987 — rejected, prior ships).
- RAG features extended for the rerank probe: lexical rank, RRF score, query-term coverage alongside cosine/ticker/title/recency.
- Evals: ablation across lexical/dense/hybrid/rerank stages on the golden set.
- Training path: TRL export verified end-to-end on the new qrels fixture (`python/tests/data/rerank_qrels.jsonl`, sha-pinned in the artifact).
- Docs: README Grounding section and the problems table updated for BM25/RRF/RankNet; suite 47 passing, 1 skipped (pgvector profile).
