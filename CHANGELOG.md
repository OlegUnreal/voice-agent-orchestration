# Changelog

## 0.8.0 — 2026-09-18

- **A/B testing framework** (`ab_test.py`, `POST /v1/ab-test`): compare two reranker variants (e.g. `lora` vs `hybrid`) on the golden eval set with deterministic hash-based assignment. Paired permutation test on MRR (200 permutations) reports whether the difference is statistically significant. Results logged to `experiments.jsonl` for MLflow-style tracking. No scipy dependency.
- Suite 49 passing, 1 skipped (pgvector profile).

## 0.7.0 — 2026-09-18

- **Model drift detection** (`drift.py`, `GET /v1/drift`): three-category monitoring — metric drift (golden-set eval vs historical experiment runs), serving drift (TTFT/failure rate/tokens per query, recent window vs baseline), input drift (query length distribution + intent mix shift). Z-test for mean shift with 2σ threshold, no scipy dependency. Returns `anyDrift` flag plus per-category detail.
- Suite 48 passing, 1 skipped (pgvector profile).

## 0.6.0 — 2026-09-18

- **SSE streaming endpoint** (`POST /v1/turn/stream`): Server-Sent Events for progressive voice-UI rendering. Streams response in phases — `thinking` → `tool` → `spoken` (chunked for streaming TTS) → `citation` → `done` — each event is JSON with `type` and `elapsed_ms`. Voice UIs start TTS before the orchestrator finishes.
- **WebSocket endpoint** (`WS /v1/ws/turn`): bidirectional JSON for real-time voice-agent communication. Client sends `{"text": "..."}`, server streams the same phase protocol as SSE. Full-duplex: supports interrupt handling and barge-in for voice UIs that need it.
- Both endpoints share the orchestrator pipeline with `POST /v1/turn` — no separate code path, no separate cache. The existing request/response endpoint stays the default for simple integrations.
- Suite 47 passing, 1 skipped (pgvector profile).

## 0.5.0 — 2026-09-18

- **Multi-tier reranking strategy**: production systems need to ship today and improve tomorrow without rewriting the pipeline. Helix resolves reranking in three tiers, selected by `HELIX_RERANKER_TIER` env var:
  1. **Fast tier** (default): RankNet pairwise logistic probe on 9 hand-crafted features (`reranker.py`). ~1ms inference, interpretable weights, CPU-only.
  2. **Accurate tier** (`HELIX_RERANKER_TIER=accurate`): HF CrossEncoder pre-trained on MS-MARCO (`cross_encoder.py`). ~50ms inference, black-box, requires `sentence-transformers`.
  3. **Fine-tuned tier** (`HELIX_RERANKER_TIER=finetuned`): PyTorch BERT fine-tuned on domain qrels (`finetuned_reranker.py`, `train_finetuned.py`). ~50ms inference, domain-adapted for financial jargon and ticker symbols. Requires PyTorch + GPU for training (RTX 2060 6GB sufficient for DistilBERT).
- Fine-tuned reranker training script (`train_finetuned.py`): fine-tune DistilBERT on the same qrels as the sklearn tier, but learns end-to-end from raw text. Model weights gitignored; only metadata and metrics committed.
- `rag.py` updated to route through the multi-tier strategy based on operator configuration. All three tiers share the same first-stage candidate set.
- Docs: README Grounding section rewritten to explain the multi-tier strategy as an architectural decision, not a demo. Problems table extended with the new row.
- Suite 47 passing, 1 skipped (pgvector profile).

## 0.4.0 — 2026-09-18

- Hybrid retrieval: hand-rolled Okapi BM25 (`bm25.py`) — saturation, IDF, explicit length normalisation — with the old token-overlap scorer kept as a documented baseline; dense cosine + BM25 fused by reciprocal rank fusion (Cormack et al., SIGIR 2009) instead of incomparable raw-score blending.
- RankNet reranker (`reranker.py`): within-query feature differences mirrored for two classes, bias refitted back into slot 0; C selected by query-grouped 4-fold CV inside the train split, holdout read only by the acceptance gate.
- Acceptance-gated artifacts: `python -m helix.reranker --fit` writes `rerank_model.json`; the runtime ships a fitted rerank **only if it beats the hand-set prior on holdout nDCG@5 and MRR**. A rejected artifact degrades to the prior with the reason and metrics on record (current fit: 0.936 vs 0.987 — rejected, prior ships).
- RAG features extended for the rerank probe: lexical rank, RRF score, query-term coverage alongside cosine/ticker/title/recency.
- Evals: ablation across lexical/dense/hybrid/rerank stages on the golden set.
- Training path: TRL export verified end-to-end on the new qrels fixture (`python/tests/data/rerank_qrels.jsonl`, sha-pinned in the artifact).
- Docs: README Grounding section and the problems table updated for BM25/RRF/RankNet; suite 47 passing, 1 skipped (pgvector profile).
