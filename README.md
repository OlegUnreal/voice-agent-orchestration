# Helix — Voice agent orchestration

Portfolio implementation of the AI/ML stack on Oleh Kornii’s résumé, as a **voice-first** copilot. No Java. Python-style engines (regimes, RAG, evals, LoRA-style rerank) run in the app; Grok is used only to speak after tools return numbers.

[![CI](https://github.com/OlegUnreal/voice-agent-orchestration/actions/workflows/ci.yml/badge.svg)](https://github.com/OlegUnreal/voice-agent-orchestration/actions/workflows/ci.yml)

## What maps to the résumé

| Résumé system | In Helix |
|---|---|
| ChainLens Quant & Market Intelligence | Markets: GBM tape, regime labels, z-score anomalies, SMA crossover backtest, scenario P&L / VaR |
| MLLLM layer (embeddings, rerank, RAG, chronological eval) | RAG lab + hashed 64-d embeddings + recency + date mask |
| SFT / LoRA / QLoRA | TrainingOps: logistic adapters (r8 / r4) trained on golden qrels |
| ChainLens AI Gateway & Financial Copilot | Voice graph + Gateway: routing, cache, structured tools, traces, approval on `propose_rebalance` |
| EvalForge TrainingOps & Model Registry | EvalForge metrics + staging → shadow → canary → production gates |
| MCP, FastAPI/Pydantic-style tools | MCP tool registry with JSON schemas |
| Voice agent orchestration | Orb STT (browser) → supervisor → specialist tools → Grok synthesis → TTS |

## Voice path

1. Speak or type.
2. Supervisor classifies intent (regime, anomaly, backtest, risk, research, eval, training).
3. Specialist agents call governed tools. **Prices, Sharpe, VaR never come from the LLM.**
4. Time-aware RAG attaches citation ids.
5. Grok-4.5 turns the tool JSON into a short spoken answer (capped tokens, cached).
6. Optional TTS (`orion`).

## Local engines (no GPU required)

- Deterministic market simulator and risk math
- Hybrid retrieval + logistic reranker you can actually train
- Golden set: Recall@K, MRR, nDCG, intent/tool accuracy, groundedness, p95, cost

## Layout

```
src/lib/helix/           engines (market, rag, evals, training, orchestrator, mcp)
src/lib/helix/chat.ts    Grok synthesis + TTS (server)
src/components/helix/    Voice, Markets, RAG, Gateway, Evals, Training
```

## Run

```bash
npm ci
npm run typecheck
npm run build
npm run dev
```

Optional: set `XAI_API_KEY` for live Grok synthesis and TTS. Without it, the quant/RAG/eval engines still run and Helix falls back to tool-authored speech.

Try: *What is BTC’s current regime?* · *Portfolio risk if BTC drops 12 percent.*
