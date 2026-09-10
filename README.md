# Helix — Voice agent orchestration

Voice-first copilot over **your** tools. Python owns the numbers (FastAPI, Pydantic, pandas, NumPy, scikit-learn). The LLM only speaks after tools return. Clone it, point `HELIX_MCP_URL` at your MCP server, talk.

[![CI](https://github.com/OlegUnreal/voice-agent-orchestration/actions/workflows/ci.yml/badge.svg)](https://github.com/OlegUnreal/voice-agent-orchestration/actions/workflows/ci.yml)

> **The model does not pick tools and does not invent prices, VaR, or positions.** That is the whole product.

## Why this exists

A chat box with a strong language model is good at talking. It is a weak broker — **including Grok in a chat, and every other general assistant in the same shape.** No slight: that is what those products are for. Helix exists because that shape fails at a few jobs they were never designed to do.

| Job | General chat (Grok chat, Claude, Gemini, local UIs — same class) | Helix |
|---|---|---|
| Last price, VaR, fill | Will **speak a number** if the tool is slow or missing | Speaks only numbers that already exist in tool JSON. Otherwise **refuses** |
| Write tools (order, cancel, kill-switch) | The model often **picks the tool** | The model never picks tools. Write tools are listed and **never auto-run** |
| Where did this fact come from? | Citations as decoration | Hops + tool JSON + source on every memory fact |
| After a week of use | Vendor logs you cannot train on | **Trace ledger** → SFT/DPO pairs from your thumbs |
| “Remember my risk cap” | Vibes in the weights | SQLite fact with `source` and `observedAt` |
| Agent spend | Invisible tool loops | Budget/latency on the turn |

That is why we built this: not a better conversationalist — a **control plane** between you and tools. The language model is the mouth. Helix is the hands that are not allowed to type a price or submit an order.

If you only want a conversation, use a chat product. If you want a voice layer over **your** MCP that cannot hallucinate a book, clone this.

Personal live wiring (Tailscale, OK-Trader) is **not** in this public tree.

## Skills that close those gaps

| Skill | Python | What it writes |
|---|---|---|
| Verifier | [`verifier.py`](python/helix/verifier.py) | `verify_fails.jsonl` when speech leaks a number |
| Trace ledger | [`ledger.py`](python/helix/ledger.py) | `traces.jsonl` every turn |
| Feedback | `POST /v1/feedback` | `preferences.jsonl` — SFT on thumbs-up, DPO on correction |
| Memory | [`memory.py`](python/helix/memory.py) | SQLite facts with provenance |

```bash
.venv/bin/helix turn "Remember that isolated 1x and 1 USDT risk cap"
.venv/bin/helix recall "risk cap"
.venv/bin/helix export          # SFT / DPO counts
```

Thumbs on each Helix reply in the voice UI. Data stays in `python/data/` (gitignored). Train later; collect now.

## Python engines

| Piece | In `python/helix/` |
|---|---|
| FastAPI + Pydantic | [`gateway/app.py`](python/helix/gateway/app.py) — structured routes, TTL cache, JSON-RPC `/mcp` |
| pandas / NumPy | [`market.py`](python/helix/market.py) — demo tape, regimes, anomalies, SMA backtest, VaR |
| RAG + embeddings + rerank | [`embeddings.py`](python/helix/embeddings.py), [`rag.py`](python/helix/rag.py), [`reranker.py`](python/helix/reranker.py) |
| scikit-learn | LogisticRegression next to NumPy SGD LoRA-style rerank probe |
| EvalForge | [`evals.py`](python/helix/evals.py) — golden set, Recall@K, MRR, nDCG, groundedness |
| TrainingOps | [`training.py`](python/helix/training.py) — qrels → adapters → staging/shadow/canary/production |
| MCP client | [`oktrader/mcp_client.py`](python/helix/oktrader/mcp_client.py) — Streamable HTTP `initialize` → `tools/list` → `tools/call` |
| Verifier | [`verifier.py`](python/helix/verifier.py) — speech may not contain numbers absent from tools |
| Ledger + prefs | [`ledger.py`](python/helix/ledger.py) — traces / SFT / DPO JSONL |
| Memory | [`memory.py`](python/helix/memory.py) — SQLite facts with source + time |

```bash
cd python
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
.venv/bin/helix serve
.venv/bin/helix turn "What is BTC's current regime?"
```

Voice UI talks to this gateway (`HELIX_ENGINE_URL`). If it is down, TypeScript engines keep the demo alive.

## Use it on your machine

```bash
# 1) optional: your MCP (any Streamable HTTP 2025-06-18 server)
export HELIX_MCP_URL='https://your-host/mcp'
export HELIX_ALLOW_SIMULATOR=false   # refuse fake prices if MCP is down

.venv/bin/helix mcp                  # initialize + tools/list
.venv/bin/helix turn "mark price and open positions"
.venv/bin/helix serve                # :8090
```

Copy [`python/mcp.env.example`](python/mcp.env.example). Without `HELIX_MCP_URL`, Helix runs the bundled demo tape so you can try the voice UI.

Write tools are discovered and **never auto-run**. Read tools (price, klines, positions, preflight, project context) are picked by intent.

```bash
# 2) voice UI
npm install && npm run dev
# optional: XAI_API_KEY for Grok narration; without it, fallbackSpoken is used
```


## Architecture


Helix is a **tool-first supervisor**, not ReAct and not a multi-LLM swarm. Two hops per turn: supervisor routes intent, one specialist calls a governed tool bundle.

```mermaid
flowchart TD
  U["User: STT or text"] --> SF["runHelixTurn TS shell"]
  SF --> GW["FastAPI gateway :8090"]
  GW --> ORC["run_orchestrator"]
  ORC --> SUP["supervisor: classifyIntent"]
  SUP --> Q["quant"]
  SUP --> C["copilot"]
  SUP --> R["research"]
  SUP --> E["eval"]
  SUP --> T["training"]
  Q --> TOOLS["governed tools"]
  C --> TOOLS
  R --> TOOLS
  E --> TOOLS
  T --> TOOLS
  TOOLS --> VER["verifier: numbers ⊆ tool JSON"]
  VER --> PAY["grokToolPayload JSON"]
  PAY --> GROK["optional Grok narration · temp 0.2"]
  GROK --> VER
  VER --> TTS["optional TTS"]
  VER --> LEDGER["traces.jsonl + prefs"]
  LEDGER --> UI["Voice UI: hops, tools, thumbs"]
```

A specialist is a **tool-selection policy**, not a second language model. For `portfolio`, copilot pulls snapshot + regime + anomalies + risk + RAG in one bundle.

### Agents

| Agent | Intents | Tools |
|---|---|---|
| **supervisor** | all | `classify_intent` in [`router.py`](python/helix/router.py) |
| **quant** | `regime`, `anomaly`, `backtest` | snapshot, regime, z-score events, SMA backtest |
| **copilot** | `risk`, `portfolio` | VaR / CVaR / scenario P&L |
| **research** | `research` | hybrid RAG + logistic rerank |
| **eval** | `eval` | golden-set metrics |
| **training** | `training` | checkpoint registry |

Intent → agent map lives in [`orchestrator.py`](python/helix/orchestrator.py):

```ts
const INTENT_AGENT = {
  regime: "quant",
  anomaly: "quant",
  backtest: "quant",
  risk: "copilot",
  research: "research",
  portfolio: "copilot",
  eval: "eval",
  training: "training",
  general: "supervisor",
};
```

### How the supervisor classifies

Two layers in [`src/lib/helix/router.ts`](src/lib/helix/router.ts), no round-trip to Grok:

1. **Lexical rules** (priority): backtest, VaR/shock, anomaly, regime, eval, LoRA/train, portfolio, research.
2. **Embedding prototypes**: hashing-trick vector of the query vs nine intent prototypes. Cosine `< 0.12` → `general`.

### Tools (MCP-style, not MCP wire protocol)

Registry in [`src/lib/helix/mcp.ts`](src/lib/helix/mcp.ts): JSON schemas + governance. Execution is inline in the orchestrator.

| Tool | Server | Governance |
|---|---|---|
| `get_market_snapshot` | market | auto |
| `detect_regime` | market | auto |
| `detect_anomalies` | market | auto |
| `run_backtest` | market | auto |
| `estimate_risk` | market | auto |
| `retrieve_evidence` | retrieval | auto |
| `get_eval_report` | eval | auto |
| `get_checkpoint_status` | eval | auto |
| `propose_rebalance` | market | **approve** — listed, never auto-run |

Each call is wrapped in `timed()` → `{ name, args, ms, ok, summary }`. That audit trail is what the Gateway view shows.

### One turn

```mermaid
sequenceDiagram
  participant UI as Voice UI
  participant SF as runHelixTurn
  participant ORC as Orchestrator
  participant ENG as Engines
  participant XAI as Grok-4.5

  UI->>SF: text (max 800)
  SF->>ORC: runOrchestrator(query)
  ORC->>ORC: classifyIntent
  ORC->>ENG: subset of tools by intent
  ENG-->>ORC: numbers + docs
  ORC-->>SF: hops, tools, citations, fallbackSpoken
  alt XAI_API_KEY and cache miss
    SF->>XAI: system: tools are the only source of numbers
    XAI-->>SF: 2–4 spoken sentences
  else cache / no key / HTTP fail
    SF-->>UI: fallbackSpoken from tools
  end
  UI->>SF: speakHelix optional TTS
```

Contract for the language model in [`src/lib/helix/chat.ts`](src/lib/helix/chat.ts):

> TOOL RESULTS are the only source of numbers. Never invent prices, Sharpe, VaR, or citations.

Payload is compact JSON: intent, tool summaries, `numbers`, citations, snapshot / risk / backtest, eval gates. Cache TTL is 10 minutes on the normalized query. If `XAI_API_KEY` is missing, the UI still runs — `fallbackSpoken` is authored from the same tool numbers.

### Grounding

A turn is `grounded` when citation ids exist **or** a quant tool fired (snapshot / backtest / risk). RAG is hybrid: cosine + token overlap + ticker/title hit + recency (14-day half-life) + a chronological mask (`doc.ts > asOf` is dropped). Rerank is a logistic adapter (LoRA analog). Up to four citation ids go to both the UI and Grok.

## What maps to the résumé

| Résumé system | In Helix |
|---|---|
| ChainLens Quant & Market Intelligence | Markets: GBM tape, regime labels, z-score anomalies, SMA crossover backtest, scenario P&L / VaR |
| MLLLM layer (embeddings, rerank, RAG, chronological eval) | RAG lab + hashed 64-d embeddings + recency + date mask |
| SFT / LoRA / QLoRA | TrainingOps: logistic adapters (r8 / r4) trained on golden qrels |
| ChainLens AI Gateway & Financial Copilot | Voice graph + Gateway: routing, cache, structured tools, traces, approval on `propose_rebalance` |
| EvalForge TrainingOps & Model Registry | EvalForge metrics + staging → shadow → canary → production gates |
| MCP, FastAPI/Pydantic-style tools | MCP tool registry with JSON schemas |
| Voice agent orchestration | Orb STT → supervisor → specialist tools → Grok synthesis → TTS |

| Pattern | In this repo | Production next step |
|---|---|---|
| Supervisor + specialists | 2-hop graph | LangGraph state machine, retry, parallel specialists |
| MCP tools | Schemas + governance labels | Real MCP server + JSON-RPC |
| Governed execution | `auto` vs `approve` in the registry | Interrupt / HITL on `propose_rebalance` |
| Structured outputs | JSON payload → Grok | JSON schema / tool-calling on synthesis |
| Model routing + cache | grok-4.5 / cache / local-tools | Dedicated gateway router |
| Traces | Seed traces + live hops | Persistent traces → feedback dataset |

## Local engines (no GPU)

- Deterministic market simulator and risk math
- Hybrid retrieval + logistic reranker you can actually train
- Golden set: Recall@K, MRR, nDCG, intent/tool accuracy, groundedness, p95, cost

## Layout

```
python/helix/            FastAPI gateway + engines (source of truth)
src/lib/helix/           TypeScript shell + fallback
src/lib/helix/engine.ts  HTTP client to the Python gateway
src/components/helix/    Voice, Markets, RAG, Gateway, Evals, Training
```


## Run

```bash
npm ci
npm run typecheck
npm run build
npm run dev
```

Optional: set `XAI_API_KEY` for live Grok synthesis and TTS. Without it, the quant / RAG / eval engines still run and Helix falls back to tool-authored speech.

Try: *What is BTC’s current regime?* · *Portfolio risk if BTC drops 12 percent.*
