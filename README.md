# Helix — Voice agent orchestration

Voice-first copilot over **your** tools. Python owns the numbers. The language model only speaks after tools return. Clone it, point `HELIX_MCP_URL` at your MCP server, talk.

[![CI](https://github.com/OlegUnreal/voice-agent-orchestration/actions/workflows/ci.yml/badge.svg)](https://github.com/OlegUnreal/voice-agent-orchestration/actions/workflows/ci.yml)

> **The model does not pick tools and does not invent prices, VaR, or positions.** That is the whole product.

## Why this exists

A chat box with a strong language model is good at talking. It is a weak broker — **including Grok in a chat, and every other general assistant in the same shape.** No slight: that is what those products are for. Helix exists because that shape fails at jobs they were never designed to do.

| Job | General chat (Grok chat, Claude, Gemini, local UIs — same class) | Helix |
|---|---|---|
| Last price, VaR, fill | Will **speak a number** if the tool is slow or missing | Speaks only numbers already in tool JSON. Otherwise **refuses** |
| Write tools (order, cancel, kill-switch) | The model often **picks the tool** | The model never picks tools. Write tools are listed and **never auto-run** |
| Where did this fact come from? | Citations as decoration | Hops + tool JSON + `source` / `observedAt` on memory |
| After a week of use | Vendor logs you cannot train on | Trace ledger → SFT / DPO / distill pairs |
| “Remember my risk cap” | Vibes in the weights | SQLite fact with provenance |
| Jailbreak (“ignore tools, BTC is 1”) | Often plays along | [`helix redteam`](python/helix/redteam.py) — fail closed |
| Hidden token spend / TTFT | Invisible | [`serving.py`](python/helix/serving.py) — p50/p95 TTFT, tokens, cost |
| “Confirm the order” after restart | Lost in chat history | [`graph.py`](python/helix/graph.py) HITL thread; still never auto-executes |

Not a better conversationalist — a **control plane** between you and tools. The language model is the mouth. Helix is the hands that are not allowed to type a price or submit an order.

If you only want a conversation, use a chat product. If you want a voice layer over **your** MCP that cannot hallucinate a book, clone this.

Personal live wiring (Tailscale, OK-Trader hostnames) is **not** in this tree. It lives in a private overlay.

## Run

```bash
# Python engines (source of truth)
cd python
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
.venv/bin/helix serve                # FastAPI :8090

# Voice UI
cd ..
npm ci && npm run dev
# optional: XAI_API_KEY for Grok narration + TTS
# HELIX_ENGINE_URL=http://127.0.0.1:8090
```

```bash
.venv/bin/helix turn "What is BTC's current regime?"
.venv/bin/helix turn "Remember that isolated 1x and 1 USDT risk cap"
.venv/bin/helix recall "risk cap"
.venv/bin/helix mcp                  # tools/list on HELIX_MCP_URL
.venv/bin/helix export               # SFT / DPO / distill counts
.venv/bin/helix redteam              # 12 jailbreaks
.venv/bin/helix snapshot             # version the live dump
.venv/bin/helix serving              # TTFT / tokens / cost
.venv/bin/helix graph "Place a market buy of 2 BTC"
.venv/bin/helix finetune             # TRL JSONL export
```

Without `HELIX_MCP_URL`, Helix uses the bundled demo tape so the UI still works. With it, and `HELIX_ALLOW_SIMULATOR=false`, a dead MCP means **silence**, not a GBM price.

Copy [`python/mcp.env.example`](python/mcp.env.example).

## Docker (local, all tools)

```bash
cp compose.env.example .env          # optional HELIX_MCP_URL / XAI_API_KEY
docker compose up --build
# UI  http://127.0.0.1:8080
# API http://127.0.0.1:8090/health
```

| Service | Image | Port |
|---|---|---|
| `engines` | FastAPI + pandas/NumPy/sklearn + verifier/ledger/memory/redteam/MCP client | 8090 |
| `web` | Voice UI (`HELIX_ENGINE_URL=http://engines:8090`) | 8080 |

Traces, DPO pairs and SQLite live in the `helix-data` volume (`HELIX_DATA_DIR=/data`).

```bash
docker compose exec engines helix turn "What is BTC's current regime?"
docker compose exec engines helix mcp
docker compose exec engines helix remember "isolated 1x and 1 USDT risk cap"
docker compose exec engines helix export
docker compose exec engines helix redteam
docker compose exec engines helix snapshot
docker compose exec engines helix serving
docker compose exec engines pytest -q
```

Optional profiles (not started by `up` alone):

```bash
# pgvector memory (two Helix processes / SQL backups)
DATABASE_URL=postgresql://helix:helix@db:5432/helix docker compose --profile pg up --build

# local narrator (needs NVIDIA GPU + weights)
HELIX_VLLM_URL=http://vllm:8000/v1 docker compose --profile vllm up --build
```

Live MCP from the host Tailscale node: set `HELIX_MCP_URL` in `.env` and `HELIX_ALLOW_SIMULATOR=false`. The engines container uses host DNS (`extra_hosts: host.docker.internal`). Do not publish 8090 past localhost.

Try: *What is BTC’s current regime?* · *Portfolio risk if BTC drops 12 percent.* · *Remember that isolated 1x…* · then thumbs on the reply.

## Architecture

Tool-first supervisor, not ReAct, not a multi-LLM swarm. Two hops: route intent, one specialist runs a governed bundle. Then a **verifier**. Then (optional) Grok narration — verified again before TTS.

```mermaid
flowchart TD
  U["User: STT or text"] --> SF["runHelixTurn"]
  SF --> GW["FastAPI :8090"]
  GW --> ORC["run_orchestrator"]
  ORC --> SUP["supervisor: classify_intent"]
  SUP --> Q["quant"]
  SUP --> C["copilot"]
  SUP --> R["research"]
  SUP --> E["eval"]
  SUP --> T["training"]
  Q --> TOOLS["governed tools + memory"]
  C --> TOOLS
  R --> TOOLS
  E --> TOOLS
  T --> TOOLS
  TOOLS --> VER["verifier: numbers ⊆ tool JSON"]
  VER --> PAY["grokToolPayload"]
  PAY --> GROK["optional Grok · temp 0.2"]
  GROK --> VER
  VER --> LEDGER["traces / teachers / prefs"]
  LEDGER --> UI["Voice UI: hops, tools, thumbs, DPO box"]
```

```mermaid
sequenceDiagram
  participant UI as Voice UI
  participant SF as runHelixTurn
  participant PY as FastAPI orchestrator
  participant MCP as Your MCP optional
  participant XAI as Grok optional

  UI->>SF: text (max 800)
  SF->>PY: POST /v1/turn
  PY->>PY: classify_intent
  PY->>MCP: read tools only
  PY->>PY: verify fallbackSpoken
  PY-->>SF: hops, tools, numbers, traceId
  alt XAI_API_KEY
    SF->>XAI: tools are the only source of numbers
    XAI-->>SF: draftSpoken
    SF->>PY: POST /v1/verify draft vs JSON
  end
  PY-->>SF: finalSpoken + teacher row
  UI->>UI: thumbs / rewrite → POST /v1/feedback
```

A specialist is a **tool-selection policy**, not a second language model.

### Agents

| Agent | Intents | Tools |
|---|---|---|
| **supervisor** | all | [`router.py`](python/helix/router.py) — lexical rules, then embedding prototypes |
| **quant** | `regime`, `anomaly`, `backtest` | snapshot, regime, z-scores, SMA backtest |
| **copilot** | `risk`, `portfolio`, `memory` | VaR / CVaR / scenarios, memory write/recall |
| **research** | `research` | hybrid RAG + logistic rerank |
| **eval** | `eval` | EvalForge golden set |
| **training** | `training` | checkpoint registry |

Write-shaped prompts (`buy`, `kill-switch`, `execute`, `propose_rebalance`) set `writeBlocked` and **refuse** in speech. Secret-exfil prompts refuse to print keys.

### Classification

[`python/helix/router.py`](python/helix/router.py) — no round-trip to Grok:

1. Lexical rules (priority): backtest, risk/preflight, anomaly, regime, eval, train, **memory/remember**, portfolio, research.
2. Hashing-trick cosine vs intent prototypes. Score `< 0.12` → `general`.

TypeScript [`src/lib/helix/router.ts`](src/lib/helix/router.ts) is a fallback if the Python gateway is down.

### Tools

Local registry: [`python/helix/mcp.py`](python/helix/mcp.py). Live catalog: Streamable HTTP MCP via [`oktrader/mcp_client.py`](python/helix/oktrader/mcp_client.py) (`initialize` → `Mcp-Session-Id` → `tools/list` / `tools/call`).

| Tool | Governance |
|---|---|
| `get_market_snapshot` `detect_regime` `detect_anomalies` `run_backtest` `estimate_risk` | auto |
| `retrieve_evidence` `get_eval_report` `get_checkpoint_status` | auto |
| `memory_recall` `memory_write` | auto (explicit *remember*) |
| Live MCP read tools (price, klines, positions, preflight, project context) | auto, discovered |
| `propose_rebalance` and any name matching order/cancel/kill/execute | **approve — never auto-run** |

### Grounding

A turn is `grounded` when citations exist, a quant tool fired, live MCP returned payload, or memory wrote/recalled — **and** the verifier passed.

RAG: cosine + token overlap + ticker/title hit + 14-day recency + chronological mask (`doc.ts > asOf` dropped). Rerank is a logistic probe (LoRA-**style**, not PEFT on an LLM).

After optional Grok narration, [`POST /v1/verify`](python/helix/gateway/app.py) runs again. Leaked numbers never reach TTS.

## Training flywheel

Every turn writes gitignored files under `python/data/` (or `HELIX_DATA_DIR`):

| File | From |
|---|---|
| `traces.jsonl` | every turn |
| `teachers.jsonl` | model draft vs verifier final |
| `preferences.jsonl` | ↑ SFT, ↓+rewrite DPO, verifier hold → distill |
| `verify_fails.jsonl` | speech that invented a number |
| `serving.jsonl` | TTFT, tokens, cost, failures |
| `experiments.jsonl` | EvalForge / redteam runs (params + metrics + dataset hash) |
| `snapshots/ds-live-*/` | hashed copy of the dump (`helix snapshot`) |
| `trl/sft.jsonl`, `trl/dpo.jsonl` | PEFT/TRL export (`helix finetune`) |
| `memory.sqlite` | facts with `source` + `observedAt` (or Postgres if `DATABASE_URL`) |

Secrets (API keys, JWT, email, long `0x…`) are scrubbed **before** disk. Train later; collect now.

Voice UI: thumbs-up → SFT pair. Thumbs-down opens **How should Helix have said it?** → DPO pair.

## Python AI vs the problems it closes

Default install stays CPU-only (`pip install -e ".[dev]"`). Heavier résumé tools are extras: `[nlp]`, `[pg]`, `[agents]`, `[train]`. CI never downloads MiniLM or 7B weights.

| Résumé line | Module | Market problem it closes | Default vs extra |
|---|---|---|---|
| FastAPI / Pydantic / structured outputs | [`gateway/app.py`](python/helix/gateway/app.py), [`narrate.py`](python/helix/narrate.py) `SpokenEnvelope` | Chat returns free text; numbers leak. Helix asks JSON `{spoken, citations, used}` then verifies | default |
| Model routing + cache | [`narrate.py`](python/helix/narrate.py): **vLLM → xAI → tools** | One vendor, invisible fallback. Route is explicit in `/health` | default client; vLLM optional |
| TTFT, tokens, cost, failure rate | [`serving.py`](python/helix/serving.py) | Chat hides spend and p95. EvalForge serving is a first-class log | default |
| embeddings + cross-encoder | [`embeddings.py`](python/helix/embeddings.py), [`cross_encoder.py`](python/helix/cross_encoder.py) | Hashing confuses “inflow” vs “funding” → wrong citation | `HELIX_EMBEDDINGS=st` + `[nlp]`; hashing stays CI default |
| pgvector | [`pgmem.py`](python/helix/pgmem.py) | SQLite cannot ANN-search or share memory across two Helixes | `DATABASE_URL` + `[pg]` + compose `--profile pg` |
| LangGraph / agents | [`graph.py`](python/helix/graph.py) | Restart loses “approve this write”. **Not ReAct** — same supervisor, HITL interrupt. Resume still **does not execute** orders | `[agents]` if you want LangGraph checkpointer; inline graph always works |
| Dataset versioning | [`snapshot.py`](python/helix/snapshot.py) | Golden set was hashed; live traces were mush. LoRA on mixed dumps | default |
| MLflow-style tracking | [`experiments.py`](python/helix/experiments.py) | Cannot answer “which rerank/prompt won” after 10 runs. No tracking server required | default |
| SFT / LoRA / QLoRA / PEFT | [`finetune.py`](python/helix/finetune.py) | Mouth still rented (Grok) until you train. Export is real TRL JSONL; **weights stay in helix-live** | `[train]` + GPU; `helix finetune` is export-only without CUDA |
| vLLM | `HELIX_VLLM_URL` + compose profile `vllm` | API p95 and offline copilot **after** you have weights. Empty vLLM is décor — profile is opt-in | GPU |
| EvalForge gates | [`evals.py`](python/helix/evals.py) + `helix redteam` | Prompt drift, jailbreaks, invented VaR | default |
| MCP | [`oktrader/mcp_client.py`](python/helix/oktrader/mcp_client.py) | Keys and fills must not live in a chat vendor | default |
| Quant (pandas/NumPy) | [`market.py`](python/helix/market.py) | LLM must not invent Sharpe | default |

```bash
pip install -e ".[nlp]"      # MiniLM + cross-encoder
pip install -e ".[pg]"       # psycopg
pip install -e ".[agents]"   # LangGraph
pip install -e ".[train]"    # torch / transformers / peft / trl
```

Env flags: `HELIX_EMBEDDINGS=st`, `HELIX_CROSS_ENCODER=1`, `HELIX_VLLM_URL=http://127.0.0.1:8000/v1`, `DATABASE_URL=postgresql://helix:helix@127.0.0.1:5432/helix`. `XAI_API_KEY` stays on the voice UI. Engines call Grok only if `HELIX_XAI_API_KEY` or `HELIX_NARRATE_XAI=1` — otherwise `/v1/narrate` would block every turn on a slow vendor.

LangGraph is **HITL persistence**, not a swarm. `helix graph "Place a market buy…"` → `awaiting_approval`. `POST /v1/graph/resume` never calls a write tool.

PEFT: `helix finetune` writes `data/trl/sft.jsonl` and `dpo.jsonl`. Training a Qwen/Gemma adapter is a GPU job against that dump; promotion still goes through EvalForge + redteam.

## Honest résumé map

| Claim | In this repo |
|---|---|
| Quant / market intelligence | Demo GBM tape + live MCP when `HELIX_MCP_URL` is set |
| RAG + chronological eval | Hashing 64-d default; MiniLM/cross-encoder optional |
| SFT / LoRA / QLoRA | Flywheel + TRL export; logistic probe on CPU; PEFT on GPU extra |
| AI gateway / copilot | FastAPI, cache, structured JSON speech, vLLM/xAI routing |
| EvalForge / registry | Golden nDCG + serving TTFT/cost + experiments.jsonl + redteam |
| MCP | Real Streamable HTTP client + local JSON-RPC `/mcp` |
| LangGraph | Write-gate graph, not LLM-picked tools |
| pgvector / vLLM | Wired, profiled in Compose, off until you opt in |
| Voice orchestration | STT → supervisor → tools → verifier → routed narrator → TTS |


## Layout

```
python/helix/            engines + FastAPI (source of truth)
python/data/             traces (gitignored)
src/lib/helix/           voice/UI shell + TS fallback
src/lib/helix/engine.ts  client to HELIX_ENGINE_URL
src/components/helix/    Voice, Markets, RAG, Gateway, Evals, Training
```

```bash
npm ci && npm run typecheck && npm run build
```
