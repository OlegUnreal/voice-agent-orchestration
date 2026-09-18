# Helix Architecture

This document describes the architecture of the Helix voice-agent orchestration system. All diagrams use Mermaid syntax and render on GitHub.

---

## System Context (C4 Level 1)

Who uses the system and what external systems it interacts with.

```mermaid
graph TB
    subgraph "External"
        U["User<br/>(Voice or Text)"]
        MCP["Your MCP Server<br/>(Optional)"]
        XAI["xAI Grok API<br/>(Optional)"]
    end
    
    subgraph "Helix System"
        UI["Voice UI<br/>(React/Next.js)"]
        GW["FastAPI Gateway<br/>(Python)"]
        ORC["Orchestrator<br/>(Intent Router + Specialists)"]
        TOOLS["Governed Tools<br/>(Market, Risk, RAG, Memory)"]
        VER["Verifier<br/>(Numbers ⊆ Tool JSON)"]
        LEDGER["Trace Ledger<br/>(traces, teachers, preferences)"]
    end
    
    U -->|STT or text| UI
    UI -->|POST /v1/turn| GW
    GW --> ORC
    ORC --> TOOLS
    TOOLS -->|read tools only| MCP
    ORC --> VER
    VER -->|optional narration| XAI
    VER --> LEDGER
    LEDGER -->|traces, feedback| UI
    GW -->|structured JSON| UI
    UI -->|TTS| U
```

**Key boundaries**:
- **User**: speaks or types. Never sees raw tool JSON.
- **Voice UI**: STT → gateway → TTS. Fallback router if Python is down.
- **FastAPI Gateway**: single entry point. Routes to orchestrator, returns structured JSON.
- **Orchestrator**: intent classification → specialist → tools → verifier.
- **Governed Tools**: auto-run for read, human-approve for write. Never auto-execute orders.
- **Verifier**: checks spoken response against tool JSON. Leaked numbers never reach TTS.
- **Trace Ledger**: every turn writes traces, teachers, preferences. Train later; collect now.

---

## Container Diagram (C4 Level 2)

What's inside the Helix system and how the containers interact.

```mermaid
graph TB
    subgraph "Voice UI (TypeScript)"
        STT["STT<br/>(Web Speech API)"]
        TTS["TTS<br/>(xAI or local)"]
        ROUTER["Fallback Router<br/>(TypeScript intent)"]
        CLIENT["Helix Client<br/>(typed HTTP)"]
    end
    
    subgraph "FastAPI Gateway (Python)"
        APP["gateway/app.py<br/>(FastAPI + Pydantic)"]
        CACHE["TtlCache<br/>(600s TTL)"]
        SUP["supervisor<br/>(classify_intent)"]
        SPEC["specialists<br/>(quant, copilot, research, eval, training)"]
        VER["verifier<br/>(enforce)"]
        NAR["narrator<br/>(vLLM → xAI → tools)"]
    end
    
    subgraph "Engines (Python)"
        MARKET["market.py<br/>(pandas/NumPy)"]
        RAG["rag.py<br/>(BM25 + dense + RRF)"]
        RERANK["reranker.py<br/>(RankNet / CrossEncoder / BERT)"]
        MEMORY["memory.py<br/>(SQLite / pgvector)"]
        ROUTER_P["router.py<br/>(lexical + cosine)"]
    end
    
    subgraph "Data (gitignored)"
        TRACES["traces.jsonl"]
        TEACHERS["teachers.jsonl"]
        PREFS["preferences.jsonl"]
        SERVING["serving.jsonl"]
        EXPERIMENTS["experiments.jsonl"]
        MEMORY_DB["memory.sqlite"]
    end
    
    STT --> CLIENT
    CLIENT -->|POST /v1/turn| APP
    APP --> CACHE
    APP --> SUP
    SUP --> ROUTER_P
    SUP --> SPEC
    SPEC --> MARKET
    SPEC --> RAG
    RAG --> RERANK
    SPEC --> MEMORY
    SPEC --> VER
    VER --> NAR
    NAR -->|optional| XAI["xAI Grok"]
    VER --> TRACES
    VER --> TEACHERS
    VER --> PREFS
    APP --> SERVING
    APP --> EXPERIMENTS
    MEMORY --> MEMORY_DB
    CLIENT --> TTS
    TTS -->|audio| U["User"]
    ROUTER -.->|if gateway down| CLIENT
```

**Container responsibilities**:
- **Voice UI**: STT/TTS, fallback router, typed client to gateway.
- **FastAPI Gateway**: request routing, caching, orchestration, verification, narration.
- **Engines**: market data, RAG, reranking, memory, intent classification.
- **Data**: gitignored files under `python/data/`. Traces, teachers, preferences, serving, experiments, memory.

---

## Component Diagram (C4 Level 3)

What's inside the FastAPI Gateway and how the components interact.

```mermaid
graph TB
    subgraph "FastAPI Gateway"
        direction TB
        ENDPOINTS["Endpoints<br/>/v1/turn, /v1/turn/stream, /v1/ws/turn"]
        CACHE["TtlCache<br/>(600s TTL, LRU)"]
        ORC["run_orchestrator<br/>(helix/orchestrator.py)"]
        ROUTER["classify_intent<br/>(helix/router.py)"]
        SUPERVISOR["supervisor<br/>(intent → specialist)"]
        QUANT["quant specialist<br/>(market, regime, anomaly, backtest)"]
        COPILOT["copilot specialist<br/>(risk, portfolio, memory)"]
        RESEARCH["research specialist<br/>(RAG + rerank)"]
        EVAL["eval specialist<br/>(EvalForge golden set)"]
        TRAINING["training specialist<br/>(checkpoint registry)"]
        VER["enforce<br/>(helix/verifier.py)"]
        NAR["narrate<br/>(helix/narrate.py)"]
    end
    
    ENDPOINTS --> CACHE
    ENDPOINTS --> ORC
    ORC --> ROUTER
    ORC --> SUPERVISOR
    SUPERVISOR --> QUANT
    SUPERVISOR --> COPILOT
    SUPERVISOR --> RESEARCH
    SUPERVISOR --> EVAL
    SUPERVISOR --> TRAINING
    QUANT --> VER
    COPILOT --> VER
    RESEARCH --> VER
    EVAL --> VER
    TRAINING --> VER
    VER --> NAR
    NAR -->|optional| XAI["xAI Grok API"]
```

**Component responsibilities**:
- **Endpoints**: REST, SSE, WebSocket. All route to `run_orchestrator`.
- **TtlCache**: 600s TTL, LRU eviction. Identical queries within TTL return cached response.
- **run_orchestrator**: classify intent → route to specialist → run tools → verify → narrate.
- **classify_intent**: lexical rules (priority) → hashing-trick cosine vs intent prototypes. Score < 0.12 → `general`.
- **supervisor**: maps intent to specialist. Each specialist is a tool-selection policy, not a second LLM.
- **specialists**: quant (market, regime, anomaly, backtest), copilot (risk, portfolio, memory), research (RAG + rerank), eval (golden set), training (checkpoints).
- **enforce**: checks spoken response against tool JSON. Leaked numbers → fallback spoken.
- **narrate**: optional Grok narration. Route: vLLM → xAI → tools. If all fail, use fallback spoken.

---

## Data Flow: Single Turn

```mermaid
sequenceDiagram
    participant U as User
    participant UI as Voice UI
    participant GW as FastAPI Gateway
    participant ORC as Orchestrator
    participant ROUTER as Intent Router
    participant SPEC as Specialist
    participant TOOLS as Governed Tools
    participant VER as Verifier
    participant NAR as Narrator
    participant XAI as xAI Grok (optional)
    participant LEDGER as Trace Ledger
    
    U->>UI: "What is BTC regime?"
    UI->>GW: POST /v1/turn {text: "..."}
    GW->>GW: check cache (600s TTL)
    alt cache hit
        GW-->>UI: cached response
    else cache miss
        GW->>ORC: run_orchestrator(text)
        ORC->>ROUTER: classify_intent(text)
        ROUTER-->>ORC: intent="regime"
        ORC->>SPEC: supervisor(intent="regime")
        SPEC->>TOOLS: get_market_snapshot("BTC")
        TOOLS-->>SPEC: {price: 65000, ...}
        SPEC->>TOOLS: detect_regime("BTC")
        TOOLS-->>SPEC: {regime: "high-vol", ...}
        SPEC-->>ORC: {spoken: "...", citations: [...], used: [...]}
        ORC->>VER: enforce(spoken, payload)
        VER->>VER: check numbers ⊆ tool JSON
        alt numbers leak
            VER-->>ORC: {ok: false, spoken: fallbackSpoken}
        else numbers ok
            VER-->>ORC: {ok: true, spoken: spoken}
        end
        ORC->>NAR: narrate(query, payload, spoken)
        NAR->>XAI: optional Grok narration
        XAI-->>NAR: draftSpoken
        NAR->>VER: POST /v1/verify draft vs JSON
        VER-->>NAR: finalSpoken
        NAR-->>ORC: finalSpoken
        ORC-->>GW: {spoken, citations, used, traceId, verified}
        GW->>LEDGER: log trace, teacher, serving
        GW-->>UI: response
    end
    UI->>U: TTS(finalSpoken)
    U->>UI: thumbs up/down
    UI->>GW: POST /v1/feedback {traceId, verdict}
    GW->>LEDGER: log preference (SFT/DPO)
```

**Key invariants**:
- Cache checked before orchestrator. Identical queries within 600s return cached response.
- Intent classified without LLM. Lexical rules + cosine similarity.
- Specialist is a tool-selection policy, not a second LLM.
- Verifier runs twice: once after specialist, once after optional Grok narration.
- Leaked numbers never reach TTS. Fallback spoken is used instead.
- Every turn writes to trace ledger. Train later; collect now.

---

## Retrieval Pipeline

```mermaid
graph LR
    subgraph "First Stage"
        Q["Query"] --> BM25["BM25<br/>(k1=1.4, b=0.4, title x2)"]
        Q --> DENSE["Dense Cosine<br/>(hashing-64 or MiniLM)"]
        BM25 --> CAND["Candidates<br/>(k=10)"]
        DENSE --> CAND
    end
    
    subgraph "Fusion"
        CAND --> RRF["Reciprocal Rank Fusion<br/>(Cormack et al., SIGIR 2009)"]
    end
    
    subgraph "Reranking (3 tiers)"
        RRF --> FAST["Fast Tier<br/>(RankNet, 1ms, sklearn)"]
        RRF --> ACC["Accurate Tier<br/>(CrossEncoder, 50ms, HF)"]
        RRF --> FT["Fine-tuned Tier<br/>(BERT, 50ms, PyTorch)"]
    end
    
    FAST --> OUT["Ranked Results"]
    ACC --> OUT
    FT --> OUT
```

**First stage**: BM25 + dense cosine. BM25 uses Okapi saturation, IDF, explicit length normalization. Dense uses hashing-trick (64-d, CI default) or MiniLM (optional, `[nlp]` extra).

**Fusion**: reciprocal rank fusion (RRF). Fuses on rank, not score. Scores from BM25 and dense cosine are incomparable; ranks are not.

**Reranking**: three tiers, selected by `HELIX_RERANKER_TIER` env var.
- **Fast** (default): RankNet pairwise logistic probe on 9 hand-crafted features. ~1ms inference, interpretable weights, CPU-only. Acceptance gate: ships only if it beats the hand prior on holdout nDCG@5.
- **Accurate**: HF CrossEncoder pre-trained on MS-MARCO. ~50ms inference, black-box, requires `sentence-transformers`.
- **Fine-tuned**: PyTorch BERT fine-tuned on domain qrels. ~50ms inference, domain-adapted for financial jargon and ticker symbols. Requires GPU for training (RTX 2060 6GB sufficient for DistilBERT).

All three tiers share the same first-stage candidate set.

---

## Training Flywheel

```mermaid
graph TB
    subgraph "Every Turn"
        TURN["Turn<br/>(query, spoken, tools, citations)"]
        TURN --> TRACES["traces.jsonl<br/>(every turn)"]
        TURN --> TEACHERS["teachers.jsonl<br/>(model draft vs verifier final)"]
        TURN --> PREFS["preferences.jsonl<br/>(↑ SFT, ↓+rewrite DPO, verifier hold → distill)"]
        TURN --> VERIFY["verify_fails.jsonl<br/>(speech that invented a number)"]
        TURN --> SERVING["serving.jsonl<br/>(TTFT, tokens, cost, failures)"]
    end
    
    subgraph "Periodic"
        SNAPSHOT["helix snapshot<br/>(hashed copy of live dump)"]
        FINETUNE["helix finetune<br/>(TRL JSONL export)"]
        REDTEAM["helix redteam<br/>(12 jailbreaks)"]
        EVALS["helix evals<br/>(golden set)"]
    end
    
    subgraph "Training (GPU)"
        SFT["SFT<br/>(data/trl/sft.jsonl)"]
        DPO["DPO<br/>(data/trl/dpo.jsonl)"]
        DISTILL["Distillation<br/>(verifier-held traces)"]
        TRAIN["Train LoRA/QLoRA<br/>(PEFT)"]
        PROMOTE["Promote<br/>(EvalForge + redteam gates)"]
    end
    
    TRACES --> SNAPSHOT
    TRACES --> FINETUNE
    TEACHERS --> FINETUNE
    PREFS --> FINETUNE
    FINETUNE --> SFT
    FINETUNE --> DPO
    FINETUNE --> DISTILL
    SFT --> TRAIN
    DPO --> TRAIN
    DISTILL --> TRAIN
    TRAIN --> PROMOTE
    PROMOTE -->|if gates pass| DEPLOY["Deploy to production"]
    REDTEAM --> PROMOTE
    EVALS --> PROMOTE
```

**Every turn writes**:
- `traces.jsonl`: query, intent, tools used, citations, latency, tokens, cost.
- `teachers.jsonl`: model draft vs verifier final. If verifier changed the spoken response, the diff is a training signal.
- `preferences.jsonl`: thumbs-up → SFT pair. Thumbs-down + rewrite → DPO pair. Verifier hold → distill pair.
- `verify_fails.jsonl`: speech that invented a number. Negative examples for safety training.
- `serving.jsonl`: TTFT, tokens in/out, cost, failure rate. For monitoring, not training.

**Periodic**:
- `helix snapshot`: hashed copy of the live dump. Version the dataset so you can answer "which reranker won on which dataset version."
- `helix finetune`: export to TRL JSONL for SFT/DPO. Training is a separate GPU job.
- `helix redteam`: 12 jailbreak prompts. Fail closed, not open.
- `helix evals`: golden set with Recall@K, MRR, nDCG, groundedness, citation correctness.

**Training (GPU)**:
- SFT: supervised fine-tuning on thumbs-up traces.
- DPO: direct preference optimization on thumbs-down + rewrite traces.
- Distillation: verifier-held traces are high-quality, no-noise examples.
- Train LoRA/QLoRA: parameter-efficient fine-tuning with PEFT.
- Promote: EvalForge gates (nDCG ≥ 0.72, Recall@5 ≥ 0.80, groundedness ≥ 0.85, p95 < 800ms, cost < $0.02) + redteam pass rate. If gates fail, model doesn't ship.

---

## Deployment (Docker Compose)

```mermaid
graph TB
    subgraph "Docker Compose"
        subgraph "engines container"
            GW["FastAPI Gateway<br/>:8090"]
            ENG["Engines<br/>(market, RAG, rerank, memory)"]
        end
        subgraph "web container"
            UI["Voice UI<br/>:8080"]
        end
        subgraph "helix-data volume"
            DATA["traces, teachers, preferences,<br/>serving, experiments, memory.sqlite"]
        end
    end
    
    subgraph "Optional Profiles"
        subgraph "pg profile"
            DB["PostgreSQL<br/>+ pgvector"]
        end
        subgraph "vllm profile"
            VLLM["vLLM<br/>(local narrator)"]
        end
    end
    
    UI -->|HELIX_ENGINE_URL=http://engines:8090| GW
    GW --> ENG
    ENG --> DATA
    GW -.->|DATABASE_URL| DB
    GW -.->|HELIX_VLLM_URL| VLLM
    
    U["User"] -->|http://127.0.0.1:8080| UI
    U -->|http://127.0.0.1:8090/health| GW
```

**Default compose**:
- `engines`: FastAPI + pandas/NumPy/sklearn + verifier/ledger/memory/redteam/MCP client. Port 8090.
- `web`: Voice UI (React/Next.js). Port 8080. `HELIX_ENGINE_URL=http://engines:8090`.
- `helix-data` volume: traces, DPO pairs, SQLite. Persists across container restarts.

**Optional profiles**:
- `pg`: PostgreSQL + pgvector. Set `DATABASE_URL` and run `docker compose --profile pg up`. Two Helix processes can share memory via ANN search.
- `vllm`: vLLM for local narrator. Requires NVIDIA GPU + weights. Set `HELIX_VLLM_URL` and run `docker compose --profile vllm up`.

**Live MCP from host**: set `HELIX_MCP_URL` in `.env` and `HELIX_ALLOW_SIMULATOR=false`. The engines container uses host DNS (`extra_hosts: host.docker.internal`). Do not publish 8090 past localhost.

---

## Security Model

```mermaid
graph TB
    subgraph "Trust Boundaries"
        subgraph "Untrusted"
            USER["User Input<br/>(voice or text)"]
            MCP["Live MCP<br/>(external tools)"]
            XAI["xAI Grok API<br/>(external LLM)"]
        end
        subgraph "Trusted"
            VER["Verifier<br/>(numbers ⊆ tool JSON)"]
            GOV["Governance<br/>(write tools require approval)"]
            SCRUB["Secret Scrubber<br/>(API keys, JWT, email, 0x…)"]
        end
    end
    
    USER -->|max 800 chars| GW["Gateway"]
    MCP -->|read tools only| GW
    XAI -->|optional narration| GW
    GW --> VER
    GW --> GOV
    GW --> SCRUB
    VER -->|leaked numbers → fallback| GW
    GOV -->|write tools → await approval| GW
    SCRUB -->|secrets removed before disk| DATA["Data Files"]
```

**Trust boundaries**:
- **User input**: max 800 chars. No SQL injection, no prompt injection (verifier checks numbers, not semantics).
- **Live MCP**: read tools only. Write tools (`propose_rebalance`, `execute`) require human approval. Dead MCP → silence, not a GBM price.
- **xAI Grok**: optional narration. Verifier runs again after Grok returns. Leaked numbers never reach TTS.

**Trusted components**:
- **Verifier**: checks spoken response against tool JSON. If numbers leak, fallback spoken is used.
- **Governance**: write tools set `writeBlocked` and refuse in speech. Secret-exfil prompts refuse to print keys.
- **Secret scrubber**: API keys, JWT, email, long `0x…` addresses are scrubbed **before** disk. Train later; collect now.

---

## Honest Boundaries

**What's wired**:
- FastAPI gateway with REST, SSE, WebSocket.
- Hybrid retrieval (BM25 + dense + RRF) with three-tier reranking.
- Model drift detection (metric, serving, input).
- A/B testing framework (paired permutation test on MRR).
- Performance dashboard (single-pane ops view).
- Training flywheel (traces → TRL JSONL → SFT/DPO).
- Docker Compose for local dev.
- Kubernetes manifests (not battle-tested).
- TorchServe config (wired, not deployed).

**What's proven**:
- CPU-only path (no GPU required for default path).
- SQLite memory (pgvector optional).
- Local tools (live MCP optional).
- Hashing embeddings (MiniLM optional).

**What's not proven**:
- Distributed training (PyTorch DDP). Requires multiple GPUs.
- Kubernetes deployment. Manifests exist, not run in production.
- TorchServe for fine-tuned reranker. Config exists, not deployed.
- Fine-tuning on RTX 2060. Metadata committed, weights gitignored.

---

**Last updated**: 2026-09-18
