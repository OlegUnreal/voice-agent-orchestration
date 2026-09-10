# Helix engines (Python)

Source of truth for the résumé stack: **FastAPI + Pydantic** gateway, **pandas/NumPy** quant, hashing-trick RAG, **scikit-learn** LoRA-style rerank, EvalForge, TrainingOps, MCP-style tools.

The TypeScript app is the voice/UI shell. It calls this service for numbers.

```bash
cd python
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/helix serve          # FastAPI gateway
.venv/bin/helix turn "What is BTC's current regime?"
.venv/bin/helix eval
.venv/bin/helix train
.venv/bin/helix export
.venv/bin/helix snapshot
.venv/bin/helix serving
.venv/bin/helix graph "Place a market buy of 2 BTC"
.venv/bin/helix finetune
.venv/bin/helix redteam
.venv/bin/pytest
```

Extras: `.[nlp]` MiniLM/cross-encoder, `.[pg]` pgvector, `.[agents]` LangGraph, `.[train]` PEFT/TRL.

From the repo root, the same engines run in Docker with the voice UI:

```bash
cp compose.env.example .env
docker compose up --build
docker compose exec engines helix serving
```


