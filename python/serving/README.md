# TorchServe Model Serving

Production-grade model serving for the fine-tuned BERT reranker using TorchServe.

**Status**: Wired but not deployed. Use as a starting point, not a copy-paste solution.

## What TorchServe provides

- **Batching**: automatically batches incoming requests for GPU efficiency.
- **Model management**: load/unload models, hot-swap versions, A/B test.
- **Metrics**: latency, throughput, errors — automatically logged.
- **gRPC + REST**: both APIs out of the box.
- **GPU allocation**: multi-GPU support with worker management.

## Quick start

```bash
# Prerequisites: train the model first
cd python
python -m helix.train_finetuned --epochs 3

# Package and serve
bash serving/serve.sh

# Query
curl http://localhost:8080/predictions/helix-reranker \
  -H "Content-Type: application/json" \
  -d '{
    "query": "BTC regime",
    "documents": [
      {"title": "Bitcoin regime analysis", "body": "BTC is in a high-volatility regime..."},
      {"title": "ETH staking flows", "body": "ETH staking has increased 15%..."}
    ]
  }'
```

**Response**:
```json
{
  "scores": [2.34, -0.87],
  "ranked_indices": [0, 1]
}
```

## Architecture

```
                    ┌─────────────────┐
                    │   TorchServe    │
                    │   :8080 (REST)  │
                    │   :8081 (mgmt)  │
                    │   :8082 (metrics)│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  helix-reranker │
                    │  (BERT model)   │
                    │  1+ workers     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  GPU / CPU      │
                    │  (inference)    │
                    └─────────────────┘
```

## Configuration

Edit `serving/config.properties`:

- `default_workers_per_model`: number of workers (1 per GPU recommended).
- `batch_size`: max batch size for inference (default: 8).
- `max_batch_delay`: wait for batch to fill in ms (default: 100).
- `model_response_timeout`: timeout in seconds (default: 300).

## Management API

```bash
# List models
curl http://localhost:8081/models

# Describe model
curl http://localhost:8081/models/helix-reranker

# Unload model
curl -X DELETE http://localhost:8081/models/helix-reranker

# Load new version
curl -X POST "http://localhost:8081/models?url=helix-reranker-v2.mar&initial_workers=1"
```

## Metrics API

```bash
# Prometheus-format metrics
curl http://localhost:8082/metrics

# Example output:
# ts_queue_latency_microseconds{model_name="helix-reranker",} 1234.0
# ts_worker_load_time_microseconds{model_name="helix-reranker",worker_name="0",} 5678.0
# ts_inference_latency_microseconds{model_name="helix-reranker",worker_name="0",} 2345.0
```

## Integration with Helix gateway

To use TorchServe as the reranker backend, update `helix/finetuned_reranker.py` to call the TorchServe API instead of loading the model locally:

```python
import requests

def rerank_via_torchserve(query: str, documents: list[dict]) -> list[float]:
    resp = requests.post(
        "http://localhost:8080/predictions/helix-reranker",
        json={"query": query, "documents": documents},
    )
    result = resp.json()
    return result["scores"]
```

This decouples the gateway from the model weights — TorchServe handles GPU allocation, batching, and model hot-swapping.

## What's not here

- **TLS**: add a reverse proxy (nginx, traefik) for HTTPS.
- **Authentication**: add API keys or OAuth via the reverse proxy.
- **Kubernetes**: use the TorchServe Helm chart or the K8s manifests in `k8s/`.
- **Multi-model serving**: TorchServe can serve multiple models; add more `.mar` files.

## What's not proven

- This handler has not been deployed to production TorchServe.
- The batching parameters are guesses — tune based on actual latency requirements.
- GPU memory requirements depend on the base model (DistilBERT: ~1GB, BERT-base: ~2GB).

Honest about what's wired vs what's proven.

## References

- [TorchServe documentation](https://pytorch.org/serve/)
- [TorchServe model archiver](https://github.com/pytorch/serve/tree/master/model-archiver)
- [Custom handler documentation](https://pytorch.org/serve/custom_handler.html)

---

**Last updated**: 2026-09-18
