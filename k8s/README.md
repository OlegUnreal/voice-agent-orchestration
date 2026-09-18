# Helix Kubernetes Deployment

This directory contains Kubernetes manifests for deploying Helix to a production cluster.

**Status**: Manifests are wired but not battle-tested. Use as a starting point, not a copy-paste solution.

## Quick Start

```bash
# Build Docker images
docker build -t helix-engines:latest -f python/Dockerfile python/
docker build -t helix-web:latest .

# Apply manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml  # Edit with real secrets first!
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/deployment-web.yaml
kubectl apply -f k8s/service-web.yaml
kubectl apply -f k8s/hpa.yaml

# Check status
kubectl -n helix get pods
kubectl -n helix get svc

# Get external IP (for LoadBalancer)
kubectl -n helix get svc helix-web -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

## Architecture

```
                    ┌─────────────────┐
                    │   LoadBalancer  │
                    │   (helix-web)   │
                    │    Port 80      │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   helix-web     │
                    │   (2 replicas)  │
                    │   React/Next.js │
                    │   Port 8080     │
                    └────────┬────────┘
                             │ HELIX_ENGINE_URL
                    ┌────────▼────────┐
                    │ helix-engines   │
                    │ (2-10 replicas) │
                    │ FastAPI Gateway │
                    │ Port 8090       │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼──────┐  ┌───▼────┐  ┌──────▼──────┐
     │ helix-data    │  │ Config │  │   Secrets   │
     │ (PVC 10Gi)    │  │  Map   │  │             │
     │ traces, etc.  │  │        │  │             │
     └───────────────┘  └────────┘  └─────────────┘
```

## Configuration

Edit `k8s/configmap.yaml` to set environment variables:

- `HELIX_EMBEDDINGS`: `hash` (default) or `st` (MiniLM, requires `[nlp]` extra)
- `HELIX_MCP_URL`: your live MCP server (optional). Without it, Helix uses the bundled demo tape.
- `HELIX_ALLOW_SIMULATOR`: `true` (default) or `false` (production: dead MCP = silence)
- `HELIX_VLLM_URL`: local vLLM narrator (optional, requires GPU)
- `DATABASE_URL`: PostgreSQL + pgvector (optional)

Edit `k8s/secret.yaml` to set API keys:

- `HELIX_XAI_API_KEY`: xAI Grok API key for narration (optional)
- `XAI_API_KEY`: xAI API key for voice UI (optional)

**Do not commit real secrets to Git.** Use external secret management (Vault, AWS Secrets Manager, etc.) for production.

## Scaling

The HorizontalPodAutoscaler scales `helix-engines` from 2 to 10 replicas based on CPU (70%) and memory (80%) utilization. Adjust thresholds in `k8s/hpa.yaml`.

The `helix-web` deployment is fixed at 2 replicas. Increase manually if needed.

## Persistent Storage

The `helix-data` PVC (10Gi) stores traces, teachers, preferences, serving logs, experiments, and SQLite memory. Adjust storage class and size in `k8s/pvc.yaml` for your cluster.

For production, consider:
- **ReadWriteMany** access mode (NFS, EFS, etc.) for multi-replica writes
- **Database**: PostgreSQL + pgvector instead of SQLite for multi-replica memory
- **Object storage**: S3/GCS for traces and experiments instead of PVC

## Monitoring

The performance dashboard (`GET /v1/dashboard`) aggregates serving, eval, drift, and experiments. Expose it via an Ingress or port-forward:

```bash
kubectl -n helix port-forward svc/helix-engines 8090:8090
curl http://localhost:8090/v1/dashboard
```

For production monitoring, scrape `/health` and `/v1/serving` with Prometheus.

## Troubleshooting

```bash
# Check pod logs
kubectl -n helix logs deployment/helix-engines
kubectl -n helix logs deployment/helix-web

# Check pod status
kubectl -n helix describe pod <pod-name>

# Exec into pod
kubectl -n helix exec -it <pod-name> -- /bin/bash

# Check PVC binding
kubectl -n helix describe pvc helix-data-pvc
```

## What's not here

- **Ingress**: depends on your ingress controller (nginx, traefik, ALB, etc.)
- **TLS**: add cert-manager or your cloud provider's TLS termination
- **Network policies**: add if your cluster requires them
- **Resource quotas**: add if your cluster enforces them
- **Pod disruption budgets**: add for production rolling updates

## What's not proven

- These manifests have not been run in a production Kubernetes cluster.
- The Docker images need to be pushed to a registry and the `image` fields updated.
- The PVC storage class needs to match your cluster's provisioner.
- The HPA thresholds are guesses — tune based on actual load.

Honest about what's wired vs what's proven.

---

**Last updated**: 2026-09-18
