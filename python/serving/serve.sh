#!/bin/bash
# Package and serve the Helix reranker with TorchServe
#
# Prerequisites:
#   pip install torchserve torch-model-archiver torch transformers
#
# This script:
#   1. Creates the model archive (.mar) from the fine-tuned weights
#   2. Starts TorchServe with the config
#   3. Queries the model
#
# Notes:
#   - The fine-tuned weights (finetuned_reranker.pt) must exist.
#     Run `python -m helix.train_finetuned` first if they don't.
#   - TorchServe handles batching, GPU allocation, and model hot-swapping.
#   - This is wired but not deployed. Use as a starting point.

set -e

cd "$(dirname "$0")/.."

echo "=== Packaging Helix reranker for TorchServe ==="

# Check if weights exist
if [ ! -f helix/finetuned_reranker.pt ]; then
    echo "Error: helix/finetuned_reranker.pt not found."
    echo "Run 'python -m helix.train_finetuned' first to train the model."
    exit 1
fi

# Create model store
mkdir -p model_store

# Package the model
echo "Creating model archive..."
torch-model-archiver \
    --model-name helix-reranker \
    --version 1.0 \
    --serialized-file helix/finetuned_reranker.pt \
    --handler serving/torchserve_handler.py \
    --extra-files helix/finetuned_reranker.meta.json \
    --export-path model_store \
    --force

echo "Model archive created: model_store/helix-reranker.mar"

# Start TorchServe
echo ""
echo "=== Starting TorchServe ==="
echo "Inference API: http://localhost:8080"
echo "Management API: http://localhost:8081"
echo "Metrics API: http://localhost:8082"
echo ""

torchserve --start \
    --model-store model_store \
    --models helix-reranker=helix-reranker.mar \
    --ts-config serving/config.properties

echo ""
echo "TorchServe started. Query with:"
echo ""
echo '  curl http://localhost:8080/predictions/helix-reranker \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"query": "BTC regime", "documents": [{"title": "Bitcoin analysis", "body": "..."}]}'"'"''
echo ""
echo "Stop with: torchserve --stop"
