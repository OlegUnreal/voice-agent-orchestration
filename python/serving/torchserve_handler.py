"""TorchServe handler for the fine-tuned BERT reranker.

This handler wraps the fine-tuned reranker model for TorchServe deployment.
TorchServe provides production-grade model serving: batching, model management,
metrics, and gRPC/REST APIs.

Usage:
    # Package the model
    torch-model-archiver --model-name helix-reranker \
        --version 1.0 --serialized-file helix/finetuned_reranker.pt \
        --handler serving/torchserve_handler.py \
        --extra-files helix/finetuned_reranker.meta.json
    
    # Start TorchServe
    torchserve --start --model-store model_store --models helix-reranker=helix-reranker.mar
    
    # Query
    curl http://localhost:8080/predictions/helix-reranker \
        -H "Content-Type: application/json" \
        -d '{"query": "BTC regime", "documents": [{"title": "...", "body": "..."}]}'

Requirements:
    pip install torchserve torch-model-archiver torch transformers

Notes:
    - TorchServe handles batching, GPU allocation, and model hot-swapping.
    - The handler preprocesses (query, doc) pairs and postprocesses scores.
    - Metrics (latency, throughput) are automatically logged by TorchServe.
    - This is wired but not deployed. Use as a starting point.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class HelixRerankerHandler(BaseHandler):
    """TorchServe handler for the fine-tuned BERT reranker.
    
    Expected input:
        {
            "query": "BTC regime",
            "documents": [
                {"title": "Bitcoin regime analysis", "body": "..."},
                {"title": "Another doc", "body": "..."}
            ]
        }
    
    Expected output:
        {
            "scores": [0.92, 0.45],
            "ranked_indices": [0, 1]
        }
    """
    
    def initialize(self, context):
        """Load model and tokenizer."""
        super().initialize(context)
        
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        
        # Load metadata
        meta_path = Path(model_dir) / "finetuned_reranker.meta.json"
        if meta_path.exists():
            self.metadata = json.loads(meta_path.read_text())
            self.base_model = self.metadata.get("base_model", "distilbert-base-uncased")
        else:
            self.base_model = "distilbert-base-uncased"
            self.metadata = {}
        
        # Load tokenizer and model
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and properties.get("gpu_id") is not None else "cpu"
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model, num_labels=1
        )
        
        # Load weights
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = Path(model_dir) / serialized_file
        if model_pt_path.exists():
            state_dict = torch.load(model_pt_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"HelixRerankerHandler initialized: base_model={self.base_model}, device={self.device}")
    
    def preprocess(self, data):
        """Preprocess input: tokenize (query, doc) pairs."""
        requests = []
        for row in data:
            body = row.get("body") or row
            if isinstance(body, str):
                body = json.loads(body)
            
            query = body["query"]
            documents = body["documents"]
            
            pairs = []
            for doc in documents:
                text = f"{query} [SEP] {doc['title']} {doc['body']}"
                enc = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                pairs.append({
                    "input_ids": enc["input_ids"].squeeze(),
                    "attention_mask": enc["attention_mask"].squeeze(),
                })
            
            requests.append(pairs)
        
        return requests
    
    def inference(self, data):
        """Run model inference."""
        results = []
        with torch.no_grad():
            for pairs in data:
                scores = []
                for pair in pairs:
                    input_ids = pair["input_ids"].unsqueeze(0).to(self.device)
                    attention_mask = pair["attention_mask"].unsqueeze(0).to(self.device)
                    
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    logit = outputs.logits.squeeze().item()
                    scores.append(logit)
                
                results.append(scores)
        
        return results
    
    def postprocess(self, data):
        """Postprocess: rank documents by score."""
        outputs = []
        for scores in data:
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            outputs.append({
                "scores": scores,
                "ranked_indices": ranked_indices,
            })
        
        return outputs
