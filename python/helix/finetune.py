"""Export TRL/PEFT datasets from the flywheel. Train only with helix-engines[train] + GPU."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from helix.store import data_dir, read_jsonl


def export_trl() -> dict[str, Any]:
    prefs = read_jsonl("preferences.jsonl")
    dest = data_dir() / "trl"
    dest.mkdir(parents=True, exist_ok=True)
    sft_path = dest / "sft.jsonl"
    dpo_path = dest / "dpo.jsonl"
    sft_n = dpo_n = 0
    with sft_path.open("w", encoding="utf-8") as sft, dpo_path.open("w", encoding="utf-8") as dpo:
        for row in prefs:
            kind = row.get("kind")
            prompt = row.get("prompt") or row.get("query") or ""
            if kind == "sft" and row.get("chosen"):
                sft.write(
                    json.dumps(
                        {
                            "messages": [
                                {"role": "system", "content": "Speak only numbers that appear in tool JSON."},
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": row["chosen"]},
                            ]
                        }
                    )
                    + "\n"
                )
                sft_n += 1
            if kind in {"dpo", "distill"} and row.get("chosen") and row.get("rejected"):
                dpo.write(
                    json.dumps({"prompt": prompt, "chosen": row["chosen"], "rejected": row["rejected"]}) + "\n"
                )
                dpo_n += 1
    return {"sft": str(sft_path), "dpo": str(dpo_path), "sftPairs": sft_n, "dpoPairs": dpo_n}


def train_peft(export_only: bool = True) -> dict[str, Any]:
    dumped = export_trl()
    if export_only or dumped["sftPairs"] < 8:
        return {**dumped, "trained": False, "reason": "export-only or fewer than 8 SFT pairs"}
    try:
        import torch
        from peft import LoraConfig, TaskType  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
    except ImportError:
        return {**dumped, "trained": False, "reason": "pip install helix-engines[train]"}
    model_id = __import__("os").environ.get("HELIX_SFT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    return {
        **dumped,
        "trained": False,
        "reason": "weights stay in helix-live; run on GPU with HELIX_SFT_MODEL",
        "model": model_id,
        "cuda": bool(torch.cuda.is_available()),
        "peftReady": True,
    }
