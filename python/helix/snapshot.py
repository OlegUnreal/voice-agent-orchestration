"""Version a training dump. Golden set was hashed; live traces were not — that gap trains on mush."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

from helix.ledger import export_dataset
from helix.store import data_dir, jsonl_path


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    if not path.exists():
        return "0" * 16
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def snapshot(tag: str | None = None) -> dict[str, Any]:
    counts = export_dataset()
    files = ["traces.jsonl", "preferences.jsonl", "teachers.jsonl", "verify_fails.jsonl", "serving.jsonl"]
    digest = hashlib.sha256()
    manifest = {}
    for name in files:
        path = jsonl_path(name)
        sha = _sha_file(path)
        size = path.stat().st_size if path.exists() else 0
        digest.update(f"{name}:{sha}:{size}".encode())
        manifest[name] = {"sha": sha, "bytes": size}
    ds_id = "ds-live-" + digest.hexdigest()[:10]
    dest = data_dir() / "snapshots" / ds_id
    dest.mkdir(parents=True, exist_ok=True)
    for name in files:
        src = jsonl_path(name)
        if src.exists():
            shutil.copy2(src, dest / name)
    mem = data_dir() / "memory.sqlite"
    if mem.exists():
        shutil.copy2(mem, dest / "memory.sqlite")
    meta = {
        "id": ds_id,
        "tag": tag or "",
        "ts": int(time.time() * 1000),
        "files": manifest,
        "counts": {k: counts[k] for k in ("traces", "sftPairs", "dpoPairs", "distillPairs", "teachers") if k in counts},
    }
    (dest / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def list_snapshots() -> list[dict[str, Any]]:
    root = data_dir() / "snapshots"
    if not root.exists():
        return []
    out = []
    for p in sorted(root.glob("*/manifest.json"), reverse=True):
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out[:30]
