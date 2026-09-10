"""On-disk JSONL / SQLite under HELIX_DATA_DIR. Never commit live files."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


def data_dir() -> Path:
    raw = os.environ.get("HELIX_DATA_DIR")
    if raw:
        path = Path(raw)
    else:
        path = Path(__file__).resolve().parent.parent / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def jsonl_path(name: str) -> Path:
    return data_dir() / name


def append_jsonl(name: str, row: dict[str, Any]) -> None:
    with jsonl_path(name).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def read_jsonl(name: str) -> list[dict[str, Any]]:
    path = jsonl_path(name)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def iter_jsonl(name: str) -> Iterable[dict[str, Any]]:
    yield from read_jsonl(name)
