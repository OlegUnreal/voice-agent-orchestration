"""Numbers and tickers in speech must already exist in tool JSON."""

from __future__ import annotations

import json
import re
from typing import Any

TICKERS = ("BTC", "ETH", "SOL", "NVDA", "AAPL")
NUMBER_RE = re.compile(
    r"(?<![A-Za-z_])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?%?"
)
# Windows, ranks, percents that routinely appear in templates.
SMALL = {str(i) for i in range(0, 21)} | {"30", "50", "95", "99", "100", "200", "365"}


def _forms(token: str) -> set[str]:
    raw = token.replace(",", "").replace("%", "").strip()
    out = {token.lower(), raw.lower()}
    try:
        val = float(raw)
    except ValueError:
        return out
    out.add(f"{val:.4f}".rstrip("0").rstrip("."))
    out.add(f"{val:.2f}")
    out.add(f"{val:.1f}")
    if abs(val - round(val)) < 1e-9:
        out.add(str(int(round(val))))
    if abs(val) <= 2:
        pct = val * 100
        out.add(f"{pct:.1f}")
        out.add(str(int(round(pct))))
    if abs(val) >= 8:
        out.add(f"{val / 100:.4f}".rstrip("0").rstrip("."))
    return {x for x in out if x}


def allowed_forms(payload: Any) -> set[str]:
    blob = json.dumps(payload, default=str)
    allowed: set[str] = set(SMALL)
    for m in NUMBER_RE.finditer(blob):
        allowed |= _forms(m.group(0))
    upper = blob.upper()
    for t in TICKERS:
        if t in upper:
            allowed.add(t)
    return allowed


def extract_spoken_numbers(spoken: str) -> list[str]:
    return [m.group(0) for m in NUMBER_RE.finditer(spoken)]


def verify_spoken(spoken: str, payload: Any) -> dict[str, Any]:
    allowed = allowed_forms(payload)
    leaks: list[str] = []
    for tok in extract_spoken_numbers(spoken):
        forms = _forms(tok)
        if forms & allowed:
            continue
        raw = tok.replace(",", "").replace("%", "")
        if raw in SMALL or raw.lstrip("+-") in SMALL:
            continue
        # Ignore 4-digit years.
        if re.fullmatch(r"20\d{2}", raw):
            continue
        leaks.append(tok)
    upper = spoken.upper()
    blob = json.dumps(payload, default=str).upper()
    for t in TICKERS:
        if re.search(rf"\b{t}\b", upper) and t not in blob:
            leaks.append(t)
    ok = not leaks
    return {"ok": ok, "leaks": leaks, "spoken": spoken if ok else None}


REFUSE = (
    "I will not speak numbers that were not in the tool results. "
    "Ask again, or inspect the tool JSON."
)


def enforce(spoken: str, payload: Any, fallback: str) -> dict[str, Any]:
    check = verify_spoken(spoken, payload)
    if check["ok"]:
        fb = verify_spoken(fallback, payload)
        safe_fallback = fallback if fb["ok"] else REFUSE
        return {"ok": True, "spoken": spoken, "leaks": [], "fallbackSpoken": safe_fallback}
    fb = verify_spoken(fallback, payload)
    safe = fallback if fb["ok"] else REFUSE
    return {"ok": False, "spoken": safe, "leaks": check["leaks"], "fallbackSpoken": safe}
