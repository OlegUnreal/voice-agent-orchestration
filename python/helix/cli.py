"""CLI: helix serve | turn | eval | train."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="helix", description="Helix Python engines")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("serve", help="Run the FastAPI gateway")
    t = sub.add_parser("turn", help="Run one supervisor turn")
    t.add_argument("text", nargs="+")
    e = sub.add_parser("eval", help="Run EvalForge golden suite")
    e.add_argument("--kind", default="lora")
    sub.add_parser("train", help="Train LoRA / QLoRA adapters")
    sub.add_parser("mcp", help="List live MCP tools")
    sub.add_parser("export", help="Dump SFT/DPO dataset counts")
    mem = sub.add_parser("remember", help="Store a provenance fact")
    mem.add_argument("text", nargs="+")
    rec = sub.add_parser("recall", help="Recall facts")
    rec.add_argument("text", nargs="*")

    args = parser.parse_args(argv)
    if args.cmd == "serve":
        from helix.gateway.app import run

        run()
        return 0
    if args.cmd == "turn":
        from helix.orchestrator import grok_tool_payload, run_orchestrator

        result = run_orchestrator(" ".join(args.text))
        print(result.fallbackSpoken)
        print("---")
        print(json.dumps(grok_tool_payload(result), indent=2, default=str)[:2000])
        return 0
    if args.cmd == "eval":
        from helix.evals import gate_check, run_eval_suite

        m = run_eval_suite(args.kind)
        print(json.dumps({"metrics": m.model_dump(), "gates": gate_check(m)}, indent=2))
        return 0
    if args.cmd == "train":
        from helix.training import train_adapters

        print(json.dumps(train_adapters(), indent=2, default=str))
        return 0
    if args.cmd == "mcp":
        from helix.oktrader.live import live_status

        print(json.dumps(live_status(), indent=2, default=str))
        return 0
    if args.cmd == "export":
        from helix.ledger import export_dataset

        print(json.dumps(export_dataset(), indent=2, default=str)[:4000])
        return 0
    if args.cmd == "remember":
        from helix.memory import parse_remember, remember

        text = " ".join(args.text)
        parsed = parse_remember("remember " + text if not text.lower().startswith("remember") else text)
        if not parsed:
            parsed = (text[:40], text)
        row = remember(*parsed, source="user")
        print(json.dumps(row, indent=2))
        return 0
    if args.cmd == "recall":
        from helix.memory import list_facts, recall

        q = " ".join(args.text)
        rows = recall(q, k=8) if q else list_facts(8)
        print(json.dumps(rows, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
