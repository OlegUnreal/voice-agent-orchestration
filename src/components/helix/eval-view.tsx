import { useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { GOLDEN, runEvalSuite, type RetrieverKind } from "@/lib/helix/evals";
import { gateCheck, getAdapterWeights } from "@/lib/helix/training";
import { cn, formatNum } from "@/lib/utils";

export function EvalView() {
  const [kind, setKind] = useState<RetrieverKind>("lora");
  const [ran, setRan] = useState(false);
  const metrics = useMemo(
    () => runEvalSuite(kind, getAdapterWeights(kind)),
    [kind],
  );
  const gates = gateCheck(metrics);

  return (
    <div className="space-y-5">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="font-display text-3xl tracking-tight">EvalForge</h1>
          <p className="mt-1 max-w-2xl text-sm text-muted">
            Golden set of {GOLDEN.length} queries. Retrieval, intent, tool-selection, groundedness, latency and cost — the same gates that promote a checkpoint.
          </p>
        </div>
        <Button size="pill" variant="outline" onClick={() => setRan(true)}>
          {ran ? "Suite current" : "Run suite"}
        </Button>
      </header>

      <div className="flex flex-wrap gap-2">
        {(["lexical", "embed", "lora"] as RetrieverKind[]).map((k) => (
          <button
            key={k}
            type="button"
            onClick={() => setKind(k)}
            className={cn(
              "rounded-full border px-3 py-2 text-xs",
              kind === k ? "border-fg/40 bg-elevated text-fg" : "border-border text-muted",
            )}
          >
            {k}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Metric label="Recall@5" value={metrics.recallAt5} />
        <Metric label="nDCG@10" value={metrics.ndcgAt10} />
        <Metric label="MRR" value={metrics.mrr} />
        <Metric label="Groundedness" value={metrics.groundedness} />
        <Metric label="Tool select" value={metrics.toolAcc} />
        <Metric label="Intent acc" value={metrics.intentAcc} />
        <Metric label="Citation" value={metrics.citationCorrect} />
        <Metric label="p95 ms" value={metrics.p95Ms} raw />
      </div>

      <section className="rounded-3xl border border-border bg-surface p-5">
        <h2 className="font-display text-lg">Promotion gates</h2>
        <ul className="mt-3 space-y-2">
          {gates.map((g) => (
            <li key={g.name} className="flex items-center justify-between text-sm">
              <span>{g.name}</span>
              <Badge tone={g.ok ? "gain" : "loss"}>{g.ok ? "pass" : "fail"}</Badge>
            </li>
          ))}
        </ul>
      </section>

      <section className="rounded-3xl border border-border bg-surface p-5">
        <h2 className="font-display text-lg">Golden / regression set</h2>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full min-w-[520px] text-left text-sm">
            <thead className="text-[11px] uppercase tracking-wider text-subtle">
              <tr>
                <th className="pb-2 font-medium">Id</th>
                <th className="pb-2 font-medium">Query</th>
                <th className="pb-2 font-medium">Intent</th>
                <th className="pb-2 font-medium">Relevant</th>
              </tr>
            </thead>
            <tbody>
              {GOLDEN.map((c) => (
                <tr key={c.id} className="border-t border-border">
                  <td className="py-2 font-mono text-xs text-subtle">{c.id}</td>
                  <td className="py-2 pr-3">{c.query}</td>
                  <td className="py-2 text-muted">{c.expectedIntent}</td>
                  <td className="py-2 font-mono text-xs text-subtle">{c.relevant.join(" ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function Metric({ label, value, raw }: { label: string; value: number; raw?: boolean }) {
  return (
    <div className="rounded-2xl border border-border bg-surface p-4">
      <p className="text-[11px] uppercase tracking-wider text-subtle">{label}</p>
      <p className="mt-1 font-mono text-lg tabular-nums">
        {raw ? formatNum(value, 1) : (value * 100).toFixed(0) + "%"}
      </p>
    </div>
  );
}
