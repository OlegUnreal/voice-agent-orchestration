import { useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  ResponsiveContainer,
} from "recharts";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  adapterLoss,
  canPromote,
  defaultCheckpoints,
  gateCheck,
  nextStage,
  trainAdapters,
} from "@/lib/helix/training";
import type { Checkpoint, PromotionStage } from "@/lib/helix/types";
import { getCheckpoints, trainEngine } from "@/lib/helix/engine";
import { cn } from "@/lib/utils";

const STAGES: PromotionStage[] = ["staging", "shadow", "canary", "production"];

export function TrainingView() {
  const [ckpts, setCkpts] = useState<Checkpoint[]>(() => defaultCheckpoints());
  const [loss, setLoss] = useState(adapterLoss());
  const [msg, setMsg] = useState<string | null>(null);

  const chart = useMemo(
    () => loss.loraLoss.map((v, i) => ({ i, lora: v, qlora: loss.qloraLoss[i] ?? v })),
    [loss],
  );

  useEffect(() => {
    getCheckpoints().then((r) => {
      if (r?.checkpoints) {
        setCkpts(r.checkpoints);
        if (r.loss) setLoss(r.loss);
      }
    });
  }, []);

  async function train() {
    const py = await trainEngine();
    if (py?.checkpoints) {
      setCkpts(py.checkpoints);
      setLoss({
        loraLoss: py.loraLoss ?? [],
        qloraLoss: py.qloraLoss ?? [],
        lastTrainMs: py.lastTrainMs ?? 0,
      });
      setMsg(`Python TrainingOps: ${py.pairs} pairs in ${Number(py.lastTrainMs).toFixed(0)} ms (NumPy SGD + sklearn).`);
      return;
    }
    const r = trainAdapters();
    setLoss(adapterLoss());
    setCkpts(defaultCheckpoints());
    setMsg(`Trained on ${r.pairs} pairs in ${r.lastTrainMs.toFixed(0)} ms. Registry metrics refreshed.`);
  }

  function promote(id: string) {
    setCkpts((prev) =>
      prev.map((c) => {
        if (c.id !== id) return c;
        const n = nextStage(c.stage);
        if (!n || !canPromote(c.metrics)) return c;
        return { ...c, stage: n };
      }),
    );
  }

  return (
    <div className="space-y-5">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="font-display text-3xl tracking-tight">TrainingOps</h1>
          <p className="mt-1 max-w-2xl text-sm text-muted">
            Versioned qrels from traces, SFT-style logistic LoRA/QLoRA on retrieval features, checkpoint compare, staging → shadow → canary → production.
          </p>
        </div>
        <Button size="pill" onClick={train}>
          Train adapters
        </Button>
      </header>
      {msg && <p className="text-sm text-muted">{msg}</p>}

      {chart.length > 0 && (
        <section className="rounded-3xl border border-border bg-surface p-5">
          <h2 className="font-display text-lg">Loss · LoRA r8 vs QLoRA r4</h2>
          <div className="mt-3 h-36">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chart}>
                <Area type="monotone" dataKey="lora" stroke="#d4d4d8" fill="#d4d4d818" strokeWidth={1.4} />
                <Area type="monotone" dataKey="qlora" stroke="#71717a" fill="transparent" strokeWidth={1.2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </section>
      )}

      <section className="rounded-3xl border border-border bg-surface p-5">
        <h2 className="font-display text-lg">Model registry</h2>
        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-[640px] text-left text-sm">
            <thead className="text-[11px] uppercase tracking-wider text-subtle">
              <tr>
                <th className="pb-2 font-medium">Checkpoint</th>
                <th className="pb-2 font-medium">Kind</th>
                <th className="pb-2 font-medium">nDCG</th>
                <th className="pb-2 font-medium">R@5</th>
                <th className="pb-2 font-medium">Ground</th>
                <th className="pb-2 font-medium">Stage</th>
                <th className="pb-2 font-medium" />
              </tr>
            </thead>
            <tbody>
              {ckpts.map((c) => {
                const ok = canPromote(c.metrics);
                return (
                  <tr key={c.id} className="border-t border-border">
                    <td className="py-3">
                      <p className="font-mono text-xs">{c.id}</p>
                      <p>{c.name}</p>
                    </td>
                    <td className="py-3 text-muted">
                      {c.kind}
                      {c.rank ? ` r${c.rank}` : ""}
                    </td>
                    <td className="py-3 font-mono tabular-nums">{(c.metrics.ndcgAt10 * 100).toFixed(0)}%</td>
                    <td className="py-3 font-mono tabular-nums">{(c.metrics.recallAt5 * 100).toFixed(0)}%</td>
                    <td className="py-3 font-mono tabular-nums">{(c.metrics.groundedness * 100).toFixed(0)}%</td>
                    <td className="py-3">
                      <Badge tone={c.stage === "production" ? "live" : "neutral"}>{c.stage}</Badge>
                    </td>
                    <td className="py-3">
                      {nextStage(c.stage) && (
                        <button
                          type="button"
                          disabled={!ok}
                          onClick={() => promote(c.id)}
                          className="text-xs text-muted hover:text-fg disabled:opacity-30"
                        >
                          Promote
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <ol className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        {STAGES.map((s) => (
          <li
            key={s}
            className={cn(
              "rounded-2xl border border-border px-3 py-3 text-center text-xs uppercase tracking-wider",
              ckpts.some((c) => c.stage === s) ? "bg-elevated text-fg" : "text-subtle",
            )}
          >
            {s}
          </li>
        ))}
      </ol>
      <p className="text-xs text-subtle">
        Lineage example: {ckpts.find((c) => c.stage === "production")?.lineage.join(" → ") || "—"}
      </p>
      {ckpts[0] && (
        <ul className="text-xs text-muted">
          {gateCheck(ckpts.find((c) => c.stage === "production")?.metrics ?? ckpts[0].metrics).map((g) => (
            <li key={g.name}>
              {g.ok ? "pass" : "fail"} · {g.name}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
