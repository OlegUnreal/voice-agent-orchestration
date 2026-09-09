import { Badge } from "@/components/ui/badge";
import { MCP_TOOLS } from "@/lib/helix/mcp";
import { SEED_TRACES } from "@/lib/helix/traces";
import { formatUsd } from "@/lib/utils";

export function GatewayView() {
  const tokens = SEED_TRACES.reduce((s, t) => s + t.tokens, 0);
  const cost = SEED_TRACES.reduce((s, t) => s + t.costUsd, 0);
  const p95 = [...SEED_TRACES].sort((a, b) => a.latencyMs - b.latencyMs)[
    Math.floor(SEED_TRACES.length * 0.95)
  ]?.latencyMs;
  const hit = SEED_TRACES.filter((t) => t.retrievalHit).length / SEED_TRACES.length;
  const fail = SEED_TRACES.filter((t) => t.feedback === "down").length;

  return (
    <div className="space-y-5">
      <header>
        <h1 className="font-display text-3xl tracking-tight">AI gateway</h1>
        <p className="mt-1 max-w-2xl text-sm text-muted">
          Model routing, response cache, structured tool results, MCP servers, audit traces and an approval gate on rebalance.
        </p>
      </header>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Kpi label="Routes" value="grok-4.5 · cache · local" />
        <Kpi label="Retrieval hit" value={`${Math.round(hit * 100)}%`} />
        <Kpi label="p95 TTFT" value={`${p95 ?? 0} ms`} />
        <Kpi label="Token spend" value={`${tokens} / ${formatUsd(cost)}`} />
      </div>

      <section className="rounded-3xl border border-border bg-surface p-5">
        <h2 className="font-display text-lg">MCP servers</h2>
        <ul className="mt-4 space-y-3">
          {MCP_TOOLS.map((t) => (
            <li key={t.name} className="flex flex-wrap items-start justify-between gap-3 border-t border-border pt-3 first:border-0 first:pt-0">
              <div>
                <p className="font-mono text-sm">{t.name}</p>
                <p className="text-xs text-muted">{t.description}</p>
              </div>
              <div className="flex items-center gap-2">
                <Badge>{t.server}</Badge>
                <Badge tone={t.governed === "approve" ? "loss" : "gain"}>
                  {t.governed === "approve" ? "approval" : "auto"}
                </Badge>
              </div>
            </li>
          ))}
        </ul>
      </section>

      <section className="rounded-3xl border border-border bg-surface p-5">
        <div className="flex items-center justify-between">
          <h2 className="font-display text-lg">Production traces</h2>
          <p className="text-xs text-subtle">{fail} labeled failure{fail === 1 ? "" : "s"} → next dataset</p>
        </div>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full min-w-[640px] text-left text-sm">
            <thead className="text-[11px] uppercase tracking-wider text-subtle">
              <tr>
                <th className="pb-2 font-medium">Id</th>
                <th className="pb-2 font-medium">Query</th>
                <th className="pb-2 font-medium">Tools</th>
                <th className="pb-2 font-medium">ms</th>
                <th className="pb-2 font-medium">tok</th>
                <th className="pb-2 font-medium">fb</th>
              </tr>
            </thead>
            <tbody>
              {SEED_TRACES.map((t) => (
                <tr key={t.id} className="border-t border-border">
                  <td className="py-2 font-mono text-xs text-subtle">{t.id}</td>
                  <td className="py-2 pr-3">{t.query}</td>
                  <td className="py-2 font-mono text-[11px] text-muted">{t.tools[0]}</td>
                  <td className="py-2 font-mono tabular-nums">{t.latencyMs}</td>
                  <td className="py-2 font-mono tabular-nums">{t.tokens}</td>
                  <td className="py-2">{t.feedback ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function Kpi({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-border bg-surface p-4">
      <p className="text-[11px] uppercase tracking-wider text-subtle">{label}</p>
      <p className="mt-1 font-mono text-sm">{value}</p>
    </div>
  );
}
