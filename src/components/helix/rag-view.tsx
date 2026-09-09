import { useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { lexicalRetrieve, retrieve } from "@/lib/helix/rag";
import { CORPUS } from "@/lib/helix/corpus";
import { cn } from "@/lib/utils";

export function RagView() {
  const [q, setQ] = useState("BTC exchange inflow anomalies");
  const [chrono, setChrono] = useState(true);
  const [query, setQuery] = useState(q);
  const hybrid = useMemo(
    () => retrieve(query, { k: 6, chronological: chrono }),
    [query, chrono],
  );
  const lex = useMemo(() => lexicalRetrieve(query, 6), [query]);

  return (
    <div className="space-y-5">
      <header>
        <h1 className="font-display text-3xl tracking-tight">MLLLM / RAG lab</h1>
        <p className="mt-1 max-w-2xl text-sm text-muted">
          Hash embeddings, recency decay, chronological mask, then a logistic cross-encoder. Same loop as LoRA promotion for a reranker.
        </p>
      </header>
      <form
        className="flex flex-col gap-3 sm:flex-row"
        onSubmit={(e) => {
          e.preventDefault();
          setQuery(q);
        }}
      >
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          className="h-12 flex-1 rounded-2xl border border-border bg-surface px-4 text-sm focus:outline-none focus:ring-2 focus:ring-accent/30"
        />
        <Button type="submit" size="pill">
          Retrieve
        </Button>
      </form>
      <label className="flex items-center gap-2 text-sm text-muted">
        <input
          type="checkbox"
          checked={chrono}
          onChange={(e) => setChrono(e.target.checked)}
          className="size-4 accent-fg"
        />
        Chronological eval — drop docs after as-of date
      </label>
      <div className="grid gap-4 lg:grid-cols-2">
        <RankList title="Lexical" rows={lex} />
        <RankList title="Hybrid + rerank" rows={hybrid} accent />
      </div>
      <section className="rounded-3xl border border-border bg-surface p-5">
        <h2 className="font-display text-lg">Corpus</h2>
        <p className="mt-1 text-sm text-muted">{CORPUS.length} dated notes — research, on-chain, strategy, risk, events.</p>
        <ul className="mt-4 grid gap-2 sm:grid-cols-2">
          {CORPUS.map((d) => (
            <li key={d.id} className="rounded-[12px] border border-border px-3 py-2">
              <p className="font-mono text-[11px] text-subtle">{d.id}</p>
              <p className="text-sm">{d.title}</p>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}

function RankList({
  title,
  rows,
  accent,
}: {
  title: string;
  rows: ReturnType<typeof retrieve>;
  accent?: boolean;
}) {
  return (
    <section className="rounded-3xl border border-border bg-surface p-5">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="font-display text-lg">{title}</h2>
        {accent && <Badge tone="live">production</Badge>}
      </div>
      <ol className="space-y-3">
        {rows.map((r, i) => (
          <li key={r.doc.id} className="text-sm">
            <div className="flex items-baseline justify-between gap-3">
              <span className={cn("font-mono text-xs text-subtle")}>
                {i + 1}. {r.doc.id}
              </span>
              <span className="font-mono text-xs tabular-nums text-muted">
                {r.score.toFixed(3)}
              </span>
            </div>
            <p className="mt-0.5">{r.doc.title}</p>
            <p className="text-xs text-subtle">
              cos {r.cosine.toFixed(2)} · recency {r.recency.toFixed(2)} · {r.doc.type}
            </p>
          </li>
        ))}
      </ol>
    </section>
  );
}
