import { useEffect, useRef, useState, type ReactNode } from "react";
import {
  Loader2,
  Mic,
  MicOff,
  ThumbsDown,
  ThumbsUp,
  Volume2,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { runHelixTurn, speakHelix } from "@/lib/helix/chat";
import { submitFeedback } from "@/lib/helix/engine";
import { cn } from "@/lib/utils";
import type { AgentHop, Citation, ToolCall } from "@/lib/helix/types";

const PROMPTS = [
  "What is BTC's current regime?",
  "Any BTC exchange inflow anomalies?",
  "Backtest SMA crossover on ETH",
  "Portfolio risk if BTC drops 12 percent",
  "Cite evidence for the SOL momentum call",
  "Remember that isolated 1x and 1 USDT risk cap",
];

type Turn = {
  role: "user" | "helix";
  text: string;
  intent?: string;
  model?: string;
  tools?: ToolCall[];
  hops?: AgentHop[];
  citations?: Citation[];
  grounded?: boolean;
  verified?: boolean;
  traceId?: string;
  query?: string;
  feedback?: "up" | "down";
};

export function VoiceView() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [listening, setListening] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const recRef = useRef<{ start: () => void; stop: () => void } | null>(null);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [turns, busy]);

  function startListen() {
    const w = window as unknown as {
      SpeechRecognition?: new () => Recog;
      webkitSpeechRecognition?: new () => Recog;
    };
    const Ctor = w.SpeechRecognition || w.webkitSpeechRecognition;
    if (!Ctor) {
      setError("This browser has no speech recognizer. Type instead.");
      return;
    }
    const rec = new Ctor();
    rec.lang = "en-US";
    rec.interimResults = true;
    rec.continuous = false;
    rec.onresult = (ev) => {
      const t = Array.from(ev.results)
        .map((r) => r[0]?.transcript ?? "")
        .join(" ");
      setDraft(t);
      if (ev.results[ev.results.length - 1]?.isFinal) {
        setListening(false);
        void send(t);
      }
    };
    rec.onerror = () => {
      setListening(false);
      setError("Mic closed. You can still type.");
    };
    rec.onend = () => setListening(false);
    recRef.current = rec;
    setError(null);
    setListening(true);
    rec.start();
  }

  function stopListen() {
    recRef.current?.stop();
    setListening(false);
  }

  async function send(text: string) {
    const q = text.trim();
    if (!q || busy) return;
    setBusy(true);
    setDraft("");
    setError(null);
    setTurns((t) => [...t, { role: "user", text: q }]);
    try {
      const res = await runHelixTurn({ data: { text: q } });
      setTurns((t) => [
        ...t,
        {
          role: "helix",
          text: res.spoken,
          intent: res.intent,
          model: res.model,
          tools: res.tools,
          hops: res.hops,
          citations: res.citations,
          grounded: res.grounded,
          verified: res.verified,
          traceId: res.traceId,
          query: q,
        },
      ]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Turn failed");
    } finally {
      setBusy(false);
    }
  }

  async function vote(index: number, verdict: "up" | "down") {
    const turn = turns[index];
    if (!turn?.traceId || turn.role !== "helix") return;
    setTurns((t) => t.map((row, i) => (i === index ? { ...row, feedback: verdict } : row)));
    await submitFeedback({
      data: {
        traceId: turn.traceId,
        verdict,
        spoken: turn.text,
        query: turn.query ?? "",
      },
    });
  }

  async function speak(text: string) {
    try {
      const res = await speakHelix({ data: { text } });
      if (!res.ok) {
        setError(res.error);
        return;
      }
      const audio = new Audio(`data:${res.mime};base64,${res.audio}`);
      await audio.play();
    } catch {
      setError("Could not play voice.");
    }
  }

  const last = [...turns].reverse().find((t) => t.role === "helix");

  return (
    <div className="flex flex-col gap-6 lg:flex-row lg:items-start">
      <section className="flex min-w-0 flex-1 flex-col">
        <div className={cn("mb-5 flex flex-col items-center", turns.length ? "pt-0" : "pt-2")}>
          <button
            type="button"
            aria-label={listening ? "Stop listening" : "Start listening"}
            onClick={() => (listening ? stopListen() : startListen())}
            className={cn(
              "helix-orb relative grid place-items-center rounded-full transition-transform duration-200 active:scale-[0.98]",
              turns.length ? "size-16" : "size-36 sm:size-44",
            )}
            data-live={listening || busy ? "true" : "false"}
          >
            {busy ? (
              <Loader2 className={cn("animate-spin text-fg", turns.length ? "size-5" : "size-8")} />
            ) : listening ? (
              <MicOff className={cn("text-fg", turns.length ? "size-5" : "size-8")} />
            ) : (
              <Mic className={cn("text-fg", turns.length ? "size-5" : "size-8")} />
            )}
          </button>
          {turns.length === 0 && (
            <>
              <p className="mt-4 font-display text-2xl tracking-tight">Helix</p>
              <p className="mt-1 max-w-sm text-center text-sm text-muted">
                Voice layer over market tools, time-aware RAG, evals and the model registry.
              </p>
            </>
          )}
        </div>

        <div ref={listRef} className="space-y-4">
          {turns.length === 0 && (
            <div className="flex flex-wrap justify-center gap-2 px-1">
              {PROMPTS.map((p) => (
                <button
                  key={p}
                  type="button"
                  onClick={() => void send(p)}
                  className="rounded-full border border-border bg-surface px-3 py-2 text-left text-xs text-muted transition-colors hover:text-fg"
                >
                  {p}
                </button>
              ))}
            </div>
          )}
          {turns.map((t, i) => (
            <article
              key={i}
              className={cn(
                "rounded-2xl px-4 py-3 text-sm leading-relaxed",
                t.role === "user"
                  ? "ml-8 bg-elevated text-fg"
                  : "mr-4 border border-border bg-surface",
              )}
            >
              <div className="mb-1 flex items-center gap-2 text-[11px] uppercase tracking-wider text-subtle">
                {t.role === "user" ? "You" : "Helix"}
                {t.intent && <Badge>{t.intent}</Badge>}
                {t.model && <span className="font-mono normal-case">{t.model}</span>}
                {t.grounded && <Badge tone="gain">grounded</Badge>}
                {t.role === "helix" && t.verified === false && <Badge tone="loss">held</Badge>}
              </div>
              <p>{t.text.replace(/\*\*/g, "")}</p>
              {t.role === "helix" && (
                <div className="mt-2 flex items-center gap-3">
                  <button
                    type="button"
                    className="inline-flex items-center gap-1 text-xs text-muted hover:text-fg"
                    onClick={() => void speak(t.text)}
                  >
                    <Volume2 className="size-3.5" />
                    Speak
                  </button>
                  {t.traceId && (
                    <>
                      <button
                        type="button"
                        aria-label="Good answer"
                        className={cn("text-muted hover:text-gain", t.feedback === "up" && "text-gain")}
                        onClick={() => void vote(i, "up")}
                      >
                        <ThumbsUp className="size-3.5" />
                      </button>
                      <button
                        type="button"
                        aria-label="Bad answer"
                        className={cn("text-muted hover:text-loss", t.feedback === "down" && "text-loss")}
                        onClick={() => void vote(i, "down")}
                      >
                        <ThumbsDown className="size-3.5" />
                      </button>
                    </>
                  )}
                </div>
              )}
            </article>
          ))}
        </div>

        <form
          className="mt-4 flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            void send(draft);
          }}
        >
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="Ask Helix, or hold the orb"
            className="h-12 min-w-0 flex-1 rounded-2xl border border-border bg-surface px-4 text-sm text-fg placeholder:text-subtle focus:outline-none focus:ring-2 focus:ring-accent/30"
          />
          <Button type="submit" disabled={busy || !draft.trim()} size="pill">
            Send
          </Button>
        </form>
        {error && <p className="mt-2 text-xs text-loss">{error}</p>}
      </section>

      <aside className="flex w-full shrink-0 flex-col gap-4 lg:sticky lg:top-4 lg:w-80">
        <Panel title="Agent graph">
          <Graph hops={last?.hops ?? []} busy={busy} />
        </Panel>
        <Panel title="Governed tools">
          {last?.tools?.length ? (
            <ul className="space-y-2">
              {last.tools.map((t, i) => (
                <li key={i} className="flex items-start justify-between gap-3 text-xs">
                  <div>
                    <p className="font-mono text-fg">{t.name}</p>
                    <p className="text-muted">{t.summary}</p>
                  </div>
                  <span className="font-mono text-subtle tabular-nums">{t.ms.toFixed(1)}ms</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-xs text-muted">Tools fire after a turn. Auto-run except rebalance.</p>
          )}
        </Panel>
        <Panel title="Citations">
          {last?.citations?.length ? (
            <ul className="space-y-2">
              {last.citations.map((c) => (
                <li key={c.id} className="text-xs">
                  <p className="font-mono text-subtle">{c.id}</p>
                  <p className="text-fg">{c.title}</p>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-xs text-muted">Evidence ids from time-aware RAG.</p>
          )}
        </Panel>
      </aside>
    </div>
  );
}

function Panel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="rounded-2xl border border-border bg-surface p-4">
      <h2 className="mb-3 font-sans text-[11px] font-medium uppercase tracking-wider text-subtle">
        {title}
      </h2>
      {children}
    </section>
  );
}

function Graph({ hops, busy }: { hops: AgentHop[]; busy: boolean }) {
  const nodes = ["supervisor", "quant", "research", "copilot", "eval", "training"] as const;
  const active = new Set(hops.map((h) => h.agent));
  return (
    <div className="grid grid-cols-2 gap-2">
      {nodes.map((n) => (
        <div
          key={n}
          className={cn(
            "rounded-[10px] border px-2 py-2 font-mono text-[11px]",
            active.has(n)
              ? "border-fg/30 bg-elevated text-fg"
              : "border-border text-subtle",
            busy && n === "supervisor" && "animate-pulse",
          )}
        >
          {n}
        </div>
      ))}
    </div>
  );
}

type Recog = {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
  start: () => void;
  stop: () => void;
  onresult: ((ev: { results: ArrayLike<{ 0?: { transcript?: string }; isFinal?: boolean }> }) => void) | null;
  onerror: (() => void) | null;
  onend: (() => void) | null;
};

