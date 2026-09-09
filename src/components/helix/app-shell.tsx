import { useState } from "react";
import {
  Activity,
  AudioLines,
  Database,
  FlaskConical,
  GitBranch,
  LineChart,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { VoiceView } from "./voice-view";
import { MarketView } from "./market-view";
import { RagView } from "./rag-view";
import { GatewayView } from "./gateway-view";
import { EvalView } from "./eval-view";
import { TrainingView } from "./training-view";

const NAV = [
  { id: "voice", label: "Voice", icon: AudioLines },
  { id: "markets", label: "Markets", icon: LineChart },
  { id: "rag", label: "RAG", icon: Database },
  { id: "gateway", label: "Gateway", icon: GitBranch },
  { id: "evals", label: "Evals", icon: FlaskConical },
  { id: "training", label: "Training", icon: Activity },
] as const;

type ViewId = (typeof NAV)[number]["id"];

export function AppShell() {
  const [view, setView] = useState<ViewId>("voice");

  return (
    <div className="flex min-h-dvh bg-bg text-fg">
      <aside className="sticky top-0 hidden h-dvh w-56 shrink-0 flex-col border-r border-border px-4 py-6 md:flex">
        <div className="px-2">
          <p className="font-display text-2xl tracking-tight">Helix</p>
          <p className="mt-1 text-xs text-subtle">Voice agent orchestration</p>
        </div>
        <nav className="mt-8 flex flex-1 flex-col gap-1">
          {NAV.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => setView(item.id)}
              className={cn(
                "flex h-11 items-center gap-3 rounded-[12px] px-3 text-sm",
                view === item.id ? "bg-elevated text-fg" : "text-muted hover:text-fg",
              )}
            >
              <item.icon className="size-4" />
              {item.label}
            </button>
          ))}
        </nav>
        <p className="px-2 text-[11px] leading-relaxed text-subtle">
          Python-style engines in the browser. Grok only synthesizes after tools return.
        </p>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center justify-between border-b border-border px-4 py-3 md:px-8">
          <p className="text-xs uppercase tracking-wider text-subtle">
            {NAV.find((n) => n.id === view)?.label}
          </p>
          <p className="font-mono text-[11px] text-subtle">as-of 2026-09-09</p>
        </header>
        <main className="min-h-0 flex-1 overflow-y-auto px-4 py-5 pb-24 md:px-8 md:pb-8">
          {view === "voice" && <VoiceView />}
          {view === "markets" && <MarketView />}
          {view === "rag" && <RagView />}
          {view === "gateway" && <GatewayView />}
          {view === "evals" && <EvalView />}
          {view === "training" && <TrainingView />}
        </main>
      </div>

      <nav className="fixed inset-x-0 bottom-0 z-20 border-t border-border bg-bg/95 backdrop-blur-sm md:hidden">
        <div className="grid grid-cols-6">
          {NAV.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => setView(item.id)}
              className={cn(
                "flex min-h-14 flex-col items-center justify-center gap-1 text-[10px]",
                view === item.id ? "text-fg" : "text-subtle",
              )}
            >
              <item.icon className="size-4" />
              {item.label}
            </button>
          ))}
        </div>
      </nav>
    </div>
  );
}
