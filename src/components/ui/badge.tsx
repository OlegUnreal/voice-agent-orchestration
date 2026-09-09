import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export function Badge({
  className,
  tone = "neutral",
  children,
}: {
  className?: string;
  tone?: "neutral" | "gain" | "loss" | "live";
  children: ReactNode;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-[11px] font-medium tracking-wide",
        tone === "neutral" && "bg-elevated text-muted",
        tone === "gain" && "bg-gain/15 text-gain",
        tone === "loss" && "bg-loss/15 text-loss",
        tone === "live" && "bg-fg text-bg",
        className,
      )}
    >
      {children}
    </span>
  );
}
