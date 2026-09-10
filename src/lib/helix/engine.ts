import { createServerFn } from "@tanstack/react-start";
import { runOrchestrator } from "./orchestrator";
import type { OrchestratorResult } from "./types";

const ENGINE = process.env.HELIX_ENGINE_URL ?? "http://127.0.0.1:8090";

async function engineFetch(path: string, init?: RequestInit) {
  const res = await fetch(`${ENGINE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    signal: AbortSignal.timeout(12_000),
  });
  if (!res.ok) throw new Error(`helix-engine ${res.status}`);
  return res.json();
}

export async function pythonTurn(text: string): Promise<OrchestratorResult & { grokPayload?: unknown }> {
  try {
    return (await engineFetch("/v1/turn", {
      method: "POST",
      body: JSON.stringify({ text }),
    })) as OrchestratorResult & { grokPayload?: unknown };
  } catch {
    return runOrchestrator(text);
  }
}

export const getMarketBook = createServerFn({ method: "GET" })
  .validator((input: { ticker: string }) => input)
  .handler(async ({ data }) => {
    try {
      return await engineFetch(`/v1/market/book?ticker=${encodeURIComponent(data.ticker)}`);
    } catch {
      return null;
    }
  });

export const retrieveEngine = createServerFn({ method: "POST" })
  .validator((input: { query: string; chronological: boolean }) => input)
  .handler(async ({ data }) => {
    try {
      return await engineFetch("/v1/rag/retrieve", {
        method: "POST",
        body: JSON.stringify({ query: data.query, k: 6, chronological: data.chronological, kind: "lora" }),
      });
    } catch {
      return null;
    }
  });

export const getEvalReport = createServerFn({ method: "GET" })
  .validator((input: { kind: string }) => input)
  .handler(async ({ data }) => {
    try {
      return await engineFetch(`/v1/evals?kind=${encodeURIComponent(data.kind)}`);
    } catch {
      return null;
    }
  });

export const trainEngine = createServerFn({ method: "POST" }).handler(async () => {
  try {
    return await engineFetch("/v1/training/train", {
      method: "POST",
      body: JSON.stringify({ epochs_lora: 56, epochs_qlora: 40 }),
    });
  } catch {
    return null;
  }
});

export const getCheckpoints = createServerFn({ method: "GET" }).handler(async () => {
  try {
    return await engineFetch("/v1/training/checkpoints");
  } catch {
    return null;
  }
});

export const getGateway = createServerFn({ method: "GET" }).handler(async () => {
  try {
    const [tools, traces] = await Promise.all([
      engineFetch("/v1/mcp/tools"),
      engineFetch("/v1/gateway/traces"),
    ]);
    return { tools, traces, python: true };
  } catch {
    return { python: false };
  }
});

export async function pythonVerify(input: {
  spoken: string;
  fallbackSpoken: string;
  payload: unknown;
  traceId?: string;
  query: string;
  model?: string;
}): Promise<{ ok: boolean; spoken: string; leaks: string[] }> {
  try {
    return (await engineFetch("/v1/verify", {
      method: "POST",
      body: JSON.stringify(input),
    })) as { ok: boolean; spoken: string; leaks: string[] };
  } catch {
    return { ok: true, spoken: input.spoken, leaks: [] };
  }
}

export const submitFeedback = createServerFn({ method: "POST" })
  .validator(
    (input: { traceId: string; verdict: "up" | "down"; spoken: string; query: string; correction?: string }) => input,
  )
  .handler(async ({ data }) => {
    try {
      return await engineFetch("/v1/feedback", {
        method: "POST",
        body: JSON.stringify(data),
      });
    } catch {
      return { ok: false };
    }
  });
