import { useEffect, useState } from "react";
import { healthCheck, isTauri, type HealthStatus } from "./lib/ipc";

type Probe =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "ok"; status: HealthStatus }
  | { kind: "error"; message: string };

export function App() {
  const [probe, setProbe] = useState<Probe>({ kind: "idle" });

  useEffect(() => {
    if (!isTauri()) {
      setProbe({
        kind: "error",
        message: "Running outside Tauri — IPC is unavailable in `pnpm dev`. Use `pnpm tauri:dev`.",
      });
      return;
    }

    setProbe({ kind: "loading" });
    healthCheck()
      .then((status) => setProbe({ kind: "ok", status }))
      .catch((err: unknown) =>
        setProbe({
          kind: "error",
          message: err instanceof Error ? err.message : String(err),
        }),
      );
  }, []);

  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-8 px-6 py-16">
      <header className="flex flex-col items-center gap-2 text-center">
        <h1 className="text-3xl font-semibold tracking-tight">EchoNote</h1>
        <p className="max-w-md text-sm text-zinc-500 dark:text-zinc-400">
          Private, local-first meeting transcription and AI summaries.
        </p>
      </header>

      <section
        aria-live="polite"
        className="w-full max-w-md rounded-lg border border-zinc-200 bg-zinc-50 p-5 font-mono text-xs leading-relaxed dark:border-zinc-800 dark:bg-zinc-900"
      >
        <HealthProbe probe={probe} />
      </section>

      <footer className="text-xs text-zinc-400 dark:text-zinc-600">
        Sprint 0 · day 4 · Tauri shell online
      </footer>
    </main>
  );
}

function HealthProbe({ probe }: { probe: Probe }) {
  switch (probe.kind) {
    case "idle":
      return <p className="text-zinc-500">Warming up…</p>;
    case "loading":
      return <p className="text-zinc-500">Calling backend health_check…</p>;
    case "error":
      return (
        <p className="text-amber-700 dark:text-amber-400">
          <span className="font-semibold">offline:</span> {probe.message}
        </p>
      );
    case "ok":
      return (
        <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1">
          <dt className="text-zinc-500">backend</dt>
          <dd className="text-emerald-700 dark:text-emerald-400">ok</dd>
          <dt className="text-zinc-500">version</dt>
          <dd>{probe.status.version}</dd>
          <dt className="text-zinc-500">target</dt>
          <dd>{probe.status.target}</dd>
          <dt className="text-zinc-500">commit</dt>
          <dd>{probe.status.commit}</dd>
          <dt className="text-zinc-500">checked</dt>
          <dd>{probe.status.timestamp}</dd>
        </dl>
      );
  }
}
