/**
 * `ModelManager` — panel for downloading and managing ML models.
 *
 * Shown as a modal/overlay toggled from the app header. Displays each
 * downloadable model with its status (present / missing) and a
 * download button with real-time progress.
 */

import type { UseModelManager, DownloadProgress } from "../../hooks/useModelManager";
import type { ModelInfo } from "../../types/models";

function formatBytes(bytes: number): string {
  if (bytes < 1_000_000) return `${(bytes / 1_000).toFixed(0)} KB`;
  if (bytes < 1_000_000_000) return `${(bytes / 1_000_000).toFixed(0)} MB`;
  return `${(bytes / 1_000_000_000).toFixed(1)} GB`;
}

export function ModelManager({
  state,
  onClose,
}: Readonly<{
  state: UseModelManager;
  onClose: () => void;
}>) {
  const { models, loading, downloading, error } = state;

  const grouped = groupByKind(models);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <div className="flex max-h-[80vh] w-full max-w-lg flex-col gap-3 overflow-hidden rounded-xl border border-zinc-200 bg-white p-5 shadow-xl dark:border-zinc-800 dark:bg-zinc-950">
        <header className="flex items-center justify-between">
          <h2 className="text-base font-semibold">Models</h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md px-2 py-1 text-xs text-zinc-500 hover:bg-zinc-100 dark:hover:bg-zinc-900"
          >
            Close
          </button>
        </header>

        {error && (
          <p className="rounded-md bg-red-50 px-3 py-2 text-xs text-red-700 dark:bg-red-950/40 dark:text-red-300">
            {error}
          </p>
        )}

        {loading ? (
          <p className="py-6 text-center text-sm text-zinc-500">Loading model status…</p>
        ) : (
          <div className="flex min-h-0 flex-col gap-4 overflow-y-auto">
            {grouped.map(([kind, items]) => (
              <ModelGroup
                key={kind}
                kind={kind}
                models={items}
                downloading={downloading}
                onDownload={state.download}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function groupByKind(models: ModelInfo[]): [string, ModelInfo[]][] {
  const order = ["asr", "llm", "vad", "embedder"];
  const map = new Map<string, ModelInfo[]>();
  for (const m of models) {
    const arr = map.get(m.kind) ?? [];
    arr.push(m);
    map.set(m.kind, arr);
  }
  return order
    .filter((k) => map.has(k))
    .map((k) => {
      const items = map.get(k);
      if (!items) throw new Error(`unreachable: ${k} passed filter`);
      return [k, items] as [string, ModelInfo[]];
    });
}

const KIND_LABELS: Record<string, string> = {
  asr: "Speech Recognition (Whisper)",
  llm: "Language Model (Summary & Chat)",
  vad: "Voice Activity Detection",
  embedder: "Speaker Embedder",
};

function ModelGroup({
  kind,
  models,
  downloading,
  onDownload,
}: Readonly<{
  kind: string;
  models: ModelInfo[];
  downloading: DownloadProgress | null;
  onDownload: (id: string) => void;
}>) {
  return (
    <div className="flex flex-col gap-2">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
        {KIND_LABELS[kind] ?? kind}
      </h3>
      {models.map((m) => (
        <ModelRow
          key={m.id}
          model={m}
          downloading={downloading}
          onDownload={onDownload}
        />
      ))}
    </div>
  );
}

function ModelRow({
  model,
  downloading,
  onDownload,
}: Readonly<{
  model: ModelInfo;
  downloading: DownloadProgress | null;
  onDownload: (id: string) => void;
}>) {
  const isDownloading = downloading?.modelId === model.id;
  const anyDownloading = downloading !== null;
  const progress =
    isDownloading && downloading.total > 0
      ? (downloading.downloaded / downloading.total) * 100
      : 0;

  return (
    <div className="flex items-center gap-3 rounded-lg border border-zinc-100 bg-zinc-50/50 px-3 py-2.5 dark:border-zinc-800/60 dark:bg-zinc-900/30">
      <StatusDot present={model.present} />

      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
        <span className="truncate text-sm font-medium text-zinc-800 dark:text-zinc-200">
          {model.label}
        </span>
        {isDownloading && downloading.total > 0 && (
          <div className="flex items-center gap-2">
            <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-700">
              <div
                className="h-full rounded-full bg-blue-500 transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
            <span className="shrink-0 text-[10px] tabular-nums text-zinc-500">
              {formatBytes(downloading.downloaded)} / {formatBytes(downloading.total)}
            </span>
          </div>
        )}
        {isDownloading && downloading.total === 0 && (
          <span className="text-[10px] text-zinc-500">Connecting…</span>
        )}
      </div>

      {model.present ? (
        <span className="shrink-0 rounded-md bg-emerald-50 px-2 py-1 text-[10px] font-medium text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-300">
          Installed
        </span>
      ) : (
        <button
          type="button"
          disabled={anyDownloading}
          onClick={() => onDownload(model.id)}
          className="shrink-0 rounded-md border border-blue-200 bg-blue-50 px-2.5 py-1 text-xs font-medium text-blue-700 hover:bg-blue-100 disabled:cursor-not-allowed disabled:opacity-50 dark:border-blue-800 dark:bg-blue-950/40 dark:text-blue-300 dark:hover:bg-blue-900/60"
        >
          {isDownloading ? "Downloading…" : `Download (${formatBytes(model.sizeBytes)})`}
        </button>
      )}
    </div>
  );
}

function StatusDot({ present }: Readonly<{ present: boolean }>) {
  return (
    <span
      className={`h-2 w-2 shrink-0 rounded-full ${
        present
          ? "bg-emerald-500"
          : "bg-zinc-300 dark:bg-zinc-600"
      }`}
    />
  );
}
