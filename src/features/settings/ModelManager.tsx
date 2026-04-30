/**
 * `ModelManager` — panel for downloading and managing ML models.
 *
 * Shown as a modal/overlay toggled from the app header. Displays each
 * downloadable model with its status (present / missing) and a
 * download button with real-time progress.
 */

import { useTranslation } from "react-i18next";

import type { UseModelManager, DownloadProgress } from "../../hooks/useModelManager";
import type { ModelInfo } from "../../types/models";
import type { ModelRecommendation } from "../../types/hardware";
import { useHardwareRecommendation } from "../../hooks/useHardwareRecommendation";

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
  const { models, loading, downloading, error, activeLlm, activeAsr } = state;
  const { t } = useTranslation();
  const { data: recommendation } = useHardwareRecommendation();

  const grouped = groupByKind(models);

  const activeIds: Record<string, string | null> = {
    llm: activeLlm,
    asr: activeAsr,
  };
  const selectHandlers: Record<string, (id: string) => void> = {
    llm: state.selectLlm,
    asr: state.selectAsr,
  };

  const recommendedIds = new Set<string>();
  if (recommendation) {
    recommendedIds.add(recommendation.asr.modelId);
    if (recommendation.llm) recommendedIds.add(recommendation.llm.modelId);
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <div className="flex max-h-[80vh] w-full max-w-lg flex-col gap-3 overflow-hidden rounded-xl border border-zinc-200 bg-white p-5 shadow-xl dark:border-zinc-800 dark:bg-zinc-950">
        <header className="flex items-center justify-between">
          <h2 className="text-base font-semibold">{t("models.title")}</h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md px-2 py-1 text-xs text-zinc-500 hover:bg-zinc-100 dark:hover:bg-zinc-900"
          >
            {t("models.close")}
          </button>
        </header>

        {recommendation && <HardwareChip recommendation={recommendation} />}

        {error && (
          <p className="rounded-md bg-red-50 px-3 py-2 text-xs text-red-700 dark:bg-red-950/40 dark:text-red-300">
            {error}
          </p>
        )}

        {loading ? (
          <p className="py-6 text-center text-sm text-zinc-500">{t("models.loading")}</p>
        ) : (
          <div className="flex min-h-0 flex-col gap-4 overflow-y-auto">
            {grouped.map(([kind, items]) => (
              <ModelGroup
                key={kind}
                kind={kind}
                models={items}
                downloading={downloading}
                activeId={activeIds[kind] ?? null}
                recommendedIds={recommendedIds}
                onDownload={state.download}
                onCancel={state.cancelDl}
                onDelete={state.remove}
                {...(selectHandlers[kind] ? { onSelect: selectHandlers[kind] } : {})}
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
  asr: "models.asr",
  llm: "models.llm",
  vad: "models.vad",
  embedder: "models.embedder",
};

function ModelGroup({
  kind,
  models,
  downloading,
  activeId,
  recommendedIds,
  onDownload,
  onCancel,
  onDelete,
  onSelect,
}: Readonly<{
  kind: string;
  models: ModelInfo[];
  downloading: DownloadProgress | null;
  activeId: string | null;
  recommendedIds: Set<string>;
  onDownload: (id: string) => void;
  onCancel: (id: string) => void;
  onDelete: (id: string) => void;
  onSelect?: (id: string) => void;
}>) {
  const { t } = useTranslation();
  const selectable = kind === "llm" || kind === "asr";
  return (
    <div className="flex flex-col gap-2">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
        {t(KIND_LABELS[kind] ?? kind)}
      </h3>
      {models.map((m) => (
        <ModelRow
          key={m.id}
          model={m}
          downloading={downloading}
          isActive={selectable && m.id === activeId}
          isRecommended={recommendedIds.has(m.id)}
          isRequired={kind === "vad" || kind === "embedder"}
          showUse={selectable}
          onDownload={onDownload}
          onCancel={onCancel}
          onDelete={onDelete}
          {...(onSelect ? { onSelect } : {})}
        />
      ))}
    </div>
  );
}

function ModelRow({
  model,
  downloading,
  isActive,
  isRecommended,
  isRequired,
  showUse,
  onDownload,
  onCancel,
  onDelete,
  onSelect,
}: Readonly<{
  model: ModelInfo;
  downloading: DownloadProgress | null;
  isActive: boolean;
  isRecommended: boolean;
  isRequired: boolean;
  showUse: boolean;
  onDownload: (id: string) => void;
  onCancel: (id: string) => void;
  onDelete: (id: string) => void;
  onSelect?: (id: string) => void;
}>) {
  const { t } = useTranslation();
  const isDownloading = downloading?.modelId === model.id;
  const anyDownloading = downloading !== null;
  const progress =
    isDownloading && downloading.total > 0
      ? (downloading.downloaded / downloading.total) * 100
      : 0;

  let borderClass = "border-zinc-100 bg-zinc-50/50 dark:border-zinc-800/60 dark:bg-zinc-900/30";
  if (isActive) {
    borderClass = "border-blue-200 bg-blue-50/50 dark:border-blue-800/60 dark:bg-blue-950/20";
  } else if (isRequired && !model.present) {
    borderClass = "border-rose-200 bg-rose-50/30 dark:border-rose-800/40 dark:bg-rose-950/10";
  } else if (isRecommended) {
    borderClass = "border-amber-200 bg-amber-50/30 dark:border-amber-800/40 dark:bg-amber-950/10";
  }

  return (
    <div className={`flex items-center gap-3 rounded-lg border px-3 py-2.5 ${borderClass}`}>
      <StatusDot present={model.present} active={isActive} />

      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
        <div className="flex items-center gap-1.5">
          <span className="truncate text-sm font-medium text-zinc-800 dark:text-zinc-200">
            {model.label}
          </span>
          {isRequired && (
            <span className="shrink-0 rounded-md bg-rose-100 px-1.5 py-0.5 text-[9px] font-semibold uppercase text-rose-700 dark:bg-rose-900/40 dark:text-rose-300">
              {t("models.required")}
            </span>
          )}
          {isRecommended && !isRequired && (
            <span className="shrink-0 rounded-md bg-amber-100 px-1.5 py-0.5 text-[9px] font-semibold uppercase text-amber-700 dark:bg-amber-900/40 dark:text-amber-300">
              {t("models.recommended")}
            </span>
          )}
        </div>
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
          <span className="text-[10px] text-zinc-500">{t("models.connecting")}</span>
        )}
      </div>

      {isDownloading && (
        <button
          type="button"
          onClick={() => onCancel(model.id)}
          className="shrink-0 rounded-md border border-red-200 bg-red-50 px-2.5 py-1 text-xs font-medium text-red-700 hover:bg-red-100 dark:border-red-800 dark:bg-red-950/40 dark:text-red-300 dark:hover:bg-red-900/60"
        >
          {t("models.cancel")}
        </button>
      )}
      {!isDownloading && model.present && (
        <div className="flex shrink-0 items-center gap-1.5">
          {showUse && !isActive && onSelect && (
            <button
              type="button"
              onClick={() => onSelect(model.id)}
              className="rounded-md border border-blue-200 bg-blue-50 px-2.5 py-1 text-[10px] font-medium text-blue-700 hover:bg-blue-100 dark:border-blue-800 dark:bg-blue-950/40 dark:text-blue-300 dark:hover:bg-blue-900/60"
            >
              {t("models.use")}
            </button>
          )}
          {isActive && (
            <span className="rounded-md bg-blue-100 px-2 py-1 text-[10px] font-medium text-blue-700 dark:bg-blue-950/40 dark:text-blue-300">
              {t("models.active")}
            </span>
          )}
          <span className="rounded-md bg-emerald-50 px-2 py-1 text-[10px] font-medium text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-300">
            {t("models.installed")}
          </span>
          <button
            type="button"
            onClick={() => onDelete(model.id)}
            className="rounded-md p-1 text-zinc-400 hover:bg-red-50 hover:text-red-600 dark:hover:bg-red-950/40 dark:hover:text-red-400"
            title={t("models.delete")}
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" className="h-3.5 w-3.5">
              <path fillRule="evenodd" d="M5 3.25V4H2.75a.75.75 0 0 0 0 1.5h.3l.815 8.15A1.5 1.5 0 0 0 5.357 15h5.285a1.5 1.5 0 0 0 1.493-1.35l.815-8.15h.3a.75.75 0 0 0 0-1.5H11v-.75A2.25 2.25 0 0 0 8.75 1h-1.5A2.25 2.25 0 0 0 5 3.25Zm2.25-.75a.75.75 0 0 0-.75.75V4h3v-.75a.75.75 0 0 0-.75-.75h-1.5ZM6.05 6a.75.75 0 0 1 .787.713l.275 5.5a.75.75 0 0 1-1.498.075l-.275-5.5A.75.75 0 0 1 6.05 6Zm3.9 0a.75.75 0 0 1 .712.787l-.275 5.5a.75.75 0 0 1-1.498-.075l.275-5.5A.75.75 0 0 1 9.95 6Z" clipRule="evenodd" />
            </svg>
          </button>
        </div>
      )}
      {!isDownloading && !model.present && (
        <button
          type="button"
          disabled={anyDownloading}
          onClick={() => onDownload(model.id)}
          className="shrink-0 rounded-md border border-blue-200 bg-blue-50 px-2.5 py-1 text-xs font-medium text-blue-700 hover:bg-blue-100 disabled:cursor-not-allowed disabled:opacity-50 dark:border-blue-800 dark:bg-blue-950/40 dark:text-blue-300 dark:hover:bg-blue-900/60"
        >
          {t("models.download", { size: formatBytes(model.sizeBytes) })}
        </button>
      )}
    </div>
  );
}

function StatusDot({ present, active }: Readonly<{ present: boolean; active: boolean }>) {
  let color = "bg-zinc-300 dark:bg-zinc-600";
  if (active) color = "bg-blue-500";
  else if (present) color = "bg-emerald-500";
  return <span className={`h-2 w-2 shrink-0 rounded-full ${color}`} />;
}

function HardwareChip({ recommendation }: Readonly<{ recommendation: ModelRecommendation }>) {
  const { t } = useTranslation();
  const hw = recommendation.hardware;
  const ramGb = (hw.totalRamBytes / 1_073_741_824).toFixed(0);

  return (
    <div className="flex flex-col gap-1.5 rounded-lg border border-indigo-100 bg-indigo-50/50 px-3 py-2 dark:border-indigo-900/40 dark:bg-indigo-950/20">
      <div className="flex items-center gap-2 text-xs text-indigo-700 dark:text-indigo-300">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" className="h-3.5 w-3.5 shrink-0">
          <path d="M1 8a7 7 0 1 1 14 0A7 7 0 0 1 1 8Zm7.75-4.25a.75.75 0 0 0-1.5 0V8c0 .414.336.75.75.75h3.25a.75.75 0 0 0 0-1.5h-2.5v-3.5Z" />
        </svg>
        <span className="font-medium">{t("models.hardwareDetected")}</span>
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-indigo-600 dark:text-indigo-400">
        <span>{hw.cpuBrand}</span>
        <span>{ramGb} GB RAM</span>
        <span>{hw.gpuName}</span>
      </div>
      {recommendation.warning && (
        <p className="mt-0.5 text-[10px] text-amber-700 dark:text-amber-400">
          ⚠ {recommendation.warning}
        </p>
      )}
    </div>
  );
}
