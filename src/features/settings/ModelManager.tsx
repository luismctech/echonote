/**
 * `ModelManager` — panel for downloading and managing ML models.
 *
 * Shown as a modal/overlay toggled from the app header. Displays each
 * downloadable model with its status (present / missing) and a
 * download button with real-time progress.
 */

import { useTranslation } from "react-i18next";
import { RefreshCw, Trash2 } from "lucide-react";

import { Modal } from "../../components/Modal";
import { ModelProviderLogo } from "../../components/ModelProviderLogo";
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
  onReplayOnboarding,
}: Readonly<{
  state: UseModelManager;
  onClose: () => void;
  onReplayOnboarding: () => void;
}>) {
  const { models, loading, downloading, error, activeLlm, activeAsr, activeEmbedder } = state;
  const { t } = useTranslation();
  const { data: recommendation } = useHardwareRecommendation();

  const sections = groupBySections(models);

  const activeIds: Record<string, string | null> = {
    llm: activeLlm,
    asr: activeAsr,
    embedder: activeEmbedder,
  };
  const selectHandlers: Record<string, (id: string) => void> = {
    llm: state.selectLlm,
    asr: state.selectAsr,
    embedder: state.selectEmbedder,
  };

  const recommendedIds = new Set<string>();
  if (recommendation) {
    recommendedIds.add(recommendation.asr.modelId);
    if (recommendation.llm) recommendedIds.add(recommendation.llm.modelId);
  }

  return (
    <Modal open onClose={onClose} className="w-full max-w-2xl">
      <div className="flex max-h-[80vh] w-full flex-col gap-3 overflow-hidden rounded-xl border bg-surface-elevated p-5 shadow-xl">
        <header className="flex shrink-0 items-center justify-between">
          <h2 className="text-ui-lg font-semibold">{t("models.title")}</h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md px-2 py-1 text-ui-sm text-content-tertiary hover:bg-surface-sunken"
          >
            {t("models.close")}
          </button>
        </header>

        {recommendation && <HardwareCard recommendation={recommendation} />}

        {error && (
          <p className="rounded-md bg-red-50 px-3 py-2 text-ui-sm text-red-700 dark:bg-red-950/40 dark:text-red-300">
            {error}
          </p>
        )}

        {loading ? (
          <p className="py-6 text-center text-ui-md text-content-tertiary">{t("models.loading")}</p>
        ) : (
          <div className="flex min-h-0 flex-col gap-5 overflow-y-auto pb-4">
            {sections.map((s) => (
              <ModelSection
                key={s.key}
                section={s}
                models={s.models}
                downloading={downloading}
                activeIds={activeIds}
                recommendedIds={recommendedIds}
                selectHandlers={selectHandlers}
                onDownload={state.download}
                onCancel={state.cancelDl}
                onDelete={state.remove}
              />
            ))}
          </div>
        )}

        {/* Footer */}
        <div className="flex shrink-0 items-center justify-center border-t border-subtle pt-3">
          <button
            type="button"
            onClick={onReplayOnboarding}
            className="flex items-center gap-1.5 text-ui-xs text-content-tertiary transition-colors hover:text-content-secondary"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            {t("models.replayOnboarding")}
          </button>
        </div>
      </div>
    </Modal>
  );
}

/** Logical sections that group model kinds together. */
const MODEL_SECTIONS = [
  { key: "transcription", kinds: ["asr"], chip: "required" as const },
  { key: "summaries", kinds: ["llm"], chip: null },
  { key: "audio", kinds: ["vad"], chip: "required" as const },
  { key: "speakers", kinds: ["embedder", "segmenter"], chip: null },
];

function groupBySections(models: ModelInfo[]): { key: string; chip: "required" | null; models: ModelInfo[] }[] {
  return MODEL_SECTIONS.map((s) => ({
    key: s.key,
    chip: s.chip,
    models: models.filter((m) => s.kinds.includes(m.kind)),
  })).filter((s) => s.models.length > 0);
}

/** Which kinds allow selecting the active model. */
const SELECTABLE_KINDS = new Set(["llm", "asr", "embedder"]);

function ModelSection({
  section,
  models,
  downloading,
  activeIds,
  recommendedIds,
  selectHandlers,
  onDownload,
  onCancel,
  onDelete,
}: Readonly<{
  section: { key: string; chip: "required" | null };
  models: ModelInfo[];
  downloading: DownloadProgress | null;
  activeIds: Record<string, string | null>;
  recommendedIds: Set<string>;
  selectHandlers: Record<string, (id: string) => void>;
  onDownload: (id: string) => void;
  onCancel: (id: string) => void;
  onDelete: (id: string) => void;
}>) {
  const { t } = useTranslation();

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-col gap-0.5">
        <div className="flex items-center gap-2">
          <h3 className="text-ui-sm font-semibold text-content-primary">
            {t(`models.section${section.key.charAt(0).toUpperCase()}${section.key.slice(1)}`)}
          </h3>
          {section.chip === "required" && (
            <span className="shrink-0 rounded bg-rose-100 px-1.5 py-0.5 text-micro font-medium text-rose-700 dark:bg-rose-900/40 dark:text-rose-300">
              {t("models.required")}
            </span>
          )}
        </div>
        <p className="text-ui-xs text-content-tertiary">
          {t(`models.section${section.key.charAt(0).toUpperCase()}${section.key.slice(1)}Desc`)}
        </p>
      </div>
      {models.map((m) => {
        const selectable = SELECTABLE_KINDS.has(m.kind);
        const activeId = activeIds[m.kind] ?? null;
        return (
          <ModelRow
            key={m.id}
            model={m}
            downloading={downloading}
            isActive={selectable && m.id === activeId}
            isRecommended={recommendedIds.has(m.id)}
            showUse={selectable}
            onDownload={onDownload}
            onCancel={onCancel}
            onDelete={onDelete}
            {...(selectable && selectHandlers[m.kind] ? { onSelect: selectHandlers[m.kind] } : {})}
          />
        );
      })}
    </div>
  );
}

function ModelRow({
  model,
  downloading,
  isActive,
  isRecommended,
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

  // Use i18n description, fall back to Rust description
  const localizedDesc = t(`models.desc.${model.id}`, { defaultValue: "" });
  const description = localizedDesc || model.description;

  return (
    <div
      className={`flex items-center gap-3 rounded-lg border p-3 transition-colors ${
        model.present
          ? "border-emerald-200 bg-emerald-50/50 dark:border-emerald-900 dark:bg-emerald-950/20"
          : "border-subtle bg-surface-sunken"
      }`}
    >
      {/* Provider logo / download spinner */}
      <div className="flex h-7 w-7 shrink-0 items-center justify-center">
        {isDownloading ? (
          <div className="h-5 w-5 animate-spin rounded-full border-2 border-accent-400 border-t-transparent" />
        ) : (
          <ModelProviderLogo modelId={model.id} size={28} />
        )}
      </div>

      {/* Info */}
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="truncate text-ui-sm font-medium text-content-primary">
            {model.label}
          </span>
          {isRecommended && (
            <span className="shrink-0 rounded bg-accent-100 px-1.5 py-0.5 text-micro font-medium text-accent-700 dark:bg-accent-50 dark:text-accent-900">
              {t("models.recommended")}
            </span>
          )}
          {isActive && (
            <span className="shrink-0 rounded bg-blue-100 px-1.5 py-0.5 text-micro font-medium text-blue-700 dark:bg-blue-950/40 dark:text-blue-300">
              {t("models.active")}
            </span>
          )}
        </div>
        <p className="text-ui-xs text-content-tertiary">
          {description} · {formatBytes(model.sizeBytes)}
        </p>

        {/* Download progress bar */}
        {isDownloading && downloading.total > 0 && (
          <div className="mt-1.5 flex items-center gap-2">
            <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-content-placeholder/20">
              <div
                className="h-full rounded-full bg-accent-600 transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
            <span className="shrink-0 text-micro tabular-nums text-content-tertiary">
              {formatBytes(downloading.downloaded)} / {formatBytes(downloading.total)}
            </span>
          </div>
        )}
        {isDownloading && downloading.total === 0 && (
          <span className="mt-1 block text-micro text-content-tertiary">{t("models.connecting")}</span>
        )}
      </div>

      {/* Actions */}
      {isDownloading && (
        <button
          type="button"
          onClick={() => onCancel(model.id)}
          className="shrink-0 text-ui-xs text-content-tertiary underline hover:text-content-secondary"
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
              className="rounded-full border border-subtle px-3 py-1 text-ui-xs font-medium text-content-secondary transition-colors hover:bg-surface-elevated"
            >
              {t("models.use")}
            </button>
          )}
          <button
            type="button"
            onClick={() => onDelete(model.id)}
            className="rounded-full p-1.5 text-content-placeholder hover:bg-red-50 hover:text-red-600 dark:hover:bg-red-950/40 dark:hover:text-red-400"
            title={t("models.delete")}
          >
            <Trash2 className="h-3.5 w-3.5" />
          </button>
        </div>
      )}
      {!isDownloading && !model.present && (
        <button
          type="button"
          disabled={anyDownloading}
          onClick={() => onDownload(model.id)}
          className="shrink-0 rounded-full border border-subtle px-3 py-1 text-ui-xs font-medium text-content-secondary transition-colors hover:bg-surface-elevated disabled:opacity-50"
        >
          {t("models.download")}
        </button>
      )}
    </div>
  );
}

function confidenceColor(c: "high" | "medium" | "low"): string {
  if (c === "high") return "text-emerald-600 dark:text-emerald-400";
  if (c === "medium") return "text-amber-600 dark:text-amber-400";
  return "text-rose-600 dark:text-rose-400";
}

function HardwareCard({ recommendation }: Readonly<{ recommendation: ModelRecommendation }>) {
  const { t } = useTranslation();
  const hw = recommendation.hardware;
  const ramGb = (hw.totalRamBytes / 1_073_741_824).toFixed(1);

  return (
    <div className="flex shrink-0 flex-col gap-3 rounded-lg border border-subtle bg-surface-sunken p-4">
      {/* 2×2 hardware grid */}
      <div className="grid grid-cols-2 gap-3 text-ui-sm">
        <div>
          <p className="text-content-tertiary">{t("onboarding.hwCpu")}</p>
          <p className="font-medium text-content-primary">{hw.cpuBrand}</p>
        </div>
        <div>
          <p className="text-content-tertiary">{t("onboarding.hwCores")}</p>
          <p className="font-medium text-content-primary">{hw.cpuCores}</p>
        </div>
        <div>
          <p className="text-content-tertiary">{t("onboarding.hwRam")}</p>
          <p className="font-medium text-content-primary">{ramGb} GB</p>
        </div>
        <div>
          <p className="text-content-tertiary">{t("onboarding.hwGpu")}</p>
          <p className="font-medium text-content-primary">{hw.gpuName || "—"}</p>
        </div>
      </div>

      {/* Recommendation */}
      <div className="border-t border-subtle pt-3">
        <div className="flex items-center justify-between">
          <p className="text-ui-xs font-medium uppercase tracking-wide text-content-tertiary">
            {t("onboarding.hwRecommended")}
          </p>
          <span className={`text-ui-xs font-medium ${confidenceColor(recommendation.asr.confidence)}`}>
            {t(`onboarding.confidence.${recommendation.asr.confidence}`)}
          </span>
        </div>
        <p className="mt-1 text-ui-sm text-content-secondary">{recommendation.asr.reason}</p>
        {recommendation.warning && (
          <p className="mt-2 text-ui-xs text-amber-600 dark:text-amber-400">
            ⚠ {recommendation.warning}
          </p>
        )}
      </div>
    </div>
  );
}
