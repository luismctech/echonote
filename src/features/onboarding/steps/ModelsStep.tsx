import { useEffect, useMemo } from "react";
import { useTranslation } from "react-i18next";

import { useHardwareRecommendation } from "../../../hooks/useHardwareRecommendation";
import { useModelManager } from "../../../hooks/useModelManager";
import type { ModelInfo } from "../../../types/models";

function formatSize(bytes: number): string {
  const mb = bytes / (1024 * 1024);
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
  return `${Math.round(mb)} MB`;
}

function formatProgress(downloaded: number, total: number): string {
  if (total === 0) return "…";
  const pct = Math.round((downloaded / total) * 100);
  return `${pct}% · ${formatSize(downloaded)} / ${formatSize(total)}`;
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

export function ModelsStep({ onNext }: Readonly<{ onNext: () => void }>) {
  const { t } = useTranslation();
  const hw = useHardwareRecommendation();
  const mm = useModelManager();

  // Determine which models are essential (ASR is required, LLM is optional but recommended)
  const essentialIds = useMemo(() => {
    const ids: string[] = [];
    if (hw.data?.asr) ids.push(hw.data.asr.modelId);
    if (hw.data?.llm) ids.push(hw.data.llm.modelId);
    return ids;
  }, [hw.data]);

  const allEssentialPresent = essentialIds.length > 0 &&
    essentialIds.every((id) => mm.models.find((m) => m.id === id)?.present);

  // Auto-start downloading the first missing essential model
  useEffect(() => {
    if (mm.downloading) return;
    const missing = essentialIds.find((id) => !mm.models.find((m) => m.id === id)?.present);
    if (missing) mm.download(missing);
  }, [essentialIds, mm.models, mm.downloading, mm.download]);

  const sections = useMemo(() => groupBySections(mm.models), [mm.models]);

  return (
    <div className="flex min-h-0 flex-1 flex-col items-center gap-6 overflow-hidden px-8 pt-8 pb-4">
      {/* Download icon */}
      <div className="flex h-16 w-16 shrink-0 items-center justify-center rounded-2xl bg-surface-sunken text-content-tertiary">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="7 10 12 15 17 10" />
          <line x1="12" y1="15" x2="12" y2="3" />
        </svg>
      </div>

      <div className="flex flex-col items-center gap-2 text-center">
        <h2 className="text-display-md font-semibold tracking-tight text-content-primary">
          {t("onboarding.modelsTitle")}
        </h2>
        <p className="max-w-sm text-ui-sm text-content-secondary">
          {t("onboarding.modelsSubtitle")}
        </p>
      </div>

      {mm.error && (
        <p className="text-ui-sm text-semantic-danger">{mm.error}</p>
      )}

      {/* Model list grouped by section */}
      <div className="flex min-h-0 w-full max-w-md flex-col gap-5 overflow-y-auto pb-2">
        {sections.map((section) => (
          <div key={section.key} className="flex flex-col gap-2">
            {/* Section header */}
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

            {/* Models in section */}
            {section.models.map((model) => {
              const isEssential = essentialIds.includes(model.id);
              const isDownloading = mm.downloading?.modelId === model.id;
              const localizedDesc = t(`models.desc.${model.id}`, { defaultValue: "" });
              const description = localizedDesc || model.description;

              return (
                <div
                  key={model.id}
                  className={`flex items-center gap-3 rounded-lg border p-3 transition-colors ${
                    model.present
                      ? "border-emerald-200 bg-emerald-50/50 dark:border-emerald-900 dark:bg-emerald-950/20"
                      : "border-subtle bg-surface-sunken"
                  }`}
                >
                  {/* Status indicator */}
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center">
                    {model.present ? (
                      <svg width="20" height="20" viewBox="0 0 16 16" fill="none" className="text-emerald-600 dark:text-emerald-400">
                        <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-6.25 6.25a.75.75 0 0 1-1.06 0L3.22 8.28a.75.75 0 0 1 1.06-1.06L7 9.94l5.72-5.72a.75.75 0 0 1 1.06 0z" fill="currentColor" />
                      </svg>
                    ) : isDownloading ? (
                      <div className="h-5 w-5 animate-spin rounded-full border-2 border-accent-400 border-t-transparent" />
                    ) : (
                      <div className="h-5 w-5 rounded-full border-2 border-content-placeholder/30" />
                    )}
                  </div>

                  {/* Info */}
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <p className="truncate text-ui-sm font-medium text-content-primary">{model.label}</p>
                      {isEssential && (
                        <span className="shrink-0 rounded bg-accent-100 px-1.5 py-0.5 text-micro font-medium text-accent-700 dark:bg-accent-50 dark:text-accent-900">
                          {t("models.recommended")}
                        </span>
                      )}
                    </div>
                    <p className="text-ui-xs text-content-tertiary">
                      {description} · {formatSize(model.sizeBytes)}
                    </p>

                    {/* Download progress bar */}
                    {isDownloading && mm.downloading && (
                      <div className="mt-1.5 flex flex-col gap-1">
                        <div className="h-1.5 w-full overflow-hidden rounded-full bg-content-placeholder/20">
                          <div
                            className="h-full rounded-full bg-accent-600 transition-all duration-300"
                            style={{ width: mm.downloading.total > 0 ? `${(mm.downloading.downloaded / mm.downloading.total) * 100}%` : "0%" }}
                          />
                        </div>
                        <p className="text-micro text-content-tertiary">
                          {formatProgress(mm.downloading.downloaded, mm.downloading.total)}
                        </p>
                      </div>
                    )}
                  </div>

                  {/* Action */}
                  {!model.present && !isDownloading && (
                    <button
                      type="button"
                      onClick={() => mm.download(model.id)}
                      disabled={mm.downloading != null}
                      className="shrink-0 rounded-full border border-subtle px-3 py-1 text-ui-xs font-medium text-content-secondary transition-colors hover:bg-surface-elevated disabled:opacity-50"
                    >
                      {t("onboarding.modelsDownload")}
                    </button>
                  )}
                  {isDownloading && (
                    <button
                      type="button"
                      onClick={() => mm.cancelDl(model.id)}
                      className="shrink-0 text-ui-xs text-content-tertiary underline hover:text-content-secondary"
                    >
                      {t("onboarding.modelsCancel")}
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        ))}
      </div>

      <button
        type="button"
        onClick={onNext}
        disabled={!allEssentialPresent}
        className="shrink-0 rounded-full bg-accent-600 px-8 py-2.5 text-ui-md font-medium text-white shadow-sm transition-all hover:bg-accent-700 hover:shadow-md active:scale-[0.98] disabled:bg-content-placeholder disabled:shadow-none"
      >
        {allEssentialPresent ? t("onboarding.continue") : t("onboarding.modelsWaiting")}
      </button>
    </div>
  );
}
