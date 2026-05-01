import { useTranslation } from "react-i18next";

import { useHardwareRecommendation } from "../../../hooks/useHardwareRecommendation";

function formatBytes(bytes: number): string {
  const gb = bytes / (1024 * 1024 * 1024);
  return `${gb.toFixed(1)} GB`;
}

function confidenceColor(c: "high" | "medium" | "low"): string {
  if (c === "high") return "text-emerald-600 dark:text-emerald-400";
  if (c === "medium") return "text-amber-600 dark:text-amber-400";
  return "text-rose-600 dark:text-rose-400";
}

export function HardwareStep({ onNext }: Readonly<{ onNext: () => void }>) {
  const { t } = useTranslation();
  const { data, loading, error } = useHardwareRecommendation();

  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-8 px-8">
      {/* CPU icon */}
      <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-surface-sunken text-content-tertiary">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <rect x="4" y="4" width="16" height="16" rx="2" />
          <rect x="9" y="9" width="6" height="6" />
          <line x1="9" y1="1" x2="9" y2="4" /><line x1="15" y1="1" x2="15" y2="4" />
          <line x1="9" y1="20" x2="9" y2="23" /><line x1="15" y1="20" x2="15" y2="23" />
          <line x1="20" y1="9" x2="23" y2="9" /><line x1="20" y1="14" x2="23" y2="14" />
          <line x1="1" y1="9" x2="4" y2="9" /><line x1="1" y1="14" x2="4" y2="14" />
        </svg>
      </div>

      <div className="flex flex-col items-center gap-4 text-center">
        <h2 className="text-display-md font-semibold tracking-tight text-content-primary">
          {t("onboarding.hardwareTitle")}
        </h2>

        {loading && (
          <p className="text-ui-md text-content-secondary animate-pulse">
            {t("onboarding.hardwareDetecting")}
          </p>
        )}

        {error && (
          <p className="text-ui-sm text-semantic-danger">{error}</p>
        )}

        {data && (
          <div className="flex w-full max-w-sm flex-col gap-4 rounded-lg border border-subtle bg-surface-sunken p-4">
            {/* Hardware summary */}
            <div className="grid grid-cols-2 gap-3 text-left text-ui-sm">
              <div>
                <p className="text-content-tertiary">{t("onboarding.hwCpu")}</p>
                <p className="font-medium text-content-primary">{data.hardware.cpuBrand}</p>
              </div>
              <div>
                <p className="text-content-tertiary">{t("onboarding.hwCores")}</p>
                <p className="font-medium text-content-primary">{data.hardware.cpuCores}</p>
              </div>
              <div>
                <p className="text-content-tertiary">{t("onboarding.hwRam")}</p>
                <p className="font-medium text-content-primary">{formatBytes(data.hardware.totalRamBytes)}</p>
              </div>
              <div>
                <p className="text-content-tertiary">{t("onboarding.hwGpu")}</p>
                <p className="font-medium text-content-primary">{data.hardware.gpuName || "—"}</p>
              </div>
            </div>

            {/* Recommendation */}
            <div className="border-t border-subtle pt-3">
              <div className="flex items-center justify-between">
                <p className="text-ui-xs font-medium uppercase tracking-wide text-content-tertiary">
                  {t("onboarding.hwRecommended")}
                </p>
                <span className={`text-ui-xs font-medium ${confidenceColor(data.asr.confidence)}`}>
                  {t(`onboarding.confidence.${data.asr.confidence}`)}
                </span>
              </div>
              <p className="mt-1 text-ui-sm text-content-secondary">{data.asr.reason}</p>
              {data.warning && (
                <p className="mt-2 text-ui-xs text-amber-600 dark:text-amber-400">
                  ⚠ {data.warning}
                </p>
              )}
            </div>
          </div>
        )}
      </div>

      <button
        type="button"
        onClick={onNext}
        disabled={loading}
        className="rounded-full bg-accent-600 px-8 py-2.5 text-ui-md font-medium text-white shadow-sm transition-all hover:bg-accent-700 hover:shadow-md active:scale-[0.98] disabled:bg-content-placeholder disabled:shadow-none"
      >
        {t("onboarding.continue")}
      </button>
    </div>
  );
}
