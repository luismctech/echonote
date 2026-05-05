import { useTranslation } from "react-i18next";
import { Cpu } from "lucide-react";

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
        <Cpu className="h-8 w-8" />
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
