import { useEffect } from "react";
import { useTranslation } from "react-i18next";

import { LogoGlow } from "../../../components/Logo";

export function CompletionStep({ onComplete }: Readonly<{ onComplete: () => void }>) {
  const { t } = useTranslation();

  // Auto-transition to the app after a brief celebration
  useEffect(() => {
    const timer = setTimeout(onComplete, 2500);
    return () => clearTimeout(timer);
  }, [onComplete]);

  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-6 px-8 animate-text-appear opacity-0">
      <LogoGlow size={72} className="drop-shadow-xl" />

      <div className="flex flex-col items-center gap-2 text-center">
        <h2 className="text-display-lg font-semibold tracking-tight text-content-primary">
          {t("onboarding.completionTitle")}
        </h2>
        <p className="text-reading-md text-content-secondary">
          {t("onboarding.completionSubtitle")}
        </p>
      </div>

      {/* Animated check */}
      <div className="flex h-14 w-14 items-center justify-center rounded-full bg-emerald-100 text-emerald-600 dark:bg-emerald-900/40 dark:text-emerald-400">
        <svg width="28" height="28" viewBox="0 0 16 16" fill="currentColor">
          <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-6.25 6.25a.75.75 0 0 1-1.06 0L3.22 8.28a.75.75 0 0 1 1.06-1.06L7 9.94l5.72-5.72a.75.75 0 0 1 1.06 0z" />
        </svg>
      </div>

      <button
        type="button"
        onClick={onComplete}
        className="text-ui-sm text-content-tertiary underline transition-colors hover:text-content-secondary"
      >
        {t("onboarding.completionSkip")}
      </button>
    </div>
  );
}
