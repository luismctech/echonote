import { useEffect } from "react";
import { useTranslation } from "react-i18next";
import { Check } from "lucide-react";

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
        <Check className="h-7 w-7" />
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
