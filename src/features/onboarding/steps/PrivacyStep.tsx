import { useTranslation } from "react-i18next";
import { ShieldCheck, Check } from "lucide-react";

const BULLETS = [
  "onboarding.privacyBullet1",
  "onboarding.privacyBullet2",
  "onboarding.privacyBullet3",
] as const;

export function PrivacyStep({ onNext }: Readonly<{ onNext: () => void }>) {
  const { t } = useTranslation();

  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-8 px-8">
      {/* Shield icon */}
      <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-emerald-50 text-emerald-600 dark:bg-emerald-950/40 dark:text-emerald-400">
        <ShieldCheck className="h-8 w-8" />
      </div>

      <div className="flex flex-col items-center gap-4 text-center">
        <h2 className="max-w-md text-display-md font-semibold leading-snug tracking-tight text-content-primary">
          {t("onboarding.privacyTitle")}
        </h2>

        <ul className="flex max-w-sm flex-col gap-2.5 text-left">
          {BULLETS.map((key) => (
            <li key={key} className="flex items-start gap-2.5 text-ui-md text-content-secondary">
              <span className="mt-1 inline-flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-400">
                <Check className="h-2.5 w-2.5" />
              </span>
              {t(key)}
            </li>
          ))}
        </ul>
      </div>

      <button
        type="button"
        onClick={onNext}
        className="rounded-full bg-accent-600 px-8 py-2.5 text-ui-md font-medium text-white shadow-sm transition-all hover:bg-accent-700 hover:shadow-md active:scale-[0.98]"
      >
        {t("onboarding.continue")}
      </button>
    </div>
  );
}
