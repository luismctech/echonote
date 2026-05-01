import { useTranslation } from "react-i18next";

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
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
          <path d="m9 12 2 2 4-4" />
        </svg>
      </div>

      <div className="flex flex-col items-center gap-4 text-center">
        <h2 className="max-w-md text-display-md font-semibold leading-snug tracking-tight text-content-primary">
          {t("onboarding.privacyTitle")}
        </h2>

        <ul className="flex max-w-sm flex-col gap-2.5 text-left">
          {BULLETS.map((key) => (
            <li key={key} className="flex items-start gap-2.5 text-ui-md text-content-secondary">
              <span className="mt-1 inline-flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-400">
                <svg width="10" height="10" viewBox="0 0 16 16" fill="currentColor"><path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-6.25 6.25a.75.75 0 0 1-1.06 0L3.22 8.28a.75.75 0 0 1 1.06-1.06L7 9.94l5.72-5.72a.75.75 0 0 1 1.06 0z" /></svg>
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
