import { useTranslation } from "react-i18next";

import { LogoAnimated } from "../../../components/Logo";

export function WelcomeStep({ onNext }: Readonly<{ onNext: () => void }>) {
  const { t } = useTranslation();

  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-8 px-8">
      <LogoAnimated size={96} className="drop-shadow-lg" />

      <div className="flex flex-col items-center gap-3 text-center">
        <h1 className="text-display-lg font-semibold tracking-tight text-content-primary">
          EchoNote
        </h1>
        <p className="max-w-sm text-reading-md text-content-secondary">
          {t("onboarding.welcomeSubtitle")}
        </p>
      </div>

      <button
        type="button"
        onClick={onNext}
        className="rounded-full bg-accent-600 px-8 py-2.5 text-ui-md font-medium text-white shadow-sm transition-all hover:bg-accent-700 hover:shadow-md active:scale-[0.98]"
      >
        {t("onboarding.getStarted")}
      </button>
    </div>
  );
}
