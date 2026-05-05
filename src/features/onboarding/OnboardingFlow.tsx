import { useCallback, useState } from "react";
import { useTranslation } from "react-i18next";
import { ArrowLeft } from "lucide-react";

import { WelcomeStep } from "./steps/WelcomeStep";
import { PrivacyStep } from "./steps/PrivacyStep";
import { PermissionsStep } from "./steps/PermissionsStep";
import { HardwareStep } from "./steps/HardwareStep";
import { ModelsStep } from "./steps/ModelsStep";
import { TestRecordingStep } from "./steps/TestRecordingStep";
import { CompletionStep } from "./steps/CompletionStep";

const STEPS = [
  "welcome",
  "privacy",
  "permissions",
  "hardware",
  "models",
  "test-recording",
  "completion",
] as const;

type Step = (typeof STEPS)[number];

export function OnboardingFlow({ onComplete }: Readonly<{ onComplete: () => void }>) {
  const { t } = useTranslation();
  const [step, setStep] = useState<Step>("welcome");

  const currentIndex = STEPS.indexOf(step);
  const progress = ((currentIndex + 1) / STEPS.length) * 100;

  const goNext = useCallback(() => {
    const idx = STEPS.indexOf(step);
    const next = STEPS[idx + 1];
    if (idx < STEPS.length - 1 && next) {
      setStep(next);
    }
  }, [step]);

  const goBack = useCallback(() => {
    const idx = STEPS.indexOf(step);
    const prev = STEPS[idx - 1];
    if (idx > 0 && prev) {
      setStep(prev);
    }
  }, [step]);

  return (
    <div className="flex h-full flex-col bg-surface-base">
      {/* ── Progress bar ── */}
      <div className="h-1 w-full bg-content-placeholder/10">
        <div
          className="h-full bg-accent-600 transition-all duration-500 ease-out"
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* ── Step content with crossfade ── */}
      <div
        key={step}
        className="flex min-h-0 flex-1 flex-col overflow-hidden animate-text-appear opacity-0"
      >
        {step === "welcome" && <WelcomeStep onNext={goNext} />}
        {step === "privacy" && <PrivacyStep onNext={goNext} />}
        {step === "permissions" && <PermissionsStep onNext={goNext} />}
        {step === "hardware" && <HardwareStep onNext={goNext} />}
        {step === "models" && <ModelsStep onNext={goNext} />}
        {step === "test-recording" && <TestRecordingStep onNext={goNext} />}
        {step === "completion" && <CompletionStep onComplete={onComplete} />}
      </div>

      {/* ── Footer: back button + step dots ── */}
      {step !== "completion" && (
        <footer className="flex shrink-0 items-center justify-between px-6 py-4">
          {/* Back button */}
          {currentIndex > 0 ? (
            <button
              type="button"
              onClick={goBack}
              className="flex items-center gap-1 text-ui-sm text-content-tertiary transition-colors hover:text-content-secondary"
            >
              <ArrowLeft className="h-3.5 w-3.5" />
              {t("onboarding.back")}
            </button>
          ) : (
            <div />
          )}

          {/* Step dots */}
          <div className="flex items-center gap-1.5">
            {STEPS.map((s, i) => (
              <div
                key={s}
                className={`h-1.5 rounded-full transition-all duration-300 ${
                  i === currentIndex
                    ? "w-4 bg-accent-600"
                    : i < currentIndex
                      ? "w-1.5 bg-accent-400"
                      : "w-1.5 bg-content-placeholder/30"
                }`}
              />
            ))}
          </div>

          {/* Spacer for symmetry */}
          <div className="w-16" />
        </footer>
      )}
    </div>
  );
}
