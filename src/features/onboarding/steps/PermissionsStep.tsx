import { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { Mic, Check } from "lucide-react";

import { startStreaming, stopStreaming, getMeetingId, deleteMeeting } from "../../../ipc/client";

/**
 * Permissions step — attempts a very short streaming session to trigger
 * the macOS permission dialog. If it succeeds, permissions are granted.
 * If it fails with a permission error, we show a button to retry or
 * open System Settings.
 */
export function PermissionsStep({ onNext }: Readonly<{ onNext: () => void }>) {
  const { t } = useTranslation();
  const [status, setStatus] = useState<"idle" | "checking" | "granted" | "denied">("idle");

  const checkPermissions = useCallback(async () => {
    setStatus("checking");
    try {
      // Wait for the "started" event before querying meetingId — the
      // recorder only populates the session→meeting mapping after
      // processing this event asynchronously.
      let resolveStarted: () => void;
      const started = new Promise<void>((r) => { resolveStarted = r; });

      const sessionId = await startStreaming(
        { chunkMs: 1_000, silenceRmsThreshold: 0.5 },
        (evt) => { if (evt.type === "started") resolveStarted(); },
      );

      await started;

      const meetingId = await getMeetingId(sessionId);
      await stopStreaming(sessionId);
      if (meetingId) await deleteMeeting(meetingId).catch(() => {});
      setStatus("granted");
    } catch {
      setStatus("denied");
    }
  }, []);

  // Auto-check on mount
  useEffect(() => {
    checkPermissions();
  }, [checkPermissions]);

  // Auto-advance when granted
  useEffect(() => {
    if (status === "granted") {
      const timer = setTimeout(onNext, 800);
      return () => clearTimeout(timer);
    }
  }, [status, onNext]);

  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-8 px-8">
      {/* Microphone icon */}
      <div className={`flex h-16 w-16 items-center justify-center rounded-2xl transition-colors ${
        status === "granted"
          ? "bg-emerald-50 text-emerald-600 dark:bg-emerald-950/40 dark:text-emerald-400"
          : status === "denied"
            ? "bg-rose-50 text-rose-600 dark:bg-rose-950/40 dark:text-rose-400"
            : "bg-surface-sunken text-content-tertiary"
      }`}>
        <Mic className="h-8 w-8" />
      </div>

      <div className="flex flex-col items-center gap-3 text-center">
        <h2 className="text-display-md font-semibold tracking-tight text-content-primary">
          {t("onboarding.permissionsTitle")}
        </h2>

        {status === "checking" && (
          <p className="text-ui-md text-content-secondary animate-pulse">
            {t("onboarding.permissionsChecking")}
          </p>
        )}

        {status === "granted" && (
          <div className="flex items-center gap-2 text-ui-md font-medium text-emerald-600 dark:text-emerald-400">
            <Check className="h-4 w-4" />
            {t("onboarding.permissionsGranted")}
          </div>
        )}

        {status === "denied" && (
          <>
            <p className="max-w-sm text-ui-md text-content-secondary">
              {t("onboarding.permissionsDenied")}
            </p>
            <div className="flex gap-3">
              <button
                type="button"
                onClick={checkPermissions}
                className="rounded-full bg-accent-600 px-6 py-2 text-ui-sm font-medium text-white shadow-sm transition-all hover:bg-accent-700"
              >
                {t("onboarding.permissionsRetry")}
              </button>
              <button
                type="button"
                onClick={onNext}
                className="rounded-full border border-subtle px-6 py-2 text-ui-sm font-medium text-content-secondary transition-all hover:bg-surface-sunken"
              >
                {t("onboarding.skip")}
              </button>
            </div>
          </>
        )}

        {status === "idle" && (
          <p className="text-ui-md text-content-tertiary">
            {t("onboarding.permissionsHint")}
          </p>
        )}
      </div>
    </div>
  );
}
