import { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

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
      // Start a minimal session — macOS will prompt for mic access if needed
      const sessionId = await startStreaming(
        { chunkMs: 1_000, silenceRmsThreshold: 0.5 },
        () => { /* discard events */ },
      );
      // Grab the meeting id before stopping so we can clean it up
      const meetingId = await getMeetingId(sessionId);
      // Immediately stop — we only wanted the permission prompt
      await stopStreaming(sessionId);
      // Delete the empty meeting so it doesn't pollute history
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
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <rect x="9" y="2" width="6" height="11" rx="3" />
          <path d="M5 10a7 7 0 0 0 14 0" />
          <line x1="12" y1="19" x2="12" y2="22" />
        </svg>
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
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-6.25 6.25a.75.75 0 0 1-1.06 0L3.22 8.28a.75.75 0 0 1 1.06-1.06L7 9.94l5.72-5.72a.75.75 0 0 1 1.06 0z" /></svg>
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
