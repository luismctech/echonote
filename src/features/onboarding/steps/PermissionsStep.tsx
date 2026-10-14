import { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { Mic, Check } from "lucide-react";

import { startStreaming, stopStreaming, getMeetingId, deleteMeeting } from "../../../ipc/client";
import { isIpcError } from "../../../types/ipc-error";

type Status = "idle" | "checking" | "granted" | "denied" | "deferred";

/** Human-readable detail for a failed probe; plain strings pass through. */
function errorDetail(err: unknown): string | null {
  if (isIpcError(err)) return err.message;
  return typeof err === "string" ? err : null;
}

/**
 * Permissions step — attempts a very short streaming session to trigger
 * the macOS permission dialog. If it succeeds, the microphone is
 * usable. A failure that is *not* about the microphone (no ASR model
 * installed yet) is reported as deferred rather than denied, so the
 * step never blames the user's audio setup for a backend problem.
 */
export function PermissionsStep({ onNext }: Readonly<{ onNext: () => void }>) {
  const { t } = useTranslation();
  const [status, setStatus] = useState<Status>("idle");
  const [detail, setDetail] = useState<string | null>(null);

  const checkPermissions = useCallback(async () => {
    setStatus("checking");
    setDetail(null);
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
    } catch (err) {
      // The probe runs a full streaming session, so it can fail for
      // reasons that have nothing to do with audio access.
      const modelMissing = isIpcError(err) && err.code === "modelNotReady";
      setStatus(modelMissing ? "deferred" : "denied");
      setDetail(errorDetail(err));
    }
  }, []);

  // Auto-check on mount
  useEffect(() => {
    checkPermissions();
  }, [checkPermissions]);

  // Auto-advance when granted or when the check could not run
  useEffect(() => {
    if (status === "granted" || status === "deferred") {
      const timer = setTimeout(onNext, 800);
      return () => clearTimeout(timer);
    }
  }, [status, onNext]);

  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-8 px-8">
      {/* Microphone icon */}
      <div className={`flex h-16 w-16 items-center justify-center rounded-2xl transition-colors ${
        status === "granted" || status === "deferred"
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

        {status === "deferred" && (
          <p className="max-w-sm text-ui-md text-content-secondary">
            {t("onboarding.permissionsDeferred")}
          </p>
        )}

        {status === "denied" && (
          <>
            <p className="max-w-sm text-ui-md text-content-secondary">
              {t("onboarding.permissionsDenied")}
            </p>
            {detail && (
              <p className="max-w-sm break-words text-ui-sm text-content-tertiary">{detail}</p>
            )}
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
