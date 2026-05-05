import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { AudioLines } from "lucide-react";

import { startStreaming, stopStreaming, getMeetingId, deleteMeeting } from "../../../ipc/client";
import type { TranscriptEvent } from "../../../types/streaming";

const TEST_DURATION_MS = 20_000;

export function TestRecordingStep({ onNext }: Readonly<{ onNext: () => void }>) {
  const { t } = useTranslation();
  const [phase, setPhase] = useState<"idle" | "recording" | "done">("idle");
  const [transcript, setTranscript] = useState<string[]>([]);
  const [elapsed, setElapsed] = useState(0);
  const sessionRef = useRef<string | null>(null);
  const meetingIdRef = useRef<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef(0);

  const handleEvent = useCallback((evt: TranscriptEvent) => {
    if (evt.type === "started" && sessionRef.current) {
      // Resolve meetingId as soon as the recorder has created it
      getMeetingId(sessionRef.current)
        .then((mid) => { meetingIdRef.current = mid ?? null; })
        .catch(() => {});
    }
    if (evt.type === "chunk") {
      const text = evt.segments.map((s) => s.text.trim()).filter(Boolean).join(" ");
      if (text) setTranscript((prev) => [...prev.slice(-8), text]);
    }
  }, []);

  const handleStart = useCallback(async () => {
    setPhase("recording");
    setTranscript([]);
    setElapsed(0);
    startTimeRef.current = Date.now();

    timerRef.current = setInterval(() => {
      setElapsed(Date.now() - startTimeRef.current);
    }, 200);

    try {
      const sessionId = await startStreaming(
        { chunkMs: 2_000, silenceRmsThreshold: 0.01 },
        handleEvent,
      );
      sessionRef.current = sessionId;

      // Auto-stop after TEST_DURATION_MS
      setTimeout(async () => {
        const sid = sessionRef.current;
        if (sid) {
          await stopStreaming(sid).catch(() => {});
          const mid = meetingIdRef.current;
          if (mid) await deleteMeeting(mid).catch(() => {});
          sessionRef.current = null;
          meetingIdRef.current = null;
        }
        if (timerRef.current) clearInterval(timerRef.current);
        setPhase("done");
      }, TEST_DURATION_MS);
    } catch {
      if (timerRef.current) clearInterval(timerRef.current);
      setPhase("done");
    }
  }, [handleEvent]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      const sid = sessionRef.current;
      if (sid) {
        const mid = meetingIdRef.current;
        stopStreaming(sid)
          .then(() => { if (mid) return deleteMeeting(mid); })
          .catch(() => {});
      }
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const progress = Math.min(100, (elapsed / TEST_DURATION_MS) * 100);

  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-8 px-8">
      {/* Waveform icon */}
      <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-surface-sunken text-content-tertiary">
        <AudioLines className="h-8 w-8" />
      </div>

      <div className="flex flex-col items-center gap-3 text-center">
        <h2 className="text-display-md font-semibold tracking-tight text-content-primary">
          {t("onboarding.testTitle")}
        </h2>
        <p className="max-w-sm text-ui-sm text-content-secondary">
          {t("onboarding.testSubtitle")}
        </p>
      </div>

      {/* Recording area */}
      <div className="flex w-full max-w-md flex-col gap-4">
        {phase === "recording" && (
          <>
            {/* Progress bar */}
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-content-placeholder/20">
              <div
                className="h-full rounded-full bg-rose-500 transition-all duration-200"
                style={{ width: `${progress}%` }}
              />
            </div>
            <p className="text-center text-ui-xs text-content-tertiary">
              {Math.ceil((TEST_DURATION_MS - elapsed) / 1000)}s {t("onboarding.testRemaining")}
            </p>
          </>
        )}

        {/* Live transcript */}
        {transcript.length > 0 && (
          <div className="min-h-[80px] rounded-lg border border-subtle bg-surface-sunken p-3">
            {transcript.map((line, i) => (
              <p
                key={`${i}-${line.slice(0, 10)}`}
                className={`text-ui-sm text-content-secondary ${i === transcript.length - 1 ? "animate-text-appear opacity-0" : ""}`}
              >
                {line}
              </p>
            ))}
          </div>
        )}

        {phase === "done" && transcript.length === 0 && (
          <p className="text-center text-ui-sm text-content-tertiary">
            {t("onboarding.testNoSpeech")}
          </p>
        )}
      </div>

      {/* Actions */}
      <div className="flex items-center gap-4">
        {phase === "idle" && (
          <>
            <button
              type="button"
              onClick={handleStart}
              className="group relative flex items-center gap-2 rounded-full bg-rose-600 px-6 py-2.5 text-ui-md font-medium text-white shadow-sm transition-all hover:bg-rose-500 hover:shadow-md active:scale-[0.98]"
            >
              <span className="absolute inset-0 rounded-full bg-rose-500 opacity-0 group-hover:opacity-0 animate-rec-breathe" />
              <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-white/90" />
              <span className="relative">{t("onboarding.testStart")}</span>
            </button>
            <button
              type="button"
              onClick={onNext}
              className="text-ui-sm text-content-tertiary transition-colors hover:text-content-secondary"
            >
              {t("onboarding.testSkip")}
            </button>
          </>
        )}

        {phase === "recording" && (
          <div className="flex items-center gap-2 text-ui-sm font-medium text-rose-600 dark:text-rose-400">
            <span className="relative flex h-2.5 w-2.5">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-rose-500 opacity-75" />
              <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-rose-500" />
            </span>
            {t("onboarding.testRecording")}
          </div>
        )}

        {phase === "done" && (
          <button
            type="button"
            onClick={onNext}
            className="rounded-full bg-accent-600 px-8 py-2.5 text-ui-md font-medium text-white shadow-sm transition-all hover:bg-accent-700 hover:shadow-md active:scale-[0.98]"
          >
            {t("onboarding.continue")}
          </button>
        )}
      </div>
    </div>
  );
}
