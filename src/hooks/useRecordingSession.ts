/**
 * `useRecordingSession` — orchestrator for the live transcription pane.
 *
 * Owns everything the live pane needs to function:
 *
 *   - the recording state machine (`RecordingState` reducer)
 *   - the rolling list of transcript lines + session stats
 *   - the auto-scrolling list ref
 *   - the `handleEvent` translator from `TranscriptEvent` → state
 *   - `start` / `stop` / `dismissError` / `reset` actions
 *   - the dedup'd error/persisted → toast effect
 *   - `canStart` / `canStop` selectors (with backend-ready gating)
 *
 * Lives in `src/hooks/` (the application layer) so the UI stays
 * dumb: `<LivePane />` only receives values + callbacks, never IPC
 * primitives. This is the React analogue of an `echo-app` use-case
 * coordinator.
 *
 * `onSessionFinished` is invoked whenever the backend signals that a
 * meeting has been persisted (`stopped`) or aborted (`failed`). The
 * meetings store uses it to refresh the sidebar list. The callback is
 * captured in a ref so callers don't need to wrap it in `useCallback`.
 */

import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import { useToast } from "../components/Toaster";
import { startStreaming, stopStreaming, pauseStreaming, resumeStreaming, addNote as ipcAddNote, getMeetingId } from "../ipc/client";
import { isIpcError } from "../types/ipc-error";
import type { ErrorCode } from "../types/ipc-error";
import {
  canStart as selectCanStart,
  canStop as selectCanStop,
  canPause as selectCanPause,
  canResume as selectCanResume,
  initialRecordingState,
  recordingReducer,
} from "../state/recording";
import type { TranscriptEvent } from "../types/streaming";
import type { Note } from "../types/meeting";
import type { StreamLine } from "../types/view";

export type RecordingStats = {
  chunks: number;
  skipped: number;
  audioMs: number;
};

const ZERO_STATS: RecordingStats = { chunks: 0, skipped: 0, audioMs: 0 };

/** Max transcript lines kept in memory during a live session. */
const MAX_LIVE_LINES = 500;

export type StartOptions = {
  /** Whisper language hint. Empty string means "auto-detect". */
  language: string;
  /** Enable diarization for this session. */
  diarize: boolean;
};

export function useRecordingSession({
  backendReady,
  onSessionFinished,
  onOpenModels,
}: {
  backendReady: boolean;
  onSessionFinished: () => void;
  onOpenModels?: () => void;
}) {
  const { t } = useTranslation();
  const toast = useToast();
  const [stream, dispatch] = useReducer(
    recordingReducer,
    initialRecordingState,
  );
  const [lines, setLines] = useState<StreamLine[]>([]);
  const [stats, setStats] = useState<RecordingStats>(ZERO_STATS);
  const [notes, setNotes] = useState<Note[]>([]);
  const sessionStartedAtRef = useRef<number | null>(null);
  const meetingIdRef = useRef<string | null>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // Capture the latest callback in a ref so we don't force callers to
  // memoise it. The `handleEvent` callback below depends on this ref,
  // not the prop directly, so its identity is stable.
  const onSessionFinishedRef = useRef(onSessionFinished);
  useEffect(() => {
    onSessionFinishedRef.current = onSessionFinished;
  }, [onSessionFinished]);

  const onOpenModelsRef = useRef(onOpenModels);
  useEffect(() => {
    onOpenModelsRef.current = onOpenModels;
  }, [onOpenModels]);

  // Auto-scroll the live transcript list as new lines arrive.
  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [lines]);

  // Surface state-machine errors AND the success "saved" event as
  // toasts (single source of truth). The reducer is the only place
  // that decides terminal kinds; here we just translate them into
  // notifications. Both branches dedupe via a signature ref so the
  // same toast doesn't pop twice when the effect re-runs without an
  // actual state change — including React 18 StrictMode's
  // double-invoke in dev, which would otherwise yield two
  // "Meeting saved" toasts on every recording stop.
  const lastReportedRef = useRef<string | null>(null);
  useEffect(() => {
    if (stream.kind === "error") {
      const signature = `error|${stream.recoverable}|${stream.message}`;
      if (lastReportedRef.current === signature) return;
      lastReportedRef.current = signature;

      const isModelMissing = stream.errorCode === "modelNotReady";

      let message: string;
      if (isModelMissing) message = t("errors.modelMissing");
      else if (stream.recoverable) message = t("errors.streamingFailed");
      else message = t("errors.streamingStopFailed");

      toast.push({
        kind: "error",
        message,
        detail: isModelMissing
          ? t("errors.modelMissingDetail")
          : stream.message,
        durationMs: isModelMissing ? 0 : undefined,
        ...(isModelMissing && onOpenModelsRef.current
          ? {
              action: {
                label: t("errors.openModels"),
                onClick: () => onOpenModelsRef.current?.(),
              },
            }
          : {}),
      });
    } else if (stream.kind === "persisted") {
      const signature = `persisted|${stream.lastTotalSegments}|${stream.lastTotalAudioMs}`;
      if (lastReportedRef.current === signature) return;
      lastReportedRef.current = signature;
      toast.push({
        kind: "success",
        message: t("toast.meetingSaved"),
        detail: t("toast.meetingDetail", {
          segments: stream.lastTotalSegments,
          seconds: (stream.lastTotalAudioMs / 1000).toFixed(1),
        }),
      });
    } else if (stream.kind !== "starting" && stream.kind !== "stopping") {
      lastReportedRef.current = null;
    }
  }, [stream, toast, t]);

  const handleEvent = useCallback((evt: TranscriptEvent) => {
    switch (evt.type) {
      case "started":
        sessionStartedAtRef.current = Date.now();
        getMeetingId(evt.sessionId).then((id) => {
          meetingIdRef.current = id;
        }).catch(() => { /* will retry on addNote if needed */ });
        dispatch({
          type: "STREAMING_STARTED",
          sessionId: evt.sessionId,
          inputFormat: evt.inputFormat,
        });
        break;
      case "chunk": {
        setStats((s) => ({
          ...s,
          chunks: s.chunks + 1,
          audioMs:
            s.audioMs +
            (evt.segments.at(-1)?.endMs ?? evt.offsetMs) -
            evt.offsetMs,
        }));
        const text = evt.segments
          .map((s) => s.text.trim())
          .filter((t) => t.length > 0)
          .join(" ");
        setLines((prev) => {
          const next = [
            ...prev,
            {
              kind: "chunk" as const,
              key: `${evt.sessionId}-${evt.chunkIndex}`,
              chunkIndex: evt.chunkIndex,
              offsetMs: evt.offsetMs,
              text: text || t("live.noSpeech"),
              language: evt.language,
              rtf: evt.rtf,
              speakerSlot: evt.speakerSlot,
            },
          ];
          return next.length > MAX_LIVE_LINES
            ? next.slice(-MAX_LIVE_LINES)
            : next;
        });
        break;
      }
      case "skipped":
        setStats((s) => ({
          ...s,
          skipped: s.skipped + 1,
          audioMs: s.audioMs + evt.durationMs,
        }));
        setLines((prev) => {
          const next = [
            ...prev,
            {
              kind: "skipped" as const,
              key: `${evt.sessionId}-${evt.chunkIndex}`,
              chunkIndex: evt.chunkIndex,
              offsetMs: evt.offsetMs,
              durationMs: evt.durationMs,
              rms: evt.rms,
            },
          ];
          return next.length > MAX_LIVE_LINES
            ? next.slice(-MAX_LIVE_LINES)
            : next;
        });
        break;
      case "stopped":
        dispatch({
          type: "STREAMING_STOPPED",
          totalSegments: evt.totalSegments,
          totalAudioMs: evt.totalAudioMs,
        });
        // Pipeline finalized the meeting in the DB — let the caller
        // (meetings store) refresh its sidebar list.
        onSessionFinishedRef.current();
        break;
      case "failed":
        dispatch({ type: "STREAMING_FAILED", message: evt.message });
        onSessionFinishedRef.current();
        break;
      case "paused":
        dispatch({ type: "STREAMING_PAUSED" });
        break;
      case "resumed":
        dispatch({ type: "STREAMING_RESUMED" });
        break;
    }
  }, [t]);

  const start = useCallback(
    async ({ language, diarize }: StartOptions) => {
      setLines([]);
      setStats(ZERO_STATS);
      setNotes([]);
      sessionStartedAtRef.current = null;
      meetingIdRef.current = null;
      dispatch({ type: "START_REQUESTED" });
      try {
        const langHint = language.trim();
        await startStreaming(
          {
            chunkMs: 5_000,
            silenceRmsThreshold: 0.005,
            diarize,
            ...(langHint.length > 0 ? { language: langHint } : {}),
          },
          handleEvent,
        );
      } catch (err) {
        let msg: string;
        let code: ErrorCode | undefined;
        if (isIpcError(err)) { msg = err.message; code = err.code; }
        else if (err instanceof Error) msg = err.message;
        else msg = String(err);
        dispatch({ type: "BACKEND_ERROR", message: msg, ...(code != null ? { errorCode: code } : {}) });
      }
    },
    [handleEvent],
  );

  const stop = useCallback(async () => {
    if (stream.kind !== "recording" && stream.kind !== "paused") return;
    const id = stream.sessionId;
    dispatch({ type: "STOP_REQUESTED" });
    try {
      await stopStreaming(id);
    } catch (err) {
      let msg: string;
      let code: ErrorCode | undefined;
      if (isIpcError(err)) { msg = err.message; code = err.code; }
      else if (err instanceof Error) msg = err.message;
      else msg = String(err);
      dispatch({ type: "BACKEND_ERROR", message: msg, ...(code != null ? { errorCode: code } : {}) });
    }
  }, [stream]);

  const pause = useCallback(async () => {
    if (stream.kind !== "recording") return;
    const id = stream.sessionId;
    dispatch({ type: "PAUSE_REQUESTED" });
    try {
      await pauseStreaming(id);
    } catch (err) {
      let msg: string;
      let code: ErrorCode | undefined;
      if (isIpcError(err)) { msg = err.message; code = err.code; }
      else if (err instanceof Error) msg = err.message;
      else msg = String(err);
      dispatch({ type: "BACKEND_ERROR", message: msg, ...(code != null ? { errorCode: code } : {}) });
    }
  }, [stream]);

  const resume = useCallback(async () => {
    if (stream.kind !== "paused") return;
    const id = stream.sessionId;
    dispatch({ type: "RESUME_REQUESTED" });
    try {
      await resumeStreaming(id);
    } catch (err) {
      let msg: string;
      let code: ErrorCode | undefined;
      if (isIpcError(err)) { msg = err.message; code = err.code; }
      else if (err instanceof Error) msg = err.message;
      else msg = String(err);
      dispatch({ type: "BACKEND_ERROR", message: msg, ...(code != null ? { errorCode: code } : {}) });
    }
  }, [stream]);

  const dismissError = useCallback(
    () => dispatch({ type: "ACKNOWLEDGE" }),
    [],
  );

  /**
   * Clear the visible transcript and reset the state machine so the
   * Start button is enabled again. Called when the user navigates
   * back to the live pane after a session finished — without this,
   * stale lines + a "✓ saved" status would imply the recording is
   * still in flight even though it isn't.
   */
  const reset = useCallback(() => {
    if (stream.kind === "persisted" || stream.kind === "error") {
      dispatch({ type: "ACKNOWLEDGE" });
      setLines([]);
      setStats(ZERO_STATS);
      setNotes([]);
      sessionStartedAtRef.current = null;
      meetingIdRef.current = null;
    }
  }, [stream.kind]);

  const addNote = useCallback(
    async (text: string) => {
      if (
        (stream.kind !== "recording" && stream.kind !== "paused") ||
        !sessionStartedAtRef.current
      ) {
        console.warn("[addNote] guard failed:", stream.kind, sessionStartedAtRef.current);
        return;
      }
      const timestampMs = Math.floor(Date.now() - sessionStartedAtRef.current);
      // Resolve the real database MeetingId (distinct from the streaming session id).
      let meetingId = meetingIdRef.current;
      if (!meetingId) {
        meetingId = await getMeetingId(stream.sessionId).catch(() => null);
        if (meetingId) meetingIdRef.current = meetingId;
      }
      if (!meetingId) {
        console.warn("[addNote] meeting id not yet available");
        return;
      }
      try {
        const note = await ipcAddNote(meetingId, text, timestampMs);
        setNotes((prev) => [...prev, note]);
      } catch (err) {
        console.error("[addNote] IPC failed:", err);
      }
    },
    [stream],
  );

  const canStart = backendReady && selectCanStart(stream);
  const canStop = selectCanStop(stream);
  const canPause = selectCanPause(stream);
  const canResume = selectCanResume(stream);

  return {
    stream,
    lines,
    stats,
    notes,
    listRef,
    canStart,
    canStop,
    canPause,
    canResume,
    start,
    stop,
    pause,
    resume,
    addNote,
    dismissError,
    reset,
  };
}
