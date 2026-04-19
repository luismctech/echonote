import {
  useCallback,
  useEffect,
  useReducer,
  useRef,
  useState,
} from "react";
import { useToast } from "./components/Toaster";
import {
  deleteMeeting,
  getMeeting,
  healthCheck,
  isTauri,
  listMeetings,
  renameSpeaker,
  startStreaming,
  stopStreaming,
  type HealthStatus,
  type Meeting,
  type MeetingId,
  type MeetingSummary,
  type Speaker,
  type SpeakerId,
  type TranscriptEvent,
} from "./lib/ipc";
import {
  displayName,
  indexSpeakers,
  paletteFor,
  shortTag,
} from "./lib/speakers";
import {
  canStart as selectCanStart,
  canStop as selectCanStop,
  initialRecordingState,
  recordingReducer,
  statusLabel,
  type RecordingState,
} from "./state/recording";

type Probe =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "ok"; status: HealthStatus }
  | { kind: "error"; message: string };

type StreamLine =
  | {
      kind: "chunk";
      key: string;
      chunkIndex: number;
      offsetMs: number;
      text: string;
      language: string | null;
      rtf: number;
      /** 0-based slot of the diarized speaker, or undefined if diarization is off. */
      speakerSlot?: number | undefined;
    }
  | {
      kind: "skipped";
      key: string;
      chunkIndex: number;
      offsetMs: number;
      durationMs: number;
      rms: number;
    };

/** Right-pane mode: live transcription or replay of a stored meeting. */
type MainView =
  | { kind: "live" }
  | { kind: "meeting"; id: MeetingId; meeting: Meeting | null; loading: boolean; error?: string };

export function App() {
  const toast = useToast();

  const [probe, setProbe] = useState<Probe>({ kind: "idle" });
  const [stream, dispatch] = useReducer(
    recordingReducer,
    initialRecordingState,
  );
  const [lines, setLines] = useState<StreamLine[]>([]);
  const [stats, setStats] = useState<{
    chunks: number;
    skipped: number;
    audioMs: number;
  }>({ chunks: 0, skipped: 0, audioMs: 0 });
  const listRef = useRef<HTMLDivElement>(null);

  const [meetings, setMeetings] = useState<MeetingSummary[]>([]);
  const [meetingsError, setMeetingsError] = useState<string | null>(null);
  const [view, setView] = useState<MainView>({ kind: "live" });
  // Diarize is opt-in to keep the existing whisper-only path unchanged
  // for users who haven't downloaded the embedder yet. Persists across
  // session restarts within a tab; resets on reload.
  const [diarize, setDiarize] = useState(false);

  const refreshMeetings = useCallback(async () => {
    if (!isTauri()) return;
    try {
      const rows = await listMeetings();
      setMeetings(rows);
      setMeetingsError(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setMeetingsError(message);
      toast.push({
        kind: "warning",
        message: "Couldn't refresh meetings list.",
        detail: message,
      });
    }
  }, [toast]);

  useEffect(() => {
    if (!isTauri()) {
      setProbe({
        kind: "error",
        message:
          "Running outside Tauri — IPC is unavailable in `pnpm dev`. Use `pnpm tauri:dev`.",
      });
      return;
    }
    setProbe({ kind: "loading" });
    healthCheck()
      .then((status) => setProbe({ kind: "ok", status }))
      .catch((err: unknown) => {
        const message = err instanceof Error ? err.message : String(err);
        setProbe({ kind: "error", message });
        toast.push({
          kind: "error",
          message: "Backend health check failed.",
          detail: message,
        });
      });
    void refreshMeetings();
  }, [refreshMeetings, toast]);

  // Auto-scroll the live transcript list as new lines arrive.
  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [lines]);

  // Surface state-machine errors as toasts (single source of truth).
  // The reducer is the only place that decides "this is an error";
  // here we just translate that into a notification. Recoverable errors
  // get the standard error toast; non-recoverable ones get an extra hint.
  const lastReportedErrorRef = useRef<string | null>(null);
  useEffect(() => {
    if (stream.kind === "error") {
      const signature = `${stream.recoverable}|${stream.message}`;
      if (lastReportedErrorRef.current === signature) return;
      lastReportedErrorRef.current = signature;
      toast.push({
        kind: "error",
        message: stream.recoverable
          ? "Streaming failed — you can retry."
          : "Streaming failed mid-stop. Some audio may not have been persisted.",
        detail: stream.message,
      });
    } else if (stream.kind === "persisted") {
      lastReportedErrorRef.current = null;
      toast.push({
        kind: "success",
        message: "Meeting saved",
        detail: `${stream.lastTotalSegments} segments · ${(
          stream.lastTotalAudioMs / 1000
        ).toFixed(1)} s`,
      });
    } else if (stream.kind !== "starting" && stream.kind !== "stopping") {
      lastReportedErrorRef.current = null;
    }
  }, [stream, toast]);

  const handleEvent = useCallback(
    (evt: TranscriptEvent) => {
      switch (evt.type) {
        case "started":
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
          setLines((prev) => [
            ...prev,
            {
              kind: "chunk",
              key: `${evt.sessionId}-${evt.chunkIndex}`,
              chunkIndex: evt.chunkIndex,
              offsetMs: evt.offsetMs,
              text: text || "[no speech]",
              language: evt.language,
              rtf: evt.rtf,
              speakerSlot: evt.speakerSlot,
            },
          ]);
          break;
        }
        case "skipped":
          setStats((s) => ({
            ...s,
            skipped: s.skipped + 1,
            audioMs: s.audioMs + evt.durationMs,
          }));
          setLines((prev) => [
            ...prev,
            {
              kind: "skipped",
              key: `${evt.sessionId}-${evt.chunkIndex}`,
              chunkIndex: evt.chunkIndex,
              offsetMs: evt.offsetMs,
              durationMs: evt.durationMs,
              rms: evt.rms,
            },
          ]);
          break;
        case "stopped":
          dispatch({
            type: "STREAMING_STOPPED",
            totalSegments: evt.totalSegments,
            totalAudioMs: evt.totalAudioMs,
          });
          // Pipeline finalized the meeting in the DB — refresh sidebar.
          void refreshMeetings();
          break;
        case "failed":
          dispatch({ type: "STREAMING_FAILED", message: evt.message });
          void refreshMeetings();
          break;
      }
    },
    [refreshMeetings],
  );

  const onStart = async () => {
    setLines([]);
    setStats({ chunks: 0, skipped: 0, audioMs: 0 });
    setView({ kind: "live" });
    dispatch({ type: "START_REQUESTED" });
    try {
      await startStreaming(
        { chunkMs: 5_000, silenceRmsThreshold: 0.005, diarize },
        handleEvent,
      );
    } catch (err) {
      dispatch({
        type: "BACKEND_ERROR",
        message: err instanceof Error ? err.message : String(err),
      });
    }
  };

  const onStop = async () => {
    if (stream.kind !== "recording") return;
    const id = stream.sessionId;
    dispatch({ type: "STOP_REQUESTED" });
    try {
      await stopStreaming(id);
    } catch (err) {
      dispatch({
        type: "BACKEND_ERROR",
        message: err instanceof Error ? err.message : String(err),
      });
    }
  };

  const openMeeting = useCallback(
    async (id: MeetingId) => {
      setView({ kind: "meeting", id, meeting: null, loading: true });
      try {
        const meeting = await getMeeting(id);
        if (!meeting) {
          setView({
            kind: "meeting",
            id,
            meeting: null,
            loading: false,
            error: "Meeting not found",
          });
        } else {
          setView({ kind: "meeting", id, meeting, loading: false });
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setView({
          kind: "meeting",
          id,
          meeting: null,
          loading: false,
          error: message,
        });
        toast.push({
          kind: "error",
          message: "Couldn't load meeting.",
          detail: message,
        });
      }
    },
    [toast],
  );

  const onRenameSpeaker = useCallback(
    async (speakerId: SpeakerId, label: string | null) => {
      if (view.kind !== "meeting" || !view.meeting) return;
      const meetingId = view.id;
      try {
        const updated = await renameSpeaker(meetingId, speakerId, label);
        // Re-render from the canonical post-rename meeting returned by
        // the backend so we don't drift from disk on the optimistic path.
        setView((prev) =>
          prev.kind === "meeting" && prev.id === meetingId
            ? { kind: "meeting", id: meetingId, meeting: updated, loading: false }
            : prev,
        );
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        toast.push({
          kind: "error",
          message: "Couldn't rename speaker.",
          detail: message,
        });
      }
    },
    [view, toast],
  );

  const onDeleteMeeting = useCallback(
    async (id: MeetingId) => {
      try {
        await deleteMeeting(id);
        await refreshMeetings();
        if (view.kind === "meeting" && view.id === id) {
          setView({ kind: "live" });
        }
        toast.push({ kind: "info", message: "Meeting deleted" });
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setMeetingsError(message);
        toast.push({
          kind: "error",
          message: "Couldn't delete meeting.",
          detail: message,
        });
      }
    },
    [refreshMeetings, view, toast],
  );

  const canStart = probe.kind === "ok" && selectCanStart(stream);
  const canStop = selectCanStop(stream);

  return (
    <main className="mx-auto flex min-h-screen max-w-6xl flex-col gap-6 px-6 py-10">
      <header className="flex flex-col items-start gap-1">
        <h1 className="text-3xl font-semibold tracking-tight">EchoNote</h1>
        <p className="text-sm text-zinc-500 dark:text-zinc-400">
          Private, local-first meeting transcription and AI summaries.
        </p>
      </header>

      <section
        aria-label="Backend probe"
        aria-live="polite"
        className="rounded-lg border border-zinc-200 bg-zinc-50 p-3 font-mono text-xs leading-relaxed dark:border-zinc-800 dark:bg-zinc-900"
      >
        <HealthProbe probe={probe} />
      </section>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-[280px_1fr]">
        <aside className="flex flex-col gap-3 rounded-lg border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-950">
          <header className="flex items-center justify-between">
            <h2 className="text-sm font-semibold tracking-wide text-zinc-700 dark:text-zinc-200">
              Meetings
            </h2>
            <button
              type="button"
              onClick={() => setView({ kind: "live" })}
              className="rounded-md border border-zinc-200 px-2 py-0.5 text-xs text-zinc-600 hover:bg-zinc-50 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800"
            >
              + Live
            </button>
          </header>
          {meetingsError && (
            <p className="rounded bg-amber-50 px-2 py-1 text-xs text-amber-800 dark:bg-amber-950/40 dark:text-amber-300">
              {meetingsError}
            </p>
          )}
          <MeetingsList
            meetings={meetings}
            activeId={view.kind === "meeting" ? view.id : null}
            onSelect={(m) => void openMeeting(m.id)}
            onDelete={(m) => void onDeleteMeeting(m.id)}
          />
        </aside>

        <section className="flex flex-col gap-4 rounded-lg border border-zinc-200 bg-white p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-950">
          {view.kind === "live" ? (
            <LivePane
              stream={stream}
              stats={stats}
              lines={lines}
              listRef={listRef}
              canStart={canStart}
              canStop={canStop}
              diarize={diarize}
              onToggleDiarize={setDiarize}
              onStart={onStart}
              onStop={onStop}
              onDismissError={() => dispatch({ type: "ACKNOWLEDGE" })}
            />
          ) : (
            <MeetingDetail view={view} onRenameSpeaker={onRenameSpeaker} />
          )}
        </section>
      </div>

      <footer className="text-xs text-zinc-400 dark:text-zinc-600">
        Sprint 1 · day 1 · frontend hardening
      </footer>
    </main>
  );
}

// ---------------------------------------------------------------------------
// Sidebar
// ---------------------------------------------------------------------------

function MeetingsList({
  meetings,
  activeId,
  onSelect,
  onDelete,
}: {
  meetings: MeetingSummary[];
  activeId: MeetingId | null;
  onSelect: (m: MeetingSummary) => void;
  onDelete: (m: MeetingSummary) => void;
}) {
  if (meetings.length === 0) {
    return (
      <p className="text-xs text-zinc-400">
        No meetings yet. Press <strong>Start</strong> to record one.
      </p>
    );
  }
  return (
    <ul className="flex flex-col gap-1 overflow-y-auto" style={{ maxHeight: "60vh" }}>
      {meetings.map((m) => {
        const active = m.id === activeId;
        return (
          <li key={m.id}>
            <div
              className={`group flex items-start gap-2 rounded-md border px-2.5 py-2 text-xs ${
                active
                  ? "border-emerald-300 bg-emerald-50 dark:border-emerald-800 dark:bg-emerald-950/40"
                  : "border-transparent hover:bg-zinc-50 dark:hover:bg-zinc-900"
              }`}
            >
              <button
                type="button"
                onClick={() => onSelect(m)}
                className="flex flex-1 flex-col items-start gap-0.5 text-left"
              >
                <span className="line-clamp-1 font-medium text-zinc-800 dark:text-zinc-100">
                  {m.title}
                </span>
                <span className="text-[10px] tabular-nums text-zinc-500 dark:text-zinc-400">
                  {formatDate(m.startedAt)} · {formatDurationMs(m.durationMs)} ·{" "}
                  {m.segmentCount} seg
                </span>
              </button>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(m);
                }}
                aria-label={`Delete ${m.title}`}
                className="opacity-0 transition-opacity group-hover:opacity-100 text-zinc-400 hover:text-rose-500"
              >
                ×
              </button>
            </div>
          </li>
        );
      })}
    </ul>
  );
}

// ---------------------------------------------------------------------------
// Main pane: Live
// ---------------------------------------------------------------------------

function LivePane({
  stream,
  stats,
  lines,
  listRef,
  canStart,
  canStop,
  diarize,
  onToggleDiarize,
  onStart,
  onStop,
  onDismissError,
}: {
  stream: RecordingState;
  stats: { chunks: number; skipped: number; audioMs: number };
  lines: StreamLine[];
  listRef: React.RefObject<HTMLDivElement>;
  canStart: boolean;
  canStop: boolean;
  diarize: boolean;
  onToggleDiarize: (next: boolean) => void;
  onStart: () => void;
  onStop: () => void;
  onDismissError: () => void;
}) {
  // Toggle is locked once a session is in flight: changing the
  // diarize flag mid-recording would make half the chunks have
  // speakers and half not, which is confusing to render.
  const toggleLocked =
    stream.kind === "starting" ||
    stream.kind === "recording" ||
    stream.kind === "stopping";
  return (
    <>
      <header className="flex items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-medium">Live transcript</h2>
          <p className="text-xs text-zinc-500 dark:text-zinc-400">
            5-second windows · whisper.cpp · {modelLabel(stream)}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <label
            className={`flex select-none items-center gap-2 text-xs ${
              toggleLocked ? "opacity-60" : "cursor-pointer"
            }`}
          >
            <input
              type="checkbox"
              checked={diarize}
              disabled={toggleLocked}
              onChange={(e) => onToggleDiarize(e.target.checked)}
              className="h-3.5 w-3.5 accent-emerald-600"
            />
            <span className="text-zinc-600 dark:text-zinc-300">Diarize</span>
          </label>
          <button
            type="button"
            onClick={onStart}
            disabled={!canStart}
            className="rounded-md bg-emerald-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-emerald-500 disabled:cursor-not-allowed disabled:bg-zinc-300 dark:disabled:bg-zinc-700"
          >
            {stream.kind === "starting" ? "Starting…" : "Start"}
          </button>
          <button
            type="button"
            onClick={onStop}
            disabled={!canStop}
            className="rounded-md bg-rose-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-rose-500 disabled:cursor-not-allowed disabled:bg-zinc-300 dark:disabled:bg-zinc-700"
          >
            {stream.kind === "stopping" ? "Stopping…" : "Stop"}
          </button>
        </div>
      </header>

      {stream.kind === "error" && (
        <div className="flex items-start justify-between gap-3 rounded-md bg-rose-50 px-3 py-2 text-xs text-rose-900 dark:bg-rose-950/40 dark:text-rose-200">
          <p>
            <strong className="font-semibold">
              {stream.recoverable ? "error:" : "fatal:"}
            </strong>{" "}
            {stream.message}
          </p>
          <button
            type="button"
            onClick={onDismissError}
            className="text-xs underline opacity-80 hover:opacity-100"
          >
            dismiss
          </button>
        </div>
      )}

      <StatsBar stats={stats} stream={stream} />

      <div
        ref={listRef}
        className="h-[60vh] overflow-y-auto rounded-md border border-zinc-100 bg-zinc-50 p-3 font-mono text-xs leading-relaxed dark:border-zinc-900 dark:bg-zinc-900/60"
      >
        {lines.length === 0 ? (
          <p className="text-zinc-400">
            {stream.kind === "recording"
              ? "Listening… speak into the microphone."
              : "Press Start to begin a session."}
          </p>
        ) : (
          <ul className="flex flex-col gap-1">
            {lines.map((line) => (
              <TranscriptRow key={line.key} line={line} />
            ))}
          </ul>
        )}
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------
// Main pane: Detail of a stored meeting
// ---------------------------------------------------------------------------

function MeetingDetail({
  view,
  onRenameSpeaker,
}: {
  view: Extract<MainView, { kind: "meeting" }>;
  onRenameSpeaker: (speakerId: SpeakerId, label: string | null) => Promise<void>;
}) {
  if (view.loading) {
    return <p className="text-sm text-zinc-500">Loading meeting…</p>;
  }
  if (view.error || !view.meeting) {
    return (
      <p className="text-sm text-amber-700 dark:text-amber-400">
        {view.error ?? "Meeting unavailable."}
      </p>
    );
  }
  const m = view.meeting;
  const speakerIndex = indexSpeakers(m.speakers);
  return (
    <>
      <header className="flex flex-col gap-1">
        <h2 className="text-lg font-medium">{m.title}</h2>
        <p className="text-xs text-zinc-500 dark:text-zinc-400">
          {formatDate(m.startedAt)} · {formatDurationMs(m.durationMs)} ·{" "}
          {m.language ?? "?"} · {m.segmentCount} segments
        </p>
        <p className="font-mono text-[10px] text-zinc-400">{m.id}</p>
      </header>

      {m.speakers.length > 0 && (
        <SpeakersPanel speakers={m.speakers} onRename={onRenameSpeaker} />
      )}

      <div className="h-[60vh] overflow-y-auto rounded-md border border-zinc-100 bg-zinc-50 p-3 text-sm leading-relaxed dark:border-zinc-900 dark:bg-zinc-900/60">
        {m.segments.length === 0 ? (
          <p className="text-zinc-400">No segments persisted for this meeting.</p>
        ) : (
          <ol className="flex flex-col gap-2">
            {m.segments.map((seg) => {
              const speaker = seg.speakerId
                ? speakerIndex.get(seg.speakerId)
                : undefined;
              return (
                <li key={seg.id} className="flex items-baseline gap-3">
                  <span className="w-12 shrink-0 font-mono text-xs tabular-nums text-zinc-500">
                    {formatTimestamp(seg.startMs)}
                  </span>
                  {speaker && (
                    <SpeakerChip
                      slot={speaker.slot}
                      label={shortTag(speaker.slot)}
                      compact
                    />
                  )}
                  <span className="flex-1">
                    {seg.text.trim() || "[no speech]"}
                  </span>
                </li>
              );
            })}
          </ol>
        )}
      </div>
    </>
  );
}

/**
 * List of diarized speakers with an inline rename input. Saves on
 * Enter or blur; clearing the input reverts to anonymous.
 */
function SpeakersPanel({
  speakers,
  onRename,
}: {
  speakers: Speaker[];
  onRename: (speakerId: SpeakerId, label: string | null) => Promise<void>;
}) {
  return (
    <section
      aria-label="Speakers"
      className="flex flex-wrap gap-2 rounded-md border border-zinc-100 bg-zinc-50 p-2 dark:border-zinc-900 dark:bg-zinc-900/40"
    >
      {speakers.map((sp) => (
        <SpeakerEditor key={sp.id} speaker={sp} onRename={onRename} />
      ))}
    </section>
  );
}

function SpeakerEditor({
  speaker,
  onRename,
}: {
  speaker: Speaker;
  onRename: (speakerId: SpeakerId, label: string | null) => Promise<void>;
}) {
  // Local input state so the user can keep typing without every
  // keystroke triggering an IPC round-trip. We commit on Enter or
  // blur; the canonical state still flows from props (the post-rename
  // meeting) so an external refresh would override stale local input.
  const [draft, setDraft] = useState(speaker.label ?? "");
  // Re-sync local draft whenever the upstream speaker label changes
  // (e.g. after renameSpeaker resolves with the canonical row), so
  // the input does not show stale text after a successful save.
  useEffect(() => {
    setDraft(speaker.label ?? "");
  }, [speaker.label]);

  const palette = paletteFor(speaker.slot);
  const placeholder = `Speaker ${speaker.slot + 1}`;
  const commit = () => {
    const next = draft.trim();
    const current = speaker.label ?? "";
    if (next === current) return;
    void onRename(speaker.id, next.length > 0 ? next : null);
  };
  return (
    <div
      className={`flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs ring-1 ring-inset ${palette.bg} ${palette.text} ${palette.ring}`}
    >
      <span className="font-semibold tabular-nums">{shortTag(speaker.slot)}</span>
      <input
        type="text"
        value={draft}
        placeholder={placeholder}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.currentTarget.blur();
          } else if (e.key === "Escape") {
            setDraft(speaker.label ?? "");
            e.currentTarget.blur();
          }
        }}
        aria-label={`Rename ${displayName(speaker)}`}
        className="w-28 bg-transparent outline-none placeholder:text-current placeholder:opacity-60"
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Misc helpers / small components
// ---------------------------------------------------------------------------

function modelLabel(stream: RecordingState): string {
  if (stream.kind === "recording" && stream.inputFormat) {
    const { sampleRateHz, channels } = stream.inputFormat;
    return `${sampleRateHz} Hz · ${channels} ch`;
  }
  return "model loads on first start";
}

function HealthProbe({ probe }: { probe: Probe }) {
  switch (probe.kind) {
    case "idle":
      return <p className="text-zinc-500">Warming up…</p>;
    case "loading":
      return <p className="text-zinc-500">Calling backend health_check…</p>;
    case "error":
      return (
        <p className="text-amber-700 dark:text-amber-400">
          <span className="font-semibold">offline:</span> {probe.message}
        </p>
      );
    case "ok":
      return (
        <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-0.5">
          <dt className="text-zinc-500">backend</dt>
          <dd className="text-emerald-700 dark:text-emerald-400">ok</dd>
          <dt className="text-zinc-500">version</dt>
          <dd>{probe.status.version}</dd>
          <dt className="text-zinc-500">target</dt>
          <dd>{probe.status.target}</dd>
          <dt className="text-zinc-500">commit</dt>
          <dd>{probe.status.commit}</dd>
        </dl>
      );
  }
}

function StatsBar({
  stats,
  stream,
}: {
  stats: { chunks: number; skipped: number; audioMs: number };
  stream: RecordingState;
}) {
  return (
    <dl className="grid grid-cols-4 gap-x-4 text-xs">
      <Stat label="status" value={statusLabel(stream)} />
      <Stat label="chunks" value={String(stats.chunks)} />
      <Stat label="skipped" value={String(stats.skipped)} />
      <Stat label="audio" value={`${(stats.audioMs / 1000).toFixed(1)} s`} />
    </dl>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col">
      <dt className="text-zinc-500">{label}</dt>
      <dd className="font-mono">{value}</dd>
    </div>
  );
}

function TranscriptRow({ line }: { line: StreamLine }) {
  const ts = formatTimestamp(line.offsetMs);
  if (line.kind === "skipped") {
    return (
      <li className="flex gap-3 text-zinc-400">
        <span className="w-12 shrink-0 tabular-nums">{ts}</span>
        <span className="italic">silence (rms={line.rms.toFixed(4)})</span>
      </li>
    );
  }
  return (
    <li className="flex items-baseline gap-3">
      <span className="w-12 shrink-0 tabular-nums text-zinc-500">{ts}</span>
      {line.speakerSlot !== undefined && (
        <SpeakerChip slot={line.speakerSlot} label={shortTag(line.speakerSlot)} compact />
      )}
      <span className="flex-1">{line.text}</span>
      <span className="shrink-0 text-zinc-400">
        {line.language ?? "?"} · rtf {line.rtf.toFixed(2)}
      </span>
    </li>
  );
}

/**
 * Coloured pill identifying a speaker. `compact` halves the padding
 * for dense rows (live transcript); the meeting-detail view uses the
 * full size in the speakers list.
 */
function SpeakerChip({
  slot,
  label,
  compact,
}: {
  slot: number;
  label: string;
  compact?: boolean;
}) {
  const palette = paletteFor(slot);
  const sizing = compact ? "px-1.5 py-0 text-[10px]" : "px-2 py-0.5 text-xs";
  return (
    <span
      className={`inline-flex shrink-0 items-center rounded-full font-medium tabular-nums ring-1 ring-inset ${palette.bg} ${palette.text} ${palette.ring} ${sizing}`}
      title={`Speaker slot ${slot + 1}`}
    >
      {label}
    </span>
  );
}

function formatTimestamp(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const m = Math.floor(totalSeconds / 60);
  const s = totalSeconds % 60;
  return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
}

function formatDate(rfc3339: string): string {
  const d = new Date(rfc3339);
  if (Number.isNaN(d.getTime())) return rfc3339;
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatDurationMs(ms: number): string {
  const totalSeconds = Math.round(ms / 1000);
  if (totalSeconds < 60) return `${totalSeconds}s`;
  const m = Math.floor(totalSeconds / 60);
  const s = totalSeconds % 60;
  return `${m}m ${s.toString().padStart(2, "0")}s`;
}
