import { useCallback, useEffect, useRef, useState } from "react";
import {
  deleteMeeting,
  getMeeting,
  healthCheck,
  isTauri,
  listMeetings,
  startStreaming,
  stopStreaming,
  type HealthStatus,
  type Meeting,
  type MeetingId,
  type MeetingSummary,
  type StreamingSessionId,
  type TranscriptEvent,
} from "./lib/ipc";

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
    }
  | {
      kind: "skipped";
      key: string;
      chunkIndex: number;
      offsetMs: number;
      durationMs: number;
      rms: number;
    };

type StreamState =
  | { kind: "idle" }
  | { kind: "starting" }
  | {
      kind: "running";
      sessionId: StreamingSessionId;
      inputFormat?: { sampleRateHz: number; channels: number };
    }
  | { kind: "stopping" }
  | { kind: "error"; message: string };

/** Right-pane mode: live transcription or replay of a stored meeting. */
type MainView =
  | { kind: "live" }
  | { kind: "meeting"; id: MeetingId; meeting: Meeting | null; loading: boolean; error?: string };

export function App() {
  const [probe, setProbe] = useState<Probe>({ kind: "idle" });
  const [stream, setStream] = useState<StreamState>({ kind: "idle" });
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

  const refreshMeetings = useCallback(async () => {
    if (!isTauri()) return;
    try {
      const rows = await listMeetings();
      setMeetings(rows);
      setMeetingsError(null);
    } catch (err) {
      setMeetingsError(err instanceof Error ? err.message : String(err));
    }
  }, []);

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
      .catch((err: unknown) =>
        setProbe({
          kind: "error",
          message: err instanceof Error ? err.message : String(err),
        }),
      );
    void refreshMeetings();
  }, [refreshMeetings]);

  // Auto-scroll the live transcript list as new lines arrive.
  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [lines]);

  const handleEvent = useCallback(
    (evt: TranscriptEvent) => {
      switch (evt.type) {
        case "started":
          setStream({
            kind: "running",
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
          setStats({
            chunks: 0,
            skipped: 0,
            audioMs: evt.totalAudioMs,
          });
          setStream({ kind: "idle" });
          // Pipeline finalized the meeting in the DB — refresh sidebar.
          void refreshMeetings();
          break;
        case "failed":
          setStream({ kind: "error", message: evt.message });
          void refreshMeetings();
          break;
      }
    },
    [refreshMeetings],
  );

  const onStart = async () => {
    setLines([]);
    setStats({ chunks: 0, skipped: 0, audioMs: 0 });
    setStream({ kind: "starting" });
    setView({ kind: "live" });
    try {
      await startStreaming(
        { chunkMs: 5_000, silenceRmsThreshold: 0.005 },
        handleEvent,
      );
    } catch (err) {
      setStream({
        kind: "error",
        message: err instanceof Error ? err.message : String(err),
      });
    }
  };

  const onStop = async () => {
    if (stream.kind !== "running") return;
    const id = stream.sessionId;
    setStream({ kind: "stopping" });
    try {
      await stopStreaming(id);
    } catch (err) {
      setStream({
        kind: "error",
        message: err instanceof Error ? err.message : String(err),
      });
    }
  };

  const openMeeting = useCallback(async (id: MeetingId) => {
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
      setView({
        kind: "meeting",
        id,
        meeting: null,
        loading: false,
        error: err instanceof Error ? err.message : String(err),
      });
    }
  }, []);

  const onDeleteMeeting = useCallback(
    async (id: MeetingId) => {
      try {
        await deleteMeeting(id);
        await refreshMeetings();
        if (view.kind === "meeting" && view.id === id) {
          setView({ kind: "live" });
        }
      } catch (err) {
        setMeetingsError(err instanceof Error ? err.message : String(err));
      }
    },
    [refreshMeetings, view],
  );

  const canStart =
    probe.kind === "ok" && (stream.kind === "idle" || stream.kind === "error");
  const canStop = stream.kind === "running";

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
              onStart={onStart}
              onStop={onStop}
            />
          ) : (
            <MeetingDetail view={view} />
          )}
        </section>
      </div>

      <footer className="text-xs text-zinc-400 dark:text-zinc-600">
        Sprint 0 · day 8 · streaming + persistence
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
  onStart,
  onStop,
}: {
  stream: StreamState;
  stats: { chunks: number; skipped: number; audioMs: number };
  lines: StreamLine[];
  listRef: React.RefObject<HTMLDivElement>;
  canStart: boolean;
  canStop: boolean;
  onStart: () => void;
  onStop: () => void;
}) {
  return (
    <>
      <header className="flex items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-medium">Live transcript</h2>
          <p className="text-xs text-zinc-500 dark:text-zinc-400">
            5-second windows · whisper.cpp · {modelLabel(stream)}
          </p>
        </div>
        <div className="flex gap-2">
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
        <p className="rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800 dark:bg-amber-950/40 dark:text-amber-300">
          <strong className="font-semibold">error:</strong> {stream.message}
        </p>
      )}

      <StatsBar stats={stats} stream={stream} />

      <div
        ref={listRef}
        className="h-[60vh] overflow-y-auto rounded-md border border-zinc-100 bg-zinc-50 p-3 font-mono text-xs leading-relaxed dark:border-zinc-900 dark:bg-zinc-900/60"
      >
        {lines.length === 0 ? (
          <p className="text-zinc-400">
            {stream.kind === "running"
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
}: {
  view: Extract<MainView, { kind: "meeting" }>;
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

      <div className="h-[60vh] overflow-y-auto rounded-md border border-zinc-100 bg-zinc-50 p-3 text-sm leading-relaxed dark:border-zinc-900 dark:bg-zinc-900/60">
        {m.segments.length === 0 ? (
          <p className="text-zinc-400">No segments persisted for this meeting.</p>
        ) : (
          <ol className="flex flex-col gap-2">
            {m.segments.map((seg) => (
              <li key={seg.id} className="flex gap-3">
                <span className="w-12 shrink-0 font-mono text-xs tabular-nums text-zinc-500">
                  {formatTimestamp(seg.startMs)}
                </span>
                <span className="flex-1">{seg.text.trim() || "[no speech]"}</span>
              </li>
            ))}
          </ol>
        )}
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------
// Misc helpers / small components
// ---------------------------------------------------------------------------

function modelLabel(stream: StreamState): string {
  if (stream.kind === "running" && stream.inputFormat) {
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
  stream: StreamState;
}) {
  const status =
    stream.kind === "running"
      ? "● recording"
      : stream.kind === "starting"
        ? "○ starting"
        : stream.kind === "stopping"
          ? "○ stopping"
          : "○ idle";
  return (
    <dl className="grid grid-cols-4 gap-x-4 text-xs">
      <Stat label="status" value={status} />
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
    <li className="flex gap-3">
      <span className="w-12 shrink-0 tabular-nums text-zinc-500">{ts}</span>
      <span className="flex-1">{line.text}</span>
      <span className="shrink-0 text-zinc-400">
        {line.language ?? "?"} · rtf {line.rtf.toFixed(2)}
      </span>
    </li>
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
