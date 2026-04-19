import {
  useCallback,
  useEffect,
  useReducer,
  useRef,
  useState,
} from "react";

import { useToast } from "./components/Toaster";
import { HealthProbe } from "./features/live/HealthProbe";
import { LivePane } from "./features/live/LivePane";
import { MeetingDetail } from "./features/meetings/MeetingDetail";
import { MeetingsList } from "./features/sidebar/MeetingsList";
import { MeetingsSearchBox } from "./features/sidebar/MeetingsSearchBox";
import { SearchResults } from "./features/sidebar/SearchResults";
import {
  deleteMeeting,
  getMeeting,
  healthCheck,
  isTauri,
  listMeetings,
  renameSpeaker,
  searchMeetings,
  startStreaming,
  stopStreaming,
} from "./lib/ipc";
import { useDebouncedValue } from "./lib/useDebouncedValue";
import {
  canStart as selectCanStart,
  canStop as selectCanStop,
  initialRecordingState,
  recordingReducer,
} from "./state/recording";
import type {
  MeetingId,
  MeetingSearchHit,
  MeetingSummary,
} from "./types/meeting";
import type { SpeakerId } from "./types/speaker";
import type { TranscriptEvent } from "./types/streaming";
import type { MainView, Probe, StreamLine } from "./types/view";

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
  // Language hint passed to whisper. `""` means "let the model auto-detect"
  // and is mapped to `undefined` in the IPC payload. We default to Spanish
  // because that's the primary target language for this build, but the
  // user can switch on the fly. The `.en`-only model will report any
  // non-English audio as "(speaking in foreign language)" — that's a
  // model-capability issue, not a UI bug; surfacing the picker makes the
  // dependency on a multilingual model obvious.
  const [language, setLanguage] = useState<string>("es");

  // Sidebar search (Sprint 1 day 8). `searchInput` mirrors the text
  // box character-by-character; `searchQuery` is the debounced value
  // we actually send to the backend so we don't fire an FTS query on
  // every keystroke. 200 ms is short enough to feel instant and long
  // enough to skip the long tail of wasted requests during fast typing.
  const [searchInput, setSearchInput] = useState("");
  const searchQuery = useDebouncedValue(searchInput, 200);
  const [searchHits, setSearchHits] = useState<MeetingSearchHit[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const isSearching = searchQuery.trim().length > 0;

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

  // Run the FTS5 query whenever the debounced input changes. The
  // `cancelled` flag protects against out-of-order responses (the user
  // can type fast enough that a slow request returns *after* a faster
  // one for a newer query, which would otherwise overwrite the fresh
  // hits with stale ones).
  useEffect(() => {
    if (!isTauri()) return;
    const query = searchQuery.trim();
    if (query.length === 0) {
      setSearchHits([]);
      setSearchError(null);
      setSearchLoading(false);
      return;
    }
    let cancelled = false;
    setSearchLoading(true);
    setSearchError(null);
    searchMeetings(query)
      .then((hits) => {
        if (cancelled) return;
        setSearchHits(hits);
        setSearchLoading(false);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        const message = err instanceof Error ? err.message : String(err);
        setSearchError(message);
        setSearchHits([]);
        setSearchLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [searchQuery]);

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

  // Switching to the live pane after a session finished should also
  // clear the previous transcript and reset the state machine to
  // idle, otherwise the user sees stale lines + a "✓ saved" status
  // and wonders why the Start button is "disabled" (it isn't, but
  // the visual context implies the recording is still in flight).
  const goLive = useCallback(() => {
    setView({ kind: "live" });
    if (stream.kind === "persisted" || stream.kind === "error") {
      dispatch({ type: "ACKNOWLEDGE" });
      setLines([]);
      setStats({ chunks: 0, skipped: 0, audioMs: 0 });
    }
  }, [stream.kind]);

  return (
    <main className="flex h-full w-full flex-col gap-3 overflow-hidden px-4 py-3 sm:px-6 sm:py-4">
      <header className="flex flex-shrink-0 items-end justify-between gap-4">
        <div className="flex flex-col">
          <h1 className="text-xl font-semibold tracking-tight sm:text-2xl">
            EchoNote
          </h1>
          <p className="hidden text-xs text-zinc-500 dark:text-zinc-400 sm:block">
            Private, local-first meeting transcription and AI summaries.
          </p>
        </div>
        <HealthProbe probe={probe} />
      </header>

      <div className="grid min-h-0 flex-1 grid-cols-1 gap-4 md:grid-cols-[260px_1fr]">
        <aside className="flex min-h-0 flex-col gap-2 overflow-hidden rounded-lg border border-zinc-200 bg-white p-3 shadow-sm dark:border-zinc-800 dark:bg-zinc-950">
          <header className="flex items-center justify-between">
            <h2 className="text-sm font-semibold tracking-wide text-zinc-700 dark:text-zinc-200">
              Meetings
            </h2>
            <button
              type="button"
              onClick={goLive}
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
          <MeetingsSearchBox
            value={searchInput}
            onChange={setSearchInput}
            loading={searchLoading}
          />
          <div className="min-h-0 flex-1 overflow-y-auto">
            {isSearching ? (
              <SearchResults
                query={searchQuery.trim()}
                hits={searchHits}
                loading={searchLoading}
                error={searchError}
                activeId={view.kind === "meeting" ? view.id : null}
                onSelect={(m) => void openMeeting(m.id)}
              />
            ) : (
              <MeetingsList
                meetings={meetings}
                activeId={view.kind === "meeting" ? view.id : null}
                onSelect={(m) => void openMeeting(m.id)}
                onDelete={(m) => void onDeleteMeeting(m.id)}
              />
            )}
          </div>
        </aside>

        <section className="flex min-h-0 min-w-0 flex-col gap-3 overflow-hidden rounded-lg border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-950">
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
              language={language}
              onChangeLanguage={setLanguage}
              onStart={onStart}
              onStop={onStop}
              onDismissError={() => dispatch({ type: "ACKNOWLEDGE" })}
            />
          ) : (
            <MeetingDetail view={view} onRenameSpeaker={onRenameSpeaker} />
          )}
        </section>
      </div>
    </main>
  );
}
