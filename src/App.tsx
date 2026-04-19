import { useCallback, useEffect, useState } from "react";

import { HealthProbe } from "./features/live/HealthProbe";
import { LivePane } from "./features/live/LivePane";
import { MeetingDetail } from "./features/meetings/MeetingDetail";
import { MeetingsList } from "./features/sidebar/MeetingsList";
import { MeetingsSearchBox } from "./features/sidebar/MeetingsSearchBox";
import { SearchResults } from "./features/sidebar/SearchResults";
import { useHealthProbe } from "./hooks/useHealthProbe";
import { useMeetingDetail } from "./hooks/useMeetingDetail";
import { useRecordingSession } from "./hooks/useRecordingSession";
import { listMeetings, searchMeetings } from "./ipc/client";
import { isTauri } from "./ipc/isTauri";
import { useIpcAction } from "./ipc/useIpcAction";
import { useDebouncedValue } from "./lib/useDebouncedValue";
import type {
  MeetingSearchHit,
  MeetingSummary,
} from "./types/meeting";
import type { MainView } from "./types/view";

export function App() {
  const probe = useHealthProbe();

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
  // user can switch on the fly.
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

  // Wraps `listMeetings` so any failure pushes a warning toast and
  // surfaces inline in the sidebar via `meetingsError`.
  const refreshList = useIpcAction(
    "Couldn't refresh meetings list.",
    listMeetings,
  );
  const refreshMeetings = useCallback(async () => {
    if (!isTauri()) return;
    const rows = await refreshList();
    if (rows === undefined) {
      // useIpcAction already toasted; mirror the message inline so it
      // stays visible after the toast auto-dismisses.
      setMeetingsError("Couldn't refresh meetings list.");
      return;
    }
    setMeetings(rows);
    setMeetingsError(null);
  }, [refreshList]);

  // Run the FTS5 query whenever the debounced input changes. The
  // `cancelled` flag protects against out-of-order responses.
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

  // Refresh the sidebar once on mount, after the probe is wired up.
  useEffect(() => {
    void refreshMeetings();
  }, [refreshMeetings]);

  const recording = useRecordingSession({
    backendReady: probe.kind === "ok",
    onSessionFinished: refreshMeetings,
  });

  const { openMeeting, renameSpeakerAction, deleteMeetingAction } =
    useMeetingDetail({
      view,
      setView,
      refreshMeetings,
      setMeetingsError,
    });

  const onStart = useCallback(async () => {
    setView({ kind: "live" });
    await recording.start({ language, diarize });
  }, [recording, language, diarize]);

  // Switching to the live pane after a session finished should also
  // clear the previous transcript and reset the state machine to
  // idle, otherwise the user sees stale lines + a "✓ saved" status
  // and wonders why the Start button is "disabled" (it isn't, but
  // the visual context implies the recording is still in flight).
  const goLive = useCallback(() => {
    setView({ kind: "live" });
    recording.reset();
  }, [recording]);

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
                onDelete={(m) => void deleteMeetingAction(m.id)}
              />
            )}
          </div>
        </aside>

        <section className="flex min-h-0 min-w-0 flex-col gap-3 overflow-hidden rounded-lg border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-950">
          {view.kind === "live" ? (
            <LivePane
              stream={recording.stream}
              stats={recording.stats}
              lines={recording.lines}
              listRef={recording.listRef}
              canStart={recording.canStart}
              canStop={recording.canStop}
              diarize={diarize}
              onToggleDiarize={setDiarize}
              language={language}
              onChangeLanguage={setLanguage}
              onStart={onStart}
              onStop={recording.stop}
              onDismissError={recording.dismissError}
            />
          ) : (
            <MeetingDetail view={view} onRenameSpeaker={renameSpeakerAction} />
          )}
        </section>
      </div>
    </main>
  );
}
