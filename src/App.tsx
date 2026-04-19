/**
 * `<App />` — the application shell.
 *
 * Composes the cross-cutting hooks (health probe, recording session)
 * with the meetings store (provided in `main.tsx`) and lays out the
 * three panes: header, sidebar rail, main pane.
 *
 * The recording session lives here, not inside `<LivePane />`,
 * because it must survive navigating to a stored meeting and back —
 * unmounting the live pane would otherwise drop the in-flight
 * transcript. The user-pref toggles (language, diarize) live here
 * for the same reason.
 */

import { useCallback, useState } from "react";

import { HealthProbe } from "./features/live/HealthProbe";
import { LivePane } from "./features/live/LivePane";
import { MeetingDetail } from "./features/meetings/MeetingDetail";
import { Sidebar } from "./features/sidebar/Sidebar";
import { useHealthProbe } from "./hooks/useHealthProbe";
import { useRecordingSession } from "./hooks/useRecordingSession";
import { useMeetings } from "./state/useMeetingsStore";

export function App() {
  const probe = useHealthProbe();
  const { goToLive, refreshMeetings, view, renameSpeaker } = useMeetings();
  // Pull the primitives we depend on out of the hook return so our
  // useCallback deps below reference stable identities (each member
  // is memoised inside the hook with `useCallback`); depending on
  // the whole `recording` object would invalidate the callbacks on
  // every render and defeat the memoisation entirely.
  const recording = useRecordingSession({
    backendReady: probe.kind === "ok",
    onSessionFinished: refreshMeetings,
  });
  const {
    start: startRecording,
    reset: resetRecording,
    stop: stopRecording,
    dismissError,
  } = recording;

  // Diarize is opt-in to keep the existing whisper-only path unchanged
  // for users who haven't downloaded the embedder yet.
  const [diarize, setDiarize] = useState(false);
  // Language hint for whisper. `""` = "auto-detect"; defaults to "es"
  // because Spanish is the primary target language for this build.
  const [language, setLanguage] = useState<string>("es");

  // Pressing Start while viewing a stored meeting must also flip the
  // pane back to live so the user sees the new transcript.
  const handleStart = useCallback(async () => {
    goToLive();
    await startRecording({ language, diarize });
  }, [goToLive, startRecording, language, diarize]);

  // Switching back to the live pane after a session finished must
  // also clear stale lines and reset the state machine to idle —
  // otherwise the user sees a "✓ saved" status with the Start button
  // looking disabled even though it isn't.
  const handleGoLive = useCallback(() => {
    goToLive();
    resetRecording();
  }, [goToLive, resetRecording]);

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
        <Sidebar onGoLive={handleGoLive} />

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
              onStart={handleStart}
              onStop={stopRecording}
              onDismissError={dismissError}
            />
          ) : (
            <MeetingDetail view={view} onRenameSpeaker={renameSpeaker} />
          )}
        </section>
      </div>
    </main>
  );
}
