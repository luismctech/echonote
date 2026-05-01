/**
 * `<Sidebar />` — meetings rail container.
 *
 * Wires the `MeetingsContext` into the prop-driven leaf components
 * (`<MeetingsSearchBox />`, `<MeetingsList />`, `<SearchResults />`)
 * so the Shell does not have to thread a dozen props through to the
 * sidebar. The "+ Live" button needs to also reset the recording
 * session, which is owned above this container, so `onGoLive` is
 * passed down from the Shell instead of being read from context.
 */

import { useTranslation } from "react-i18next";

import { LogoMark } from "../../components/Logo";
import { useMeetings } from "../../state/useMeetingsStore";
import { MeetingsList } from "./MeetingsList";
import { MeetingsSearchBox } from "./MeetingsSearchBox";
import { SearchResults } from "./SearchResults";

export function Sidebar({
  onGoLive,
  isRecording = false,
}: Readonly<{
  onGoLive: () => void;
  isRecording?: boolean;
}>) {
  const { t } = useTranslation();
  const {
    meetings,
    meetingsError,
    view,
    goToMeeting,
    deleteMeeting,
    search,
  } = useMeetings();

  const isSearching = search.query.trim().length > 0;
  const activeId = view.kind === "meeting" ? view.id : null;

  return (
    <aside className="flex h-full min-h-0 flex-col gap-2 overflow-hidden rounded-lg border bg-surface-elevated p-3 shadow-sm">
      <header className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <LogoMark size={18} className="flex-shrink-0 opacity-60" />
          <h2 className="text-ui-md font-semibold tracking-wide text-content-secondary">
            {t("sidebar.meetings")}
          </h2>
        </div>
        <button
          type="button"
          onClick={onGoLive}
          className={`flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-ui-sm ${
            isRecording
              ? "border-rose-300 bg-rose-50 text-rose-700 dark:border-rose-800 dark:bg-rose-950/40 dark:text-rose-300"
              : "border text-content-secondary hover:bg-surface-sunken"
          }`}
        >
          {isRecording && (
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-rose-400 opacity-75" />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-rose-500" />
            </span>
          )}
          {isRecording ? t("sidebar.recording") : t("sidebar.live")}
        </button>
      </header>

      {meetingsError && (
        <p className="rounded bg-amber-50 px-2 py-1 text-ui-sm text-amber-800 dark:bg-amber-950/40 dark:text-amber-300">
          {meetingsError}
        </p>
      )}

      <MeetingsSearchBox
        value={search.input}
        onChange={search.setInput}
        loading={search.loading}
      />

      <div className="min-h-0 flex-1 overflow-y-auto">
        {isSearching ? (
          <SearchResults
            query={search.query.trim()}
            hits={search.hits}
            loading={search.loading}
            error={search.error}
            activeId={activeId}
            onSelect={(m) => void goToMeeting(m.id)}
          />
        ) : (
          <MeetingsList
            meetings={meetings}
            activeId={activeId}
            onSelect={(m) => void goToMeeting(m.id)}
            onDelete={(m) => void deleteMeeting(m.id)}
          />
        )}
      </div>
    </aside>
  );
}
