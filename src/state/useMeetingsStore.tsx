/**
 * `MeetingsProvider` / `useMeetings` — the meetings + view + search store.
 *
 * Consolidates everything that used to be scattered across App.tsx as
 * loose `useState` calls (meetings list + load error, current view,
 * search input/query/hits/loading/error) into a single context. The
 * sidebar and main pane consume from here so App.tsx becomes a thin
 * shell that only owns global cross-cutting concerns (the recording
 * session, user prefs, layout).
 *
 * Internally the provider:
 *
 *   - holds the meetings list and `meetingsError` from the on-mount
 *     `listMeetings` refresh
 *   - holds the right-pane `view` discriminated union (`live` or a
 *     specific `meeting`) — the discriminated union doubles as the
 *     ad-hoc router
 *   - debounces the search input and runs the FTS5 query, with
 *     out-of-order response protection via a cancellation flag
 *   - composes `useMeetingDetail` so `goToMeeting`, `deleteMeeting`,
 *     and `renameSpeaker` are exposed as ready-to-use actions
 *
 * What the provider does NOT own:
 *
 *   - the recording state machine (lives in `useRecordingSession`,
 *     which must sit ABOVE the view switch so the live transcript
 *     survives navigating to a stored meeting and back)
 *   - the toast API (already provided by `<ToastProvider />`)
 *   - the backend health probe (`useHealthProbe`)
 *
 * Composition order in `main.tsx`:
 *
 *   <ToastProvider>
 *     <MeetingsProvider>
 *       <App />
 *     </MeetingsProvider>
 *   </ToastProvider>
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { useTranslation } from "react-i18next";

import { useMeetingDetail } from "../hooks/useMeetingDetail";
import { listMeetings, searchMeetings } from "../ipc/client";
import { isTauri } from "../ipc/isTauri";
import { useIpcAction } from "../ipc/useIpcAction";
import { useDebouncedValue } from "../lib/useDebouncedValue";
import type {
  MeetingId,
  MeetingSearchHit,
  MeetingSummary,
} from "../types/meeting";
import type { SpeakerId } from "../types/speaker";
import type { MainView } from "../types/view";

// ---------------------------------------------------------------------------
// Context shape
// ---------------------------------------------------------------------------

export type MeetingsContextValue = {
  // Listing
  meetings: MeetingSummary[];
  meetingsError: string | null;
  refreshMeetings: () => Promise<void>;
  // View / ad-hoc routing
  view: MainView;
  goToLive: () => void;
  goToMeeting: (id: MeetingId) => Promise<void>;
  deleteMeeting: (id: MeetingId) => Promise<void>;
  renameSpeaker: (
    speakerId: SpeakerId,
    label: string | null,
  ) => Promise<void>;
  renameMeeting: (title: string) => Promise<void>;
  // Search
  search: {
    input: string;
    query: string;
    hits: MeetingSearchHit[];
    loading: boolean;
    error: string | null;
    setInput: (next: string) => void;
  };
};

const MeetingsContext = createContext<MeetingsContextValue | null>(null);

/** Read-only handle to the meetings store. Throws when used outside a provider. */
export function useMeetings(): MeetingsContextValue {
  const ctx = useContext(MeetingsContext);
  if (!ctx) {
    throw new Error("useMeetings must be used inside <MeetingsProvider>");
  }
  return ctx;
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

const SEARCH_DEBOUNCE_MS = 200;

export function MeetingsProvider({ children }: { children: ReactNode }) {
  const { t } = useTranslation();
  // Listing -----------------------------------------------------------------
  const [meetings, setMeetings] = useState<MeetingSummary[]>([]);
  const [meetingsError, setMeetingsError] = useState<string | null>(null);

  const refreshList = useIpcAction(
    t("toast.refreshFailed"),
    listMeetings,
  );
  const refreshMeetings = useCallback(async () => {
    if (!isTauri()) return;
    const rows = await refreshList();
    if (rows === undefined) {
      // useIpcAction already toasted; mirror the message inline so it
      // stays visible after the toast auto-dismisses.
      setMeetingsError(t("toast.refreshFailed"));
      return;
    }
    setMeetings(rows);
    setMeetingsError(null);
  }, [refreshList, t]);

  // View --------------------------------------------------------------------
  const [view, setView] = useState<MainView>({ kind: "live" });
  const goToLive = useCallback(() => setView({ kind: "live" }), []);

  // Meeting actions (open / rename / delete) compose useMeetingDetail.
  const { openMeeting, renameMeetingAction, renameSpeakerAction, deleteMeetingAction } =
    useMeetingDetail({
      view,
      setView,
      refreshMeetings,
      setMeetingsError,
    });

  // Search ------------------------------------------------------------------
  const [searchInput, setSearchInput] = useState("");
  const searchQuery = useDebouncedValue(searchInput, SEARCH_DEBOUNCE_MS);
  const [searchHits, setSearchHits] = useState<MeetingSearchHit[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);

  // Run the FTS5 query whenever the debounced input changes. The
  // `cancelled` flag protects against out-of-order responses (a slow
  // request returning *after* a faster one for a newer query, which
  // would otherwise overwrite the fresh hits with stale ones).
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

  // On-mount refresh --------------------------------------------------------
  useEffect(() => {
    void refreshMeetings();
  }, [refreshMeetings]);

  // Public value ------------------------------------------------------------
  const value = useMemo<MeetingsContextValue>(
    () => ({
      meetings,
      meetingsError,
      refreshMeetings,
      view,
      goToLive,
      goToMeeting: openMeeting,
      deleteMeeting: deleteMeetingAction,
      renameSpeaker: renameSpeakerAction,
      renameMeeting: renameMeetingAction,
      search: {
        input: searchInput,
        query: searchQuery,
        hits: searchHits,
        loading: searchLoading,
        error: searchError,
        setInput: setSearchInput,
      },
    }),
    [
      meetings,
      meetingsError,
      refreshMeetings,
      view,
      goToLive,
      openMeeting,
      deleteMeetingAction,
      renameSpeakerAction,
      renameMeetingAction,
      searchInput,
      searchQuery,
      searchHits,
      searchLoading,
      searchError,
    ],
  );

  return (
    <MeetingsContext.Provider value={value}>
      {children}
    </MeetingsContext.Provider>
  );
}
