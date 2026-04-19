/**
 * `useMeetingSummary` — load + generate the LLM summary for one meeting.
 *
 * Owns the small piece of view-state the `SummaryPanel` needs:
 *
 *   - `summary`     the most recent `Summary` for the meeting, or
 *                    `null` when none has been generated yet
 *   - `loading`     true while the initial `getSummary` is in flight
 *   - `generating`  true while `summarize_meeting` is running on the
 *                    backend (typically several seconds — the UI must
 *                    render a spinner instead of a frozen panel)
 *   - `error`       the latest *load* error, surfaced inline in the
 *                    panel; generation errors go straight to a toast
 *                    via `useIpcAction` (no need to mirror them here)
 *   - `generate()`  trigger a (re)generate; returns the new summary
 *                    on success, `undefined` on failure
 *   - `reset()`     drop local state, used when the active meeting
 *                    changes (the parent owns "which meeting is open")
 *
 * The hook treats meeting changes as full re-mounts: pass a different
 * `meetingId` and the loading/error/summary fields all re-initialise.
 * That matches how `MeetingDetail` is keyed, and keeps the state
 * machine inside the hook trivial.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { getSummary, summarizeMeeting } from "../ipc/client";
import { useIpcAction } from "../ipc/useIpcAction";
import type { MeetingId } from "../types/meeting";
import type { Summary } from "../types/summary";

export type UseMeetingSummary = {
  summary: Summary | null;
  loading: boolean;
  generating: boolean;
  error: string | null;
  generate: () => Promise<Summary | undefined>;
};

export function useMeetingSummary(meetingId: MeetingId | null): UseMeetingSummary {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Track the last meeting id we issued a `getSummary` for so a slow
  // response for the previous meeting can't overwrite the panel after
  // the user already navigated away. Same pattern `useMeetingDetail`
  // uses for `getMeeting`.
  const requestedRef = useRef<MeetingId | null>(null);

  useEffect(() => {
    requestedRef.current = meetingId;
    if (!meetingId) {
      setSummary(null);
      setLoading(false);
      setError(null);
      return;
    }
    setLoading(true);
    setError(null);
    setSummary(null);
    let cancelled = false;
    (async () => {
      try {
        const fetched = await getSummary(meetingId);
        if (cancelled || requestedRef.current !== meetingId) return;
        setSummary(fetched);
      } catch (err) {
        if (cancelled || requestedRef.current !== meetingId) return;
        const message = err instanceof Error ? err.message : String(err);
        setError(message);
      } finally {
        if (!cancelled && requestedRef.current === meetingId) {
          setLoading(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [meetingId]);

  // Generation goes through `useIpcAction` so failures auto-toast
  // with the same UX as every other IPC error. We layer our own
  // `generating` flag on top because the toast layer doesn't expose
  // a "request in flight" signal — the SummaryPanel needs it to
  // disable the button + render a spinner.
  const generateCall = useIpcAction(
    "Couldn't generate summary.",
    summarizeMeeting,
  );

  const generate = useCallback(async (): Promise<Summary | undefined> => {
    if (!meetingId) return undefined;
    setGenerating(true);
    try {
      const fresh = await generateCall(meetingId);
      // Only commit the result if the user hasn't navigated away
      // mid-generation. The use case still upserted to disk, so
      // re-opening the meeting will show it — but mutating the panel
      // for a different meeting would be a UX bug.
      if (fresh && requestedRef.current === meetingId) {
        setSummary(fresh);
      }
      return fresh;
    } finally {
      setGenerating(false);
    }
  }, [meetingId, generateCall]);

  return { summary, loading, generating, error, generate };
}
