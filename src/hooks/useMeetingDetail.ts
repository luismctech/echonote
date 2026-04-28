/**
 * `useMeetingDetail` — orchestrator for the meeting-replay pane.
 *
 * Encapsulates the three IPC-touching meeting actions that used to
 * live inline in `App.tsx`:
 *
 *   - `openMeeting(id)`   — load a meeting and switch the right pane
 *                            to it; surfaces load errors inline AND
 *                            via toast so the user always sees them
 *   - `renameSpeakerAction(speakerId, label)` — rename + re-render
 *                            from the canonical post-rename meeting
 *                            returned by the backend
 *   - `deleteMeetingAction(id)` — delete + refresh sidebar; if the
 *                            deleted meeting is currently open, fall
 *                            back to the live pane
 *
 * The hook deliberately receives `view` / `setView` / `refresh` /
 * `setMeetingsError` from outside instead of owning them — the view
 * state and meetings list are still in `App.tsx` for now and will
 * move into `useMeetingsStore` (Phase 4). Keeping this hook a pure
 * action layer means it slots into either world unchanged.
 */

import { useCallback, useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";

import { useToast } from "../components/Toaster";
import {
  deleteMeeting,
  getMeeting,
  renameMeeting,
  renameSpeaker,
} from "../ipc/client";
import { useIpcAction } from "../ipc/useIpcAction";
import type { MeetingId } from "../types/meeting";
import type { SpeakerId } from "../types/speaker";
import type { MainView } from "../types/view";

export type MeetingsView = MainView;

export function useMeetingDetail({
  view,
  setView,
  refreshMeetings,
  setMeetingsError,
}: {
  view: MeetingsView;
  setView: (next: MeetingsView | ((prev: MeetingsView) => MeetingsView)) => void;
  refreshMeetings: () => Promise<void>;
  setMeetingsError: (message: string | null) => void;
}) {
  const { t } = useTranslation();
  const toast = useToast();

  // Keep a ref to the latest `view` so the action callbacks don't
  // need it as a dependency (view changes on every navigation / load
  // cycle, destabilising every callback downstream).
  const viewRef = useRef(view);
  useEffect(() => {
    viewRef.current = view;
  }, [view]);

  // Wrap renameSpeaker via useIpcAction since the only failure
  // surface is a toast — the success path needs no extra plumbing.
  // openMeeting and deleteMeeting both have inline error state in
  // addition to the toast, so they keep their own try/catch.
  const renameSpeakerCall = useIpcAction(
    t("toast.renameSpeakerFailed"),
    renameSpeaker,
  );

  // Track the last requested meeting id so a slow `getMeeting` for an
  // older click can't overwrite the view after the user already opened
  // a different meeting. Same pattern the FTS search uses with a
  // `cancelled` flag, just keyed by id instead of effect lifetime.
  const lastRequestedRef = useRef<MeetingId | null>(null);

  const openMeeting = useCallback(
    async (id: MeetingId) => {
      lastRequestedRef.current = id;
      setView({ kind: "meeting", id, meeting: null, loading: true });
      try {
        const meeting = await getMeeting(id);
        if (lastRequestedRef.current !== id) return;
        if (!meeting) {
          setView({
            kind: "meeting",
            id,
            meeting: null,
            loading: false,
            error: t("toast.meetingNotFound"),
          });
        } else {
          setView({ kind: "meeting", id, meeting, loading: false });
        }
      } catch (err) {
        if (lastRequestedRef.current !== id) return;
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
          message: t("toast.loadFailed"),
          detail: message,
        });
      }
    },
    [setView, toast, t],
  );

  const renameSpeakerAction = useCallback(
    async (speakerId: SpeakerId, label: string | null) => {
      const v = viewRef.current;
      if (v.kind !== "meeting" || !v.meeting) return;
      const meetingId = v.id;
      const updated = await renameSpeakerCall(meetingId, speakerId, label);
      if (!updated) return;
      // Re-render from the canonical post-rename meeting returned by
      // the backend so we don't drift from disk on the optimistic path.
      setView((prev) =>
        prev.kind === "meeting" && prev.id === meetingId
          ? { kind: "meeting", id: meetingId, meeting: updated, loading: false }
          : prev,
      );
    },
    [renameSpeakerCall, setView],
  );

  const renameMeetingAction = useCallback(
    async (title: string) => {
      const v = viewRef.current;
      if (v.kind !== "meeting" || !v.meeting) return;
      const meetingId = v.id;
      try {
        await renameMeeting(meetingId, title);
        // Patch the in-memory meeting so the header re-renders immediately.
        setView((prev) =>
          prev.kind === "meeting" && prev.id === meetingId && prev.meeting
            ? { ...prev, meeting: { ...prev.meeting, title } }
            : prev,
        );
        await refreshMeetings();
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        toast.push({
          kind: "error",
          message: t("toast.renameFailed"),
          detail: message,
        });
      }
    },
    [setView, refreshMeetings, toast, t],
  );

  const deleteMeetingAction = useCallback(
    async (id: MeetingId) => {
      try {
        const removed = await deleteMeeting(id);
        await refreshMeetings();
        if (viewRef.current.kind === "meeting" && viewRef.current.id === id) {
          setView({ kind: "live" });
        }
        // Backend returns `false` when the row was already gone (race
        // with another window or a stale sidebar click). Surface that
        // honestly instead of pretending the delete succeeded.
        toast.push(
          removed
            ? { kind: "info", message: t("toast.meetingDeleted") }
            : {
                kind: "warning",
                message: t("toast.alreadyGone"),
                detail: t("toast.alreadyGoneDetail"),
              },
        );
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setMeetingsError(message);
        toast.push({
          kind: "error",
          message: t("toast.deleteFailed"),
          detail: message,
        });
      }
    },
    [refreshMeetings, setView, setMeetingsError, toast, t],
  );

  return {
    openMeeting,
    renameMeetingAction,
    renameSpeakerAction,
    deleteMeetingAction,
  };
}
