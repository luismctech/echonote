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

import { useCallback } from "react";

import { useToast } from "../components/Toaster";
import {
  deleteMeeting,
  getMeeting,
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
  const toast = useToast();

  // Wrap renameSpeaker via useIpcAction since the only failure
  // surface is a toast — the success path needs no extra plumbing.
  // openMeeting and deleteMeeting both have inline error state in
  // addition to the toast, so they keep their own try/catch.
  const renameSpeakerCall = useIpcAction(
    "Couldn't rename speaker.",
    renameSpeaker,
  );

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
    [setView, toast],
  );

  const renameSpeakerAction = useCallback(
    async (speakerId: SpeakerId, label: string | null) => {
      if (view.kind !== "meeting" || !view.meeting) return;
      const meetingId = view.id;
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
    [view, renameSpeakerCall, setView],
  );

  const deleteMeetingAction = useCallback(
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
    [refreshMeetings, view, setView, setMeetingsError, toast],
  );

  return {
    openMeeting,
    renameSpeakerAction,
    deleteMeetingAction,
  };
}
