/**
 * `useSpeakerTools` — post-diarization corrections for an open meeting.
 *
 * Wraps the three irreversible edits (merge two speakers, reassign
 * segments to a speaker, delete segments). Each action runs the IPC
 * call, reloads the meeting from disk and exposes the fresh aggregate
 * as `meeting`, so the view re-renders from the canonical state
 * instead of patching it optimistically.
 *
 * The reloaded meeting is kept as an override tied to the `base`
 * meeting it replaced: as soon as the store hands down a different
 * meeting (navigation, or a rename that returns the canonical row) the
 * override is dropped and the store's value wins again. The meetings
 * store stays the owner of the view; this hook never writes to it.
 *
 * Errors are kept inline (`error`) for the meeting they happened on;
 * actions resolve to `true` on success so callers can clear their
 * selection.
 */

import { useCallback, useState } from "react";
import { useTranslation } from "react-i18next";

import { deleteSegments, getMeeting, mergeMeetingSpeakers, reassignSegments } from "../ipc/client";
import type { SegmentId } from "../types/chat";
import type { Meeting, MeetingId } from "../types/meeting";
import type { SpeakerId } from "../types/speaker";
import { userErrorMessage } from "../types/ipc-error";

export function useSpeakerTools(
  base: Meeting | null,
  /** Called after a successful edit (e.g. refresh the sidebar counts). */
  onEdited?: () => Promise<void> | void,
) {
  const { t } = useTranslation();
  const [override, setOverride] = useState<{ base: Meeting; meeting: Meeting } | null>(null);
  const [failure, setFailure] = useState<{ meetingId: MeetingId; message: string } | null>(null);
  const [busy, setBusy] = useState(false);

  const meeting = override && override.base === base ? override.meeting : base;
  const error = failure && failure.meetingId === base?.id ? failure.message : null;

  const run = useCallback(
    async (op: (meetingId: MeetingId) => Promise<unknown>): Promise<boolean> => {
      if (!base) return false;
      const meetingId = base.id;
      setBusy(true);
      setFailure(null);
      try {
        await op(meetingId);
        const fresh = await getMeeting(meetingId);
        if (fresh) setOverride({ base, meeting: fresh });
        void onEdited?.();
        return true;
      } catch (err) {
        setFailure({ meetingId, message: userErrorMessage(err, t) });
        return false;
      } finally {
        setBusy(false);
      }
    },
    [base, onEdited, t],
  );

  /** Fold `absorbed` into `surviving`; the absorbed speaker disappears. */
  const merge = useCallback(
    (absorbed: SpeakerId, surviving: SpeakerId) =>
      run((id) => mergeMeetingSpeakers(id, absorbed, surviving)),
    [run],
  );

  /** Give `segmentIds` to `speakerId`, or to no speaker when `null`. */
  const reassign = useCallback(
    (segmentIds: SegmentId[], speakerId: SpeakerId | null) =>
      run((id) => reassignSegments(id, segmentIds, speakerId)),
    [run],
  );

  /** Permanently delete `segmentIds` from the meeting. */
  const remove = useCallback(
    (segmentIds: SegmentId[]) => run((id) => deleteSegments(id, segmentIds)),
    [run],
  );

  const clearError = useCallback(() => setFailure(null), []);

  return { meeting, busy, error, merge, reassign, remove, clearError };
}
