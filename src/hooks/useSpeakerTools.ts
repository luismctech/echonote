/**
 * `useSpeakerTools` — post-diarization corrections for an open meeting.
 *
 * Wraps the three irreversible edits (merge two speakers, reassign
 * segments to a speaker, delete segments). Each action runs the IPC
 * call, reloads the meeting from disk and hands the fresh aggregate to
 * `onReloaded`, which puts it into the meetings store. The store stays
 * the single owner of the view, so later actions that build on its
 * copy (e.g. a title rename) never resurrect pre-edit data. The reload
 * also runs when an edit fails, because a multi-step edit may have
 * been partly saved.
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
  /** Receives the meeting reloaded after an edit attempt. */
  onReloaded?: (fresh: Meeting) => Promise<void> | void,
) {
  const { t } = useTranslation();
  const [failure, setFailure] = useState<{ meetingId: MeetingId; message: string } | null>(null);
  const [busy, setBusy] = useState(false);

  const meeting = base;
  const error = failure && failure.meetingId === base?.id ? failure.message : null;

  const run = useCallback(
    async (op: (meetingId: MeetingId) => Promise<unknown>): Promise<boolean> => {
      if (!base) return false;
      const meetingId = base.id;
      setBusy(true);
      setFailure(null);
      let ok = false;
      try {
        await op(meetingId);
        ok = true;
      } catch (err) {
        setFailure({ meetingId, message: userErrorMessage(err, t) });
      }
      try {
        const fresh = await getMeeting(meetingId);
        if (fresh) await onReloaded?.(fresh);
      } catch (err) {
        if (ok) setFailure({ meetingId, message: userErrorMessage(err, t) });
      } finally {
        setBusy(false);
      }
      return ok;
    },
    [base, onReloaded, t],
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
