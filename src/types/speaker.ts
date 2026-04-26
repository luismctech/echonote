/**
 * Diarized-speaker domain types.
 *
 * Mirrors `crates/echo-domain/src/entities/speaker.rs`. The `slot`
 * is the 0-based arrival order; the UI palette is indexed by it so
 * each speaker keeps a stable colour across renames and reloads.
 */

/** UUIDv7 string identifying a diarized speaker within a meeting. */
export type SpeakerId = string;

/**
 * One clustered voice within a meeting.
 *
 * `label` is `null` for anonymous speakers; render `Speaker {slot+1}`
 * in that case (see `src/lib/speakers.ts#displayName`).
 */
export type Speaker = {
  id: SpeakerId;
  slot: number;
  label: string | null;
};
