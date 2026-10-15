/**
 * Provisional ("draft") text for the live view.
 *
 * A speech chunk only closes at a pause, so its final text can arrive
 * many seconds after its first word. Meanwhile the backend sends
 * `draft` events with what the open chunk holds so far. The live view
 * shows at most one draft line, after the final lines, until the chunk
 * it belongs to closes. Drafts are never persisted.
 */

import type { TranscriptEvent } from "../types/streaming";

/** The draft line currently shown, if any. */
export type DraftLine = {
  /** Index the final chunk of this speech will arrive with. */
  chunkIndex: number;
  /** Session-time start of that chunk (ms). */
  offsetMs: number;
  /** Text transcribed so far. */
  text: string;
};

/**
 * Next draft state after `evt`.
 *
 * - `draft` replaces the current draft (only the latest is kept);
 * - a `chunk` or `skipped` event with the draft's chunk index or a
 *   later one clears it: that chunk closed, and its final outcome
 *   replaces the provisional text;
 * - `started`, `stopped` and `failed` clear it;
 * - any other event returns `draft` unchanged (same reference).
 */
export function applyDraftEvent(
  draft: DraftLine | null,
  evt: TranscriptEvent,
): DraftLine | null {
  switch (evt.type) {
    case "draft":
      return { chunkIndex: evt.chunkIndex, offsetMs: evt.offsetMs, text: evt.text };
    case "chunk":
    case "skipped":
      return draft !== null && evt.chunkIndex >= draft.chunkIndex ? null : draft;
    case "started":
    case "stopped":
    case "failed":
      return null;
    default:
      return draft;
  }
}
