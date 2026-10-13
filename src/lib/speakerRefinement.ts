/**
 * Live-pane handling of the end-of-session speaker refinement events
 * (`speakersReassigned`, `speakersMerged`; ADR 0001).
 *
 * The backend has already persisted the change when these events
 * arrive, so the stored meeting view picks it up on its next fetch.
 * These helpers only patch the in-memory live lines and build the
 * one-line notice shown after stop. Every patch is set-to-value, so
 * applying the same event twice gives the same lines.
 */

import type { SpeakerId } from "../types/speaker";
import type { ChunkSpeaker, SpeakerMerge, TranscriptEvent } from "../types/streaming";
import type { StreamLine } from "../types/view";

/**
 * Set the refined dominant speaker on every live line whose chunk is
 * listed. Lines already evicted by the live-line cap are simply absent.
 */
export function applyChunkSpeakers(
  lines: StreamLine[],
  chunks: ChunkSpeaker[],
): StreamLine[] {
  if (chunks.length === 0) return lines;
  const byChunk = new Map(chunks.map((c) => [c.chunkIndex, c]));
  return lines.map((line) => {
    const change = line.kind === "chunk" ? byChunk.get(line.chunkIndex) : undefined;
    if (line.kind !== "chunk" || change === undefined) return line;
    return {
      ...line,
      speakerId: change.speakerId ?? undefined,
      speakerSlot: change.speakerSlot ?? undefined,
    };
  });
}

/**
 * Fold absorbed speakers into their survivors on the live lines. Merge
 * chains resolve to the final survivor (`a→b`, `b→c` moves `a` to `c`);
 * speakers caught in a merge cycle keep their lines. The survivor's slot
 * is taken from a line it already owns; lines whose survivor never
 * appeared keep their slot.
 */
export function applySpeakerMerges(
  lines: StreamLine[],
  merges: SpeakerMerge[],
): StreamLine[] {
  if (merges.length === 0) return lines;
  const slotOf = new Map<SpeakerId, number>();
  for (const line of lines) {
    if (line.kind === "chunk" && line.speakerId !== undefined && line.speakerSlot !== undefined) {
      slotOf.set(line.speakerId, line.speakerSlot);
    }
  }
  const survivorOf = new Map(merges.map((m) => [m.absorbed, m.surviving]));
  return lines.map((line) => {
    if (line.kind !== "chunk" || line.speakerId === undefined) return line;
    const survivor = finalSurvivor(survivorOf, line.speakerId);
    const slot = survivor === undefined ? undefined : slotOf.get(survivor);
    if (survivor === undefined || slot === undefined) return line;
    return { ...line, speakerId: survivor, speakerSlot: slot };
  });
}

/**
 * Follow `survivorOf` from `id` to the speaker that is not absorbed any
 * further. `undefined` when `id` is not absorbed or the chain loops.
 */
function finalSurvivor(
  survivorOf: Map<SpeakerId, SpeakerId>,
  id: SpeakerId,
): SpeakerId | undefined {
  const seen = new Set<SpeakerId>();
  let current = id;
  for (;;) {
    const next = survivorOf.get(current);
    if (next === undefined) break;
    if (seen.has(current)) return undefined;
    seen.add(current);
    current = next;
  }
  return current === id ? undefined : current;
}

/** Counts for the post-stop notice. */
export type RefinementSummary = {
  /** Segments that changed speaker (0 for a merge-only refine). */
  segmentsMoved: number;
  /** Speakers folded into another speaker. */
  speakersMerged: number;
};

/**
 * Summarise a refinement event for the notice, or `null` for any other
 * event or a refinement that changed nothing.
 */
export function refinementSummary(evt: TranscriptEvent): RefinementSummary | null {
  if (evt.type === "speakersReassigned") {
    return { segmentsMoved: evt.segments.length, speakersMerged: evt.retired.length };
  }
  if (evt.type === "speakersMerged" && evt.merges.length > 0) {
    return { segmentsMoved: 0, speakersMerged: evt.merges.length };
  }
  return null;
}
