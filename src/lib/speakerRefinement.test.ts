import { describe, expect, it } from "vitest";

import { applyChunkSpeakers, applySpeakerMerges, refinementSummary } from "./speakerRefinement";
import type { TranscriptEvent } from "../types/streaming";
import type { StreamLine } from "../types/view";

const chunk = (chunkIndex: number, speakerId?: string, speakerSlot?: number): StreamLine => ({
  kind: "chunk",
  key: `s-${chunkIndex}`,
  chunkIndex,
  offsetMs: chunkIndex * 5_000,
  text: `chunk ${chunkIndex}`,
  language: "es",
  rtf: 0.1,
  speakerId,
  speakerSlot,
});

const skipped: StreamLine = {
  kind: "skipped",
  key: "s-9",
  chunkIndex: 9,
  offsetMs: 45_000,
  durationMs: 1_000,
  rms: 0.001,
};

const reassigned: Extract<TranscriptEvent, { type: "speakersReassigned" }> = {
  type: "speakersReassigned",
  sessionId: "session",
  planId: "plan",
  created: [{ id: "c", slot: 2 }],
  retired: [{ speaker: "b", majorityRecipient: "a" }],
  segments: [
    { segmentId: "s1", speakerId: "a" },
    { segmentId: "s2", speakerId: "c" },
    { segmentId: "s3", speakerId: "a" },
  ],
  chunks: [
    { chunkIndex: 1, speakerId: "a", speakerSlot: 0 },
    { chunkIndex: 2, speakerId: "c", speakerSlot: 2 },
  ],
};

describe("applyChunkSpeakers", () => {
  it("sets the refined speaker on listed chunks only", () => {
    const lines = [chunk(0, "a", 0), chunk(1, "b", 1), chunk(2, "b", 1), skipped];
    const out = applyChunkSpeakers(lines, reassigned.chunks);
    expect(out.map((l) => (l.kind === "chunk" ? l.speakerSlot : "skip"))).toEqual([
      0,
      0,
      2,
      "skip",
    ]);
    expect(out[0]).toBe(lines[0]);
  });

  it("is idempotent and ignores chunks no longer in the pane", () => {
    // Chunk 1 is listed but was already evicted from the pane.
    const lines = [chunk(2, "b", 1)];
    const once = applyChunkSpeakers(lines, reassigned.chunks);
    expect(once).toEqual([chunk(2, "c", 2)]);
    expect(applyChunkSpeakers(once, reassigned.chunks)).toEqual(once);
  });

  it("clears the speaker of a chunk the refine left without one", () => {
    const out = applyChunkSpeakers(
      [chunk(3, "a", 0)],
      [{ chunkIndex: 3, speakerId: null, speakerSlot: null }],
    );
    expect(out[0]).toMatchObject({ speakerId: undefined, speakerSlot: undefined });
  });
});

describe("applySpeakerMerges", () => {
  it("moves absorbed lines to the survivor's slot", () => {
    const lines = [chunk(0, "a", 0), chunk(1, "b", 1), skipped];
    const out = applySpeakerMerges(lines, [{ absorbed: "b", surviving: "a" }]);
    expect(out[1]).toMatchObject({ speakerId: "a", speakerSlot: 0 });
    expect(out[0]).toBe(lines[0]);
  });

  it("resolves merge chains to the final survivor", () => {
    const lines = [chunk(0, "a", 0), chunk(1, "b", 1), chunk(2, "c", 2)];
    const out = applySpeakerMerges(lines, [
      { absorbed: "a", surviving: "b" },
      { absorbed: "b", surviving: "c" },
    ]);
    expect(out.map((l) => (l.kind === "chunk" ? [l.speakerId, l.speakerSlot] : null))).toEqual([
      ["c", 2],
      ["c", 2],
      ["c", 2],
    ]);
  });

  it("leaves lines in a merge cycle alone", () => {
    const lines = [chunk(0, "a", 0), chunk(1, "b", 1)];
    const out = applySpeakerMerges(lines, [
      { absorbed: "a", surviving: "b" },
      { absorbed: "b", surviving: "a" },
    ]);
    expect(out).toEqual(lines);
  });

  it("leaves lines without a speaker alone", () => {
    const lines = [chunk(0), chunk(1, "b", 1)];
    const out = applySpeakerMerges(lines, [{ absorbed: "b", surviving: "a" }]);
    expect(out[0]).toBe(lines[0]);
  });

  it("leaves lines alone when the survivor's slot is unknown", () => {
    const lines = [chunk(1, "b", 1)];
    expect(applySpeakerMerges(lines, [{ absorbed: "b", surviving: "z" }])).toEqual(lines);
  });
});

describe("refinementSummary", () => {
  it("counts moved segments and retired speakers of a reassignment", () => {
    expect(refinementSummary(reassigned)).toEqual({ segmentsMoved: 3, speakersMerged: 1 });
  });

  it("counts merges of a merge-only refine", () => {
    const evt: TranscriptEvent = {
      type: "speakersMerged",
      sessionId: "session",
      merges: [{ absorbed: "b", surviving: "a" }],
    };
    expect(refinementSummary(evt)).toEqual({ segmentsMoved: 0, speakersMerged: 1 });
  });

  it("is null for other events and empty merges", () => {
    expect(refinementSummary({ type: "paused", sessionId: "session" })).toBeNull();
    expect(
      refinementSummary({ type: "speakersMerged", sessionId: "session", merges: [] }),
    ).toBeNull();
  });
});
