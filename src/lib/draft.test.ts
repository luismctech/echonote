import { describe, expect, it } from "vitest";

import type { TranscriptEvent } from "../types/streaming";
import { applyDraftEvent, type DraftLine } from "./draft";

const sessionId = "s1" as TranscriptEvent["sessionId"];

function draftEvent(chunkIndex: number, text: string): TranscriptEvent {
  return { type: "draft", sessionId, chunkIndex, offsetMs: chunkIndex * 1000, text };
}

function chunkEvent(chunkIndex: number): TranscriptEvent {
  return {
    type: "chunk",
    sessionId,
    chunkIndex,
    offsetMs: chunkIndex * 1000,
    segments: [],
    language: null,
    rtf: 0.1,
    speakers: [],
  };
}

const draft: DraftLine = { chunkIndex: 2, offsetMs: 2000, text: "hello" };

describe("applyDraftEvent", () => {
  it("keeps only the latest draft", () => {
    const first = applyDraftEvent(null, draftEvent(2, "hello"));
    expect(first).toEqual(draft);
    expect(applyDraftEvent(first, draftEvent(2, "hello world"))).toEqual({
      ...draft,
      text: "hello world",
    });
  });

  it("clears the draft when its final chunk or a later one arrives", () => {
    expect(applyDraftEvent(draft, chunkEvent(2))).toBeNull();
    expect(applyDraftEvent(draft, chunkEvent(3))).toBeNull();
  });

  it("keeps the draft when an earlier chunk arrives", () => {
    expect(applyDraftEvent(draft, chunkEvent(1))).toBe(draft);
  });

  it("clears the draft when its speech is skipped instead", () => {
    const skipped: TranscriptEvent = {
      type: "skipped",
      sessionId,
      chunkIndex: 2,
      offsetMs: 2000,
      durationMs: 4000,
      rms: 0.01,
    };
    expect(applyDraftEvent(draft, skipped)).toBeNull();
  });

  it("clears the draft when the session starts, stops or fails", () => {
    const events: TranscriptEvent[] = [
      { type: "started", sessionId, inputFormat: { sampleRateHz: 16000, channels: 1 } },
      { type: "stopped", sessionId, totalSegments: 1, totalAudioMs: 1000 },
      { type: "failed", sessionId, message: "boom" },
    ];
    for (const evt of events) expect(applyDraftEvent(draft, evt)).toBeNull();
  });

  it("ignores unrelated events", () => {
    const progress: TranscriptEvent = { type: "progress", sessionId, lagMs: 0, backlogMs: 0 };
    expect(applyDraftEvent(draft, progress)).toBe(draft);
    expect(applyDraftEvent(null, progress)).toBeNull();
  });
});
