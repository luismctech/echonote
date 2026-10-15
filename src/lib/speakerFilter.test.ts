import { describe, expect, it } from "vitest";

import { filterSegmentsBySpeakers, pruneSpeakerFilter, toggleSpeaker } from "./speakerFilter";

const seg = (id: string, speakerId: string | null) => ({ id, speakerId });

describe("toggleSpeaker", () => {
  it("adds a speaker that is not in the filter", () => {
    expect([...toggleSpeaker(new Set(), "a")]).toEqual(["a"]);
  });

  it("removes a speaker that is already in the filter", () => {
    expect([...toggleSpeaker(new Set(["a", "b"]), "a")]).toEqual(["b"]);
  });

  it("supports several selected speakers and never mutates its input", () => {
    const input = new Set(["a"]);
    const next = toggleSpeaker(input, "b");
    expect([...next].sort()).toEqual(["a", "b"]);
    expect([...input]).toEqual(["a"]);
  });
});

describe("filterSegmentsBySpeakers", () => {
  const segments = [seg("1", "a"), seg("2", "b"), seg("3", null), seg("4", "a")];

  it("returns every segment (same array) when the filter is empty", () => {
    expect(filterSegmentsBySpeakers(segments, new Set())).toBe(segments);
  });

  it("keeps only segments of the selected speakers, in order", () => {
    expect(filterSegmentsBySpeakers(segments, new Set(["a"])).map((s) => s.id)).toEqual(["1", "4"]);
    expect(filterSegmentsBySpeakers(segments, new Set(["a", "b"])).map((s) => s.id)).toEqual([
      "1",
      "2",
      "4",
    ]);
  });

  it("never shows segments without a speaker while filtering", () => {
    expect(filterSegmentsBySpeakers(segments, new Set(["b"])).map((s) => s.id)).toEqual(["2"]);
  });
});

describe("pruneSpeakerFilter", () => {
  it("drops ids that are no longer present (e.g. after a merge)", () => {
    expect([...pruneSpeakerFilter(new Set(["a", "gone"]), ["a", "b"])]).toEqual(["a"]);
  });

  it("returns the same set when nothing was dropped", () => {
    const filter = new Set(["a"]);
    expect(pruneSpeakerFilter(filter, ["a", "b"])).toBe(filter);
  });
});
