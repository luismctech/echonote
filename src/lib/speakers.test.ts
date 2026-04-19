import { describe, expect, it } from "vitest";

import {
  displayName,
  indexSpeakers,
  paletteFor,
  shortTag,
  SPEAKER_PALETTE,
} from "./speakers";
import type { Speaker } from "../types/speaker";

const speaker = (id: string, slot: number, label: string | null): Speaker => ({
  id,
  slot,
  label,
});

describe("paletteFor", () => {
  it("is stable across calls (same slot → same entry)", () => {
    expect(paletteFor(0)).toBe(paletteFor(0));
    expect(paletteFor(3)).toBe(paletteFor(3));
  });

  it("cycles past the palette length so every slot maps to something", () => {
    const wrapped = paletteFor(SPEAKER_PALETTE.length);
    expect(wrapped).toBe(paletteFor(0));
    expect(paletteFor(SPEAKER_PALETTE.length + 2)).toBe(paletteFor(2));
  });

  it("clamps NaN/negative slots to a real entry instead of crashing", () => {
    expect(paletteFor(Number.NaN)).toBe(SPEAKER_PALETTE[0]);
    expect(paletteFor(-1)).toBe(paletteFor((-1 >>> 0) % SPEAKER_PALETTE.length));
  });
});

describe("displayName", () => {
  it("uses the user label when set", () => {
    expect(displayName(speaker("a", 0, "Alice"))).toBe("Alice");
  });

  it("falls back to `Speaker {slot+1}` for anonymous speakers", () => {
    expect(displayName(speaker("a", 0, null))).toBe("Speaker 1");
    expect(displayName(speaker("a", 4, null))).toBe("Speaker 5");
  });

  it("treats whitespace-only labels as anonymous", () => {
    expect(displayName(speaker("a", 2, "   "))).toBe("Speaker 3");
  });

  it("trims surrounding whitespace from labels", () => {
    expect(displayName(speaker("a", 0, "  Bob  "))).toBe("Bob");
  });
});

describe("shortTag", () => {
  it("formats as S{slot+1}", () => {
    expect(shortTag(0)).toBe("S1");
    expect(shortTag(7)).toBe("S8");
  });
});

describe("indexSpeakers", () => {
  it("returns an empty map for an empty list (so callers don't need guards)", () => {
    expect(indexSpeakers([]).size).toBe(0);
  });

  it("indexes by id with O(1) lookups", () => {
    const a = speaker("a", 0, null);
    const b = speaker("b", 1, "Bob");
    const map = indexSpeakers([a, b]);
    expect(map.get("a")).toBe(a);
    expect(map.get("b")).toBe(b);
    expect(map.get("missing")).toBeUndefined();
  });
});
