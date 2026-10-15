import { describe, expect, it } from "vitest";

import {
  EMPTY_SELECTION,
  pruneSelection,
  selectAllVisible,
  selectRange,
  toggleSegment,
} from "./segmentSelection";

const ids = ["s1", "s2", "s3", "s4", "s5", "s6"];
const sorted = (s: ReadonlySet<string>) => [...s].sort();

describe("toggleSegment", () => {
  it("selects an unselected segment and makes it the anchor", () => {
    const next = toggleSegment(EMPTY_SELECTION, "s2");
    expect(sorted(next.selected)).toEqual(["s2"]);
    expect(next.anchor).toBe("s2");
  });

  it("deselects a selected segment and moves the anchor to it", () => {
    const next = toggleSegment(toggleSegment(EMPTY_SELECTION, "s2"), "s2");
    expect(next.selected.size).toBe(0);
    expect(next.anchor).toBe("s2");
  });
});

describe("selectRange", () => {
  it("selects from the anchor to the clicked segment, inclusive", () => {
    const next = selectRange(toggleSegment(EMPTY_SELECTION, "s2"), ids, "s5");
    expect(sorted(next.selected)).toEqual(["s2", "s3", "s4", "s5"]);
  });

  it("works backwards when the clicked segment is before the anchor", () => {
    const next = selectRange(toggleSegment(EMPTY_SELECTION, "s5"), ids, "s3");
    expect(sorted(next.selected)).toEqual(["s3", "s4", "s5"]);
  });

  it("keeps the anchor so a second shift-click extends from the same point", () => {
    const first = selectRange(toggleSegment(EMPTY_SELECTION, "s2"), ids, "s3");
    const second = selectRange(first, ids, "s4");
    expect(second.anchor).toBe("s2");
    expect(sorted(second.selected)).toEqual(["s2", "s3", "s4"]);
  });

  it("adds the range to an existing selection", () => {
    const start = toggleSegment(toggleSegment(EMPTY_SELECTION, "s1"), "s4");
    const next = selectRange(start, ids, "s6");
    expect(sorted(next.selected)).toEqual(["s1", "s4", "s5", "s6"]);
  });

  it("uses the visible order (filtered list), skipping hidden segments", () => {
    const visible = ["s1", "s3", "s5"];
    const next = selectRange(toggleSegment(EMPTY_SELECTION, "s1"), visible, "s5");
    expect(sorted(next.selected)).toEqual(["s1", "s3", "s5"]);
  });

  it("falls back to a toggle when there is no anchor or it is not visible", () => {
    expect(sorted(selectRange(EMPTY_SELECTION, ids, "s3").selected)).toEqual(["s3"]);
    const hiddenAnchor = toggleSegment(EMPTY_SELECTION, "s2");
    const next = selectRange(hiddenAnchor, ["s1", "s3"], "s3");
    expect(sorted(next.selected)).toEqual(["s2", "s3"]);
    expect(next.anchor).toBe("s3");
  });
});

describe("selectAllVisible", () => {
  it("selects every visible segment and keeps earlier picks", () => {
    const next = selectAllVisible(toggleSegment(EMPTY_SELECTION, "s6"), ["s1", "s2"]);
    expect(sorted(next.selected)).toEqual(["s1", "s2", "s6"]);
  });
});

describe("pruneSelection", () => {
  it("drops segments that are gone or hidden and clears a dropped anchor", () => {
    const start = selectRange(toggleSegment(EMPTY_SELECTION, "s1"), ids, "s3");
    const next = pruneSelection(start, ["s2", "s3"]);
    expect(sorted(next.selected)).toEqual(["s2", "s3"]);
    expect(next.anchor).toBeNull();
  });

  it("returns the same state when nothing was dropped", () => {
    const start = toggleSegment(EMPTY_SELECTION, "s1");
    expect(pruneSelection(start, ids)).toBe(start);
  });
});
