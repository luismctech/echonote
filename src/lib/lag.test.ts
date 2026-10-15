import { describe, expect, it } from "vitest";

import { lagIndicatorSeconds } from "./lag";

describe("lagIndicatorSeconds", () => {
  it("hides the indicator up to 3 s of lag", () => {
    expect(lagIndicatorSeconds(0)).toBeNull();
    expect(lagIndicatorSeconds(3000)).toBeNull();
  });

  it("shows whole seconds above 3 s of lag", () => {
    expect(lagIndicatorSeconds(3001)).toBe(3);
    expect(lagIndicatorSeconds(12_400)).toBe(12);
    expect(lagIndicatorSeconds(12_600)).toBe(13);
  });
});
