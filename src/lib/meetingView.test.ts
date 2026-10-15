import { describe, expect, it } from "vitest";

import { withFreshMeeting } from "./meetingView";
import type { Meeting } from "../types/meeting";
import type { MainView } from "../types/view";

const meeting = (id: string, segments: number, title = "T"): Meeting =>
  ({ id, title, segments: Array.from({ length: segments }, (_, i) => ({ id: `s${i}` })) }) as unknown as Meeting;

describe("withFreshMeeting", () => {
  it("replaces the open meeting with the reloaded one", () => {
    const view: MainView = { kind: "meeting", id: "m1", meeting: meeting("m1", 10), loading: false };
    const next = withFreshMeeting(view, meeting("m1", 4));
    expect(next.kind === "meeting" && next.meeting?.segments.length).toBe(4);
  });

  it("keeps the edit when the title is renamed afterwards", () => {
    // A title rename spreads the store's meeting; it must already be the fresh one.
    const view: MainView = { kind: "meeting", id: "m1", meeting: meeting("m1", 10), loading: false };
    const edited = withFreshMeeting(view, meeting("m1", 4));
    if (edited.kind !== "meeting" || !edited.meeting) throw new Error("unexpected view");
    const renamed = { ...edited, meeting: { ...edited.meeting, title: "New" } };
    expect(renamed.meeting.segments.length).toBe(4);
  });

  it("ignores a reload for a meeting that is no longer open", () => {
    const view: MainView = { kind: "meeting", id: "m2", meeting: meeting("m2", 3), loading: false };
    expect(withFreshMeeting(view, meeting("m1", 4))).toBe(view);
    const live: MainView = { kind: "live" };
    expect(withFreshMeeting(live, meeting("m1", 4))).toBe(live);
  });
});
