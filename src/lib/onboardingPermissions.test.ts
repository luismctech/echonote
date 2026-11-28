import { describe, expect, it } from "vitest";

import { permissionsOutcome } from "./onboardingPermissions";

describe("permissionsOutcome", () => {
  it("waits while either check is running", () => {
    expect(permissionsOutcome("checking", "pending")).toBe("wait");
    expect(permissionsOutcome("granted", "checking")).toBe("wait");
  });

  it("advances when the microphone works and system audio needs nothing more", () => {
    expect(permissionsOutcome("granted", "granted")).toBe("advance");
    expect(permissionsOutcome("granted", "notNeeded")).toBe("advance");
    expect(permissionsOutcome("deferred", "granted")).toBe("advance");
  });

  it("stops for the user when system audio was declined", () => {
    expect(permissionsOutcome("granted", "denied")).toBe("review");
    expect(permissionsOutcome("deferred", "denied")).toBe("review");
  });

  it("stops for the user when the microphone failed", () => {
    expect(permissionsOutcome("denied", "granted")).toBe("review");
    expect(permissionsOutcome("denied", "notNeeded")).toBe("review");
  });
});
