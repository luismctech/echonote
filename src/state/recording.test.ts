import { describe, expect, it } from "vitest";

import {
  canStart,
  canStop,
  initialRecordingState,
  recordingReducer,
  statusLabel,
  type RecordingAction,
  type RecordingState,
} from "./recording";

const SESSION = "01900000-0000-7000-8000-000000000000";
const FORMAT = { sampleRateHz: 44_100, channels: 1 };

function reduce(
  state: RecordingState,
  ...actions: RecordingAction[]
): RecordingState {
  return actions.reduce(recordingReducer, state);
}

describe("recordingReducer — happy path", () => {
  it("walks Idle → Starting → Recording → Stopping → Persisted → Idle", () => {
    let s: RecordingState = initialRecordingState;
    expect(s.kind).toBe("idle");

    s = recordingReducer(s, { type: "START_REQUESTED" });
    expect(s.kind).toBe("starting");

    s = recordingReducer(s, {
      type: "STREAMING_STARTED",
      sessionId: SESSION,
      inputFormat: FORMAT,
    });
    expect(s).toEqual({
      kind: "recording",
      sessionId: SESSION,
      inputFormat: FORMAT,
    });

    s = recordingReducer(s, { type: "STOP_REQUESTED" });
    expect(s).toEqual({ kind: "stopping", sessionId: SESSION });

    s = recordingReducer(s, {
      type: "STREAMING_STOPPED",
      totalSegments: 12,
      totalAudioMs: 30_000,
    });
    expect(s).toEqual({
      kind: "persisted",
      lastTotalSegments: 12,
      lastTotalAudioMs: 30_000,
    });

    s = recordingReducer(s, { type: "ACKNOWLEDGE" });
    expect(s).toEqual({ kind: "idle" });
  });

  it("allows Persisted → Starting (start a new session right away)", () => {
    const persisted: RecordingState = {
      kind: "persisted",
      lastTotalSegments: 1,
      lastTotalAudioMs: 1_000,
    };
    expect(recordingReducer(persisted, { type: "START_REQUESTED" })).toEqual({
      kind: "starting",
    });
  });
});

describe("recordingReducer — error transitions", () => {
  it("Starting + BACKEND_ERROR → recoverable error", () => {
    const s = reduce(initialRecordingState, { type: "START_REQUESTED" }, {
      type: "BACKEND_ERROR",
      message: "mic permission denied",
    });
    expect(s).toEqual({
      kind: "error",
      message: "mic permission denied",
      recoverable: true,
    });
  });

  it("Recording + STREAMING_FAILED → recoverable error", () => {
    const s = reduce(
      initialRecordingState,
      { type: "START_REQUESTED" },
      {
        type: "STREAMING_STARTED",
        sessionId: SESSION,
        inputFormat: FORMAT,
      },
      { type: "STREAMING_FAILED", message: "whisper crashed" },
    );
    expect(s).toEqual({
      kind: "error",
      message: "whisper crashed",
      recoverable: true,
    });
  });

  it("Stopping + STREAMING_FAILED → non-recoverable error (data may be partial)", () => {
    const s = reduce(
      initialRecordingState,
      { type: "START_REQUESTED" },
      {
        type: "STREAMING_STARTED",
        sessionId: SESSION,
        inputFormat: FORMAT,
      },
      { type: "STOP_REQUESTED" },
      { type: "STREAMING_FAILED", message: "db write failed" },
    );
    expect(s).toEqual({
      kind: "error",
      message: "db write failed",
      recoverable: false,
    });
  });

  it("Recording can self-terminate via STREAMING_STOPPED (e.g. duration cap)", () => {
    const s = reduce(
      initialRecordingState,
      { type: "START_REQUESTED" },
      {
        type: "STREAMING_STARTED",
        sessionId: SESSION,
        inputFormat: FORMAT,
      },
      {
        type: "STREAMING_STOPPED",
        totalSegments: 3,
        totalAudioMs: 10_000,
      },
    );
    expect(s).toEqual({
      kind: "persisted",
      lastTotalSegments: 3,
      lastTotalAudioMs: 10_000,
    });
  });

  it("Error + ACKNOWLEDGE returns to idle", () => {
    const s = recordingReducer(
      { kind: "error", message: "x", recoverable: true },
      { type: "ACKNOWLEDGE" },
    );
    expect(s).toEqual({ kind: "idle" });
  });

  it("Error + START_REQUESTED retries when recoverable", () => {
    const s = recordingReducer(
      { kind: "error", message: "x", recoverable: true },
      { type: "START_REQUESTED" },
    );
    expect(s).toEqual({ kind: "starting" });
  });

  it("Error + START_REQUESTED is a no-op when NOT recoverable", () => {
    const before: RecordingState = {
      kind: "error",
      message: "x",
      recoverable: false,
    };
    const after = recordingReducer(before, { type: "START_REQUESTED" });
    expect(after).toBe(before);
  });
});

describe("recordingReducer — illegal transitions are no-ops", () => {
  it("Idle ignores STOP_REQUESTED", () => {
    const before = initialRecordingState;
    expect(recordingReducer(before, { type: "STOP_REQUESTED" })).toBe(before);
  });

  it("Idle ignores STREAMING_STARTED arriving out of band", () => {
    const before = initialRecordingState;
    expect(
      recordingReducer(before, {
        type: "STREAMING_STARTED",
        sessionId: SESSION,
        inputFormat: FORMAT,
      }),
    ).toBe(before);
  });

  it("Recording ignores a duplicate STREAMING_STARTED", () => {
    const before: RecordingState = {
      kind: "recording",
      sessionId: SESSION,
      inputFormat: FORMAT,
    };
    expect(
      recordingReducer(before, {
        type: "STREAMING_STARTED",
        sessionId: "other",
        inputFormat: FORMAT,
      }),
    ).toBe(before);
  });

  it("Stopping ignores a fresh START_REQUESTED (must wait for the stop to land)", () => {
    const before: RecordingState = { kind: "stopping", sessionId: SESSION };
    expect(recordingReducer(before, { type: "START_REQUESTED" })).toBe(before);
  });
});

describe("selectors", () => {
  it("canStart only when idle, persisted, or recoverable error", () => {
    expect(canStart({ kind: "idle" })).toBe(true);
    expect(
      canStart({
        kind: "persisted",
        lastTotalSegments: 1,
        lastTotalAudioMs: 0,
      }),
    ).toBe(true);
    expect(canStart({ kind: "error", message: "x", recoverable: true })).toBe(
      true,
    );
    expect(canStart({ kind: "error", message: "x", recoverable: false })).toBe(
      false,
    );
    expect(canStart({ kind: "starting" })).toBe(false);
    expect(
      canStart({
        kind: "recording",
        sessionId: SESSION,
        inputFormat: FORMAT,
      }),
    ).toBe(false);
    expect(canStart({ kind: "stopping", sessionId: SESSION })).toBe(false);
  });

  it("canStop only when recording", () => {
    expect(
      canStop({ kind: "recording", sessionId: SESSION, inputFormat: FORMAT }),
    ).toBe(true);
    expect(canStop({ kind: "stopping", sessionId: SESSION })).toBe(false);
    expect(canStop({ kind: "idle" })).toBe(false);
  });

  it("statusLabel covers every variant", () => {
    expect(statusLabel({ kind: "idle" })).toMatch(/idle/);
    expect(statusLabel({ kind: "starting" })).toMatch(/starting/);
    expect(
      statusLabel({
        kind: "recording",
        sessionId: SESSION,
        inputFormat: FORMAT,
      }),
    ).toMatch(/recording/);
    expect(statusLabel({ kind: "stopping", sessionId: SESSION })).toMatch(
      /stopping/,
    );
    expect(
      statusLabel({
        kind: "persisted",
        lastTotalSegments: 1,
        lastTotalAudioMs: 0,
      }),
    ).toMatch(/saved/);
    expect(
      statusLabel({ kind: "error", message: "x", recoverable: true }),
    ).toMatch(/error/);
  });
});
