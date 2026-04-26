/**
 * Recording state machine.
 *
 * Sprint 1 day 1: model the lifecycle of a streaming session as an explicit
 * tagged union + reducer. This makes impossible states unrepresentable
 * (e.g. you can never be `recording` without a `sessionId`) and centralises
 * every transition in one place that the UI just dispatches into.
 *
 * Lifecycle:
 *
 *     Idle ── START_REQUESTED ──▶ Starting
 *      ▲                              │
 *      │             STREAMING_STARTED │
 *      │                              ▼
 *      │                          Recording ── STOP_REQUESTED ──▶ Stopping
 *      │                              │                              │
 *      │            STREAMING_FAILED  │            STREAMING_STOPPED │
 *      │                              ▼                              ▼
 *      └── ACKNOWLEDGE ───────────  Error                       Persisted ──┐
 *                                     ▲                              │     │
 *                                     │                              │     │
 *                                     └── BACKEND_ERROR ─────────────┘     │
 *                                                                          │
 *      ┌─────────────── ACKNOWLEDGE / START_REQUESTED ──────────────────────┘
 *      ▼
 *     Idle
 *
 * Any (state, action) pair not listed below is a no-op (the reducer returns
 * the previous state). Tests assert both legal transitions and at least the
 * three most-likely-to-regress illegal ones.
 */

import type { AudioFormat, StreamingSessionId } from "../types/streaming";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type RecordingState =
  | { kind: "idle" }
  | { kind: "starting" }
  | {
      kind: "recording";
      sessionId: StreamingSessionId;
      inputFormat?: AudioFormat;
    }
  | { kind: "stopping"; sessionId: StreamingSessionId }
  | {
      kind: "persisted";
      lastTotalSegments: number;
      lastTotalAudioMs: number;
    }
  | { kind: "error"; message: string; recoverable: boolean };

export type RecordingAction =
  | { type: "START_REQUESTED" }
  | {
      type: "STREAMING_STARTED";
      sessionId: StreamingSessionId;
      inputFormat: AudioFormat;
    }
  | { type: "STOP_REQUESTED" }
  | {
      type: "STREAMING_STOPPED";
      totalSegments: number;
      totalAudioMs: number;
    }
  | { type: "STREAMING_FAILED"; message: string }
  | { type: "BACKEND_ERROR"; message: string }
  | { type: "ACKNOWLEDGE" };

export const initialRecordingState: RecordingState = { kind: "idle" };

// ---------------------------------------------------------------------------
// Reducer
// ---------------------------------------------------------------------------

export function recordingReducer(
  state: RecordingState,
  action: RecordingAction,
): RecordingState {
  switch (state.kind) {
    case "idle":
      if (action.type === "START_REQUESTED") return { kind: "starting" };
      return state;

    case "starting":
      if (action.type === "STREAMING_STARTED") {
        return {
          kind: "recording",
          sessionId: action.sessionId,
          inputFormat: action.inputFormat,
        };
      }
      if (action.type === "BACKEND_ERROR") {
        return { kind: "error", message: action.message, recoverable: true };
      }
      if (action.type === "STREAMING_FAILED") {
        return { kind: "error", message: action.message, recoverable: true };
      }
      return state;

    case "recording":
      if (action.type === "STOP_REQUESTED") {
        return { kind: "stopping", sessionId: state.sessionId };
      }
      if (action.type === "STREAMING_FAILED") {
        return { kind: "error", message: action.message, recoverable: true };
      }
      if (action.type === "STREAMING_STOPPED") {
        // Backend self-terminated (e.g. duration cap reached) without a UI stop.
        return {
          kind: "persisted",
          lastTotalSegments: action.totalSegments,
          lastTotalAudioMs: action.totalAudioMs,
        };
      }
      return state;

    case "stopping":
      if (action.type === "STREAMING_STOPPED") {
        return {
          kind: "persisted",
          lastTotalSegments: action.totalSegments,
          lastTotalAudioMs: action.totalAudioMs,
        };
      }
      if (action.type === "STREAMING_FAILED") {
        return { kind: "error", message: action.message, recoverable: false };
      }
      if (action.type === "BACKEND_ERROR") {
        return { kind: "error", message: action.message, recoverable: false };
      }
      return state;

    case "persisted":
      if (action.type === "ACKNOWLEDGE") return { kind: "idle" };
      if (action.type === "START_REQUESTED") return { kind: "starting" };
      return state;

    case "error":
      if (action.type === "ACKNOWLEDGE") return { kind: "idle" };
      if (action.type === "START_REQUESTED" && state.recoverable) {
        return { kind: "starting" };
      }
      return state;
  }
}

// ---------------------------------------------------------------------------
// Selectors
// ---------------------------------------------------------------------------

/** True when the user is allowed to press Start right now. */
export function canStart(state: RecordingState): boolean {
  switch (state.kind) {
    case "idle":
    case "persisted":
      return true;
    case "error":
      return state.recoverable;
    case "starting":
    case "recording":
    case "stopping":
      return false;
  }
}

/** True when the user is allowed to press Stop right now. */
export function canStop(state: RecordingState): boolean {
  return state.kind === "recording";
}

/** Short label for the status pill ("● recording", "○ idle", …). */
export function statusLabel(state: RecordingState): string {
  switch (state.kind) {
    case "idle":
      return "○ idle";
    case "starting":
      return "○ starting";
    case "recording":
      return "● recording";
    case "stopping":
      return "○ stopping";
    case "persisted":
      return "✓ saved";
    case "error":
      return "✗ error";
  }
}
