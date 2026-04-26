/**
 * Typed IPC client for the Tauri backend.
 *
 * Sprint 0 day 4 hand-rolls these types. Once the backend surface grows
 * (Sprint 1), generation will be delegated to `tauri-specta`, which emits
 * this file from Rust `#[specta::specta]` annotations. Hand-rolled shapes
 * here must match `src-tauri/src/commands.rs` one-for-one.
 */

import { Channel, invoke } from "@tauri-apps/api/core";

// ---------------------------------------------------------------------------
// Health (Sprint 0 day 4)
// ---------------------------------------------------------------------------

export type HealthStatus = {
  /** ISO 8601 instant the backend answered. */
  timestamp: string;
  /** EchoNote semver, pulled from Cargo.toml at compile time. */
  version: string;
  /** Target triple the backend was compiled for. */
  target: string;
  /** Short git hash, `unknown` outside CI or when .git is absent. */
  commit: string;
};

/** True when the frontend is running inside a Tauri webview. */
export const isTauri = (): boolean =>
  typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

export async function healthCheck(): Promise<HealthStatus> {
  return invoke<HealthStatus>("health_check");
}

// ---------------------------------------------------------------------------
// Streaming transcription (Sprint 0 day 7)
// ---------------------------------------------------------------------------

/** UUIDv7 string identifying a streaming session. */
export type StreamingSessionId = string;

export type AudioFormat = {
  sampleRateHz: number;
  channels: number;
};

export type Segment = {
  id: string;
  startMs: number;
  endMs: number;
  text: string;
  speakerId: string | null;
  confidence: number | null;
};

/** Discriminated union of every event the backend may emit. */
export type TranscriptEvent =
  | {
      type: "started";
      sessionId: StreamingSessionId;
      inputFormat: AudioFormat;
    }
  | {
      type: "chunk";
      sessionId: StreamingSessionId;
      chunkIndex: number;
      offsetMs: number;
      segments: Segment[];
      language: string | null;
      rtf: number;
    }
  | {
      type: "skipped";
      sessionId: StreamingSessionId;
      chunkIndex: number;
      offsetMs: number;
      durationMs: number;
      rms: number;
    }
  | {
      type: "stopped";
      sessionId: StreamingSessionId;
      totalSegments: number;
      totalAudioMs: number;
    }
  | {
      type: "failed";
      sessionId: StreamingSessionId;
      message: string;
    };

export type StartStreamingOptions = {
  language?: string;
  deviceId?: string;
  chunkMs?: number;
  silenceRmsThreshold?: number;
};

/**
 * Start a streaming transcription session. The supplied `onEvent`
 * callback will fire for every backend event until a `stopped` or
 * `failed` event arrives.
 *
 * Returns the assigned session id, which must be passed back to
 * {@link stopStreaming} to terminate the session early.
 */
export async function startStreaming(
  options: StartStreamingOptions | undefined,
  onEvent: (event: TranscriptEvent) => void,
): Promise<StreamingSessionId> {
  const channel = new Channel<TranscriptEvent>();
  channel.onmessage = onEvent;
  return invoke<StreamingSessionId>("start_streaming", {
    options: options ?? null,
    onEvent: channel,
  });
}

/**
 * Stop a streaming session. Resolves to `true` when the session was
 * found and stopped, `false` when the id was unknown (already stopped
 * or never existed). Idempotent.
 */
export async function stopStreaming(
  sessionId: StreamingSessionId,
): Promise<boolean> {
  return invoke<boolean>("stop_streaming", { sessionId });
}

// ---------------------------------------------------------------------------
// Meetings (Sprint 0 day 8)
// ---------------------------------------------------------------------------

/** UUIDv7 string identifying a persisted meeting. */
export type MeetingId = string;

/** Lightweight projection used by the sidebar listing. */
export type MeetingSummary = {
  id: MeetingId;
  title: string;
  startedAt: string;
  endedAt: string | null;
  durationMs: number;
  language: string | null;
  segmentCount: number;
};

/** Full meeting aggregate (header + segments). */
export type Meeting = MeetingSummary & {
  inputFormat: AudioFormat;
  segments: Segment[];
};

/**
 * List meetings, newest first.
 *
 * @param limit Maximum rows to return. `0` (the default) means no cap.
 */
export async function listMeetings(limit = 0): Promise<MeetingSummary[]> {
  return invoke<MeetingSummary[]>("list_meetings", { limit });
}

/** Fetch a single meeting (header + segments) or `null` when missing. */
export async function getMeeting(id: MeetingId): Promise<Meeting | null> {
  return invoke<Meeting | null>("get_meeting", { id });
}

/** Delete a meeting and its segments. Resolves to `true` when deleted. */
export async function deleteMeeting(id: MeetingId): Promise<boolean> {
  return invoke<boolean>("delete_meeting", { id });
}
