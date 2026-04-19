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

/** UUIDv7 string identifying a diarized speaker within a meeting. */
export type SpeakerId = string;

/**
 * One clustered voice within a meeting.
 *
 * `slot` is the 0-based arrival order; the UI palette is indexed by
 * it so the colour stays stable across renames and reloads. `label`
 * is `null` for anonymous speakers; render `Speaker {slot+1}` then.
 */
export type Speaker = {
  id: SpeakerId;
  slot: number;
  label: string | null;
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
      /**
       * Speaker the diarizer assigned to every segment in this chunk.
       * `undefined` (omitted on the wire) when no diarizer is wired
       * into the pipeline OR when the chunk was too short to embed.
       */
      speakerId?: SpeakerId;
      /**
       * Arrival-order slot of {@link speakerId}, mirrored so the UI
       * palette can colour the chip without round-tripping through
       * the speakers list. `undefined` whenever `speakerId` is
       * `undefined`.
       */
      speakerSlot?: number;
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

/**
 * Where the backend should pull audio from.
 *
 * - `microphone`: default cpal input. Requires Microphone permission
 *   on macOS the first time it runs.
 * - `systemOutput`: the system audio mix (the "other side of the
 *   call"). macOS 13+ only — uses ScreenCaptureKit and requires
 *   Screen Recording permission. The backend ignores `deviceId` for
 *   this source.
 */
export type AudioSourceKind = "microphone" | "systemOutput";

export type StartStreamingOptions = {
  source?: AudioSourceKind;
  language?: string;
  deviceId?: string;
  chunkMs?: number;
  silenceRmsThreshold?: number;
  /**
   * Enable speaker diarization. When `true`, the backend loads the
   * speaker embedder and attaches an online diarizer to the pipeline,
   * so every chunk event carries a `speakerId` + `speakerSlot` and
   * the meeting persists its speakers. Defaults to `false`.
   */
  diarize?: boolean;
  /**
   * Override path to the speaker-embedder ONNX. Most callers should
   * leave this unset so the backend uses its configured default.
   */
  embedModelPath?: string;
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

/** Full meeting aggregate (header + segments + speakers). */
export type Meeting = MeetingSummary & {
  inputFormat: AudioFormat;
  segments: Segment[];
  /** Diarized speakers, ordered by `slot` ascending. May be empty. */
  speakers: Speaker[];
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

/**
 * Set or clear a speaker's user-visible label. Pass `null` (or an
 * empty string) to revert the speaker to anonymous so the UI renders
 * `Speaker N` again. Returns the freshly-loaded meeting so the caller
 * can re-render speakers + segment chips from a single source of
 * truth without an extra `getMeeting` round-trip.
 */
export async function renameSpeaker(
  meetingId: MeetingId,
  speakerId: SpeakerId,
  label: string | null,
): Promise<Meeting> {
  return invoke<Meeting>("rename_speaker", {
    meetingId,
    speakerId,
    label,
  });
}

// ---------------------------------------------------------------------------
// Meeting search (Sprint 1 day 8)
// ---------------------------------------------------------------------------

/**
 * One hit returned by {@link searchMeetings}.
 *
 * The backend collapses results to one row per meeting and sorts by
 * FTS5 BM25 rank ascending — *smaller is better* (negative numbers
 * are the strongest matches). The UI should preserve that ordering.
 *
 * `snippet` is pre-rendered with `<mark>...</mark>` markers around
 * the matched terms. Render it with `dangerouslySetInnerHTML` — the
 * markers are emitted by SQLite over our own indexed text, so the
 * XSS surface is the same as showing the segment body raw, which the
 * rest of the UI already does.
 */
export type MeetingSearchHit = {
  meeting: MeetingSummary;
  snippet: string;
  rank: number;
};

/**
 * Full-text search over segment text. Empty / whitespace-only queries
 * resolve to `[]` without hitting the backend index, so it's safe to
 * wire this up to a debounced `onChange`.
 *
 * @param query Raw user input. FTS5 syntax characters (`"`, `*`,
 *   `(`, `)`, `^`, `:`, `+`, `-`, `~`) are stripped server-side, so
 *   the caller does not need to escape anything.
 * @param limit Maximum hits to return. Defaults to 20 (a comfortable
 *   sidebar page); pass `0` for no cap.
 */
export async function searchMeetings(
  query: string,
  limit = 20,
): Promise<MeetingSearchHit[]> {
  return invoke<MeetingSearchHit[]>("search_meetings", { query, limit });
}
