/**
 * Typed IPC client for the Tauri backend.
 *
 * This module is the only place in the frontend that calls
 * `@tauri-apps/api/core#invoke`. Higher layers (hooks, components)
 * import these wrappers, never `invoke` directly — that's how we
 * keep the adapter boundary intact (see src/README.md).
 *
 * Hand-rolled today; once the backend surface stabilises, type
 * generation will be delegated to `tauri-specta`, which will emit
 * the request/response shapes from Rust `#[specta::specta]` so this
 * file becomes a thin call-site index.
 */

import { Channel, invoke } from "@tauri-apps/api/core";

import type { HealthStatus } from "../types/health";
import type {
  Meeting,
  MeetingId,
  MeetingSearchHit,
  MeetingSummary,
} from "../types/meeting";
import type { SpeakerId } from "../types/speaker";
import type {
  StartStreamingOptions,
  StreamingSessionId,
  TranscriptEvent,
} from "../types/streaming";
import type { Summary } from "../types/summary";

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

export async function healthCheck(): Promise<HealthStatus> {
  return invoke<HealthStatus>("health_check");
}

// ---------------------------------------------------------------------------
// Streaming transcription
// ---------------------------------------------------------------------------

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
// Meetings
// ---------------------------------------------------------------------------

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
// Meeting search
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// LLM summaries (Sprint 1 day 9)
// ---------------------------------------------------------------------------

/**
 * Generate (or regenerate) the local-LLM summary for a meeting.
 *
 * Loads the LLM lazily on first call (the model file is ~4.4 GB for
 * Qwen 2.5 7B Q4_K_M, so the first invocation in a session is the
 * slow one). Subsequent calls reuse the cached model. The use case
 * upserts on (meetingId, template), so re-running this command on
 * the same meeting REPLACES the previous summary instead of
 * appending — matches the `Generate again` UX.
 *
 * Backend errors surface as plain strings ready for the toast
 * layer (`not found:`, `empty transcript:`, `llm:`, `storage:`).
 */
export async function summarizeMeeting(
  meetingId: MeetingId,
): Promise<Summary> {
  return invoke<Summary>("summarize_meeting", { meetingId });
}

/**
 * Fetch the most recent summary for a meeting, or `null` when none
 * has been generated yet. Wired into `MeetingDetail` mount so the
 * panel can decide between rendering an existing summary and the
 * "Generate" affordance without paying for a regenerate.
 */
export async function getSummary(
  meetingId: MeetingId,
): Promise<Summary | null> {
  return invoke<Summary | null>("get_summary", { meetingId });
}
