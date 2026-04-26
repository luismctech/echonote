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

import type { ChatEvent, ChatMessage } from "../types/chat";
import type { DownloadEvent, ModelInfo } from "../types/models";
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
 * @param meetingId Target meeting.
 * @param template Template id (`"general"`, `"oneOnOne"`, etc.).
 *   Defaults to `"general"` when omitted or `null`.
 */
export async function summarizeMeeting(
  meetingId: MeetingId,
  template?: string,
): Promise<Summary> {
  return invoke<Summary>("summarize_meeting", {
    meetingId,
    template: template ?? null,
  });
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

// ---------------------------------------------------------------------------
// Export (CU-08)
// ---------------------------------------------------------------------------

export type ExportFormat = "markdown" | "txt";

/**
 * Export a meeting (with its summary, if any) to a file.
 *
 * The caller is responsible for obtaining `destPath` via a save-file
 * dialog before invoking this.
 */
export async function exportMeeting(
  meetingId: MeetingId,
  format: ExportFormat,
  destPath: string,
): Promise<void> {
  return invoke<void>("export_meeting", {
    meetingId,
    format,
    destPath,
  });
}

// ---------------------------------------------------------------------------
// Model management
// ---------------------------------------------------------------------------

/** Get the status of all downloadable models. */
export async function getModelStatus(): Promise<ModelInfo[]> {
  return invoke<ModelInfo[]>("get_model_status");
}

/** Download a model, streaming progress events to the callback. */
export async function downloadModel(
  modelId: string,
  onEvent: (event: DownloadEvent) => void,
): Promise<void> {
  const channel = new Channel<DownloadEvent>();
  channel.onmessage = onEvent;
  return invoke<void>("download_model", { modelId, onEvent: channel });
}

// ---------------------------------------------------------------------------
// Chat with transcript (Sprint 1 day 10 — CU-05)
// ---------------------------------------------------------------------------

/**
 * Run one chat turn against a meeting's transcript. The backend
 * streams the assistant's reply token-by-token through the supplied
 * `onEvent` callback and resolves the promise once the stream
 * terminates (with a `finished` or `failed` event).
 *
 * **Cancellation** is implicit: when the React component unmounts
 * (or the user closes the chat panel), the `Channel` is dropped,
 * and the backend aborts the decode loop on the next channel send
 * failure. No explicit `cancelChat` command is needed.
 *
 * @param meetingId Target meeting whose transcript will be used as
 *   context for the model.
 * @param history Previous chat turns (excluding the system prompt,
 *   which is assembled by the backend from the transcript). Pass
 *   `[]` for the first turn.
 * @param question The current user message.
 * @param onEvent Callback that fires for every {@link ChatEvent}.
 */
export async function askAboutMeeting(
  meetingId: MeetingId,
  history: ChatMessage[],
  question: string,
  onEvent: (event: ChatEvent) => void,
): Promise<void> {
  const channel = new Channel<ChatEvent>();
  channel.onmessage = onEvent;
  return invoke<void>("ask_about_meeting", {
    meetingId,
    history: history.length > 0 ? history : null,
    question,
    onEvent: channel,
  });
}
