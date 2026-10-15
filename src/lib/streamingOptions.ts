import type { AudioSourceKind, StartStreamingOptions } from "../types/streaming";

/** What the user picked before starting a live session. */
export type StartOptions = {
  /** Whisper language hint. Empty string means "auto-detect". */
  language: string;
  /** Enable diarization for this session. */
  diarize: boolean;
  /** Keep this session's audio next to the meeting. */
  saveAudio: boolean;
  /** Audio capture source. Defaults to microphone. */
  source?: AudioSourceKind;
};

/**
 * Backend options for a live session. The language hint and a
 * non-default source are only sent when set, so the backend keeps its
 * own defaults otherwise.
 */
export function streamingOptionsFor({
  language,
  diarize,
  saveAudio,
  source,
}: StartOptions): StartStreamingOptions {
  const langHint = language.trim();
  return {
    chunkMs: 5_000,
    silenceRmsThreshold: 0.005,
    diarize,
    saveAudio,
    ...(langHint.length > 0 ? { language: langHint } : {}),
    ...(source && source !== "microphone" ? { source } : {}),
  };
}
