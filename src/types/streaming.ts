/**
 * Live streaming domain types.
 *
 * Mirrors `crates/echo-domain/src/entities/streaming.rs` and the
 * `start_streaming` / `stop_streaming` commands.
 */

import type { Segment } from "./meeting";
import type { SpeakerId } from "./speaker";

/** UUIDv7 string identifying a streaming session. */
export type StreamingSessionId = string;

export type AudioFormat = {
  sampleRateHz: number;
  channels: number;
};

/**
 * Where the backend should pull audio from.
 *
 * - `microphone`: default cpal input. Requires Microphone permission.
 * - `systemOutput`: system audio loopback (macOS 13+, ScreenCaptureKit).
 *   The backend ignores `deviceId` for this source.
 * - `mixed`: microphone + system audio merged. Best with headphones:
 *   mic captures the local speaker, system captures remote participants.
 *   Use `setMixSources` to mute/unmute each channel during a session.
 */
export type AudioSourceKind = "microphone" | "systemOutput" | "mixed";

/** Information about the system's current default audio output device. */
export type OutputDeviceInfo = {
  name: string;
  /** True when the device name suggests headphones / earbuds. */
  isHeadphones: boolean;
};

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
    }
  | {
      type: "paused";
      sessionId: StreamingSessionId;
    }
  | {
      type: "resumed";
      sessionId: StreamingSessionId;
    };
