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
   * Keep this session's audio (16 kHz mono WAV) next to the meeting
   * database; it is deleted with the meeting. Defaults to `false`.
   */
  saveAudio?: boolean;
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
      /**
       * Every distinct speaker referenced by `segments` (a segment may
       * belong to a different speaker than the chunk-dominant one).
       * Omitted by older backends; treat `undefined` as empty.
       */
      speakers?: SpeakerRef[];
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
    }
  | {
      /**
       * Offline diarization refinement applied at end-of-session.
       * Carries `(absorbed, surviving)` SpeakerId pairs; the recorder
       * has already rewritten persisted segments and dropped absorbed
       * speaker rows by the time the frontend sees this. UI listeners
       * may use it to refresh any in-memory speakers list they kept.
       */
      type: "speakersMerged";
      sessionId: StreamingSessionId;
      merges: SpeakerMerge[];
    }
  | {
      /**
       * Offline re-segmentation applied at end-of-session (ADR 0001).
       * Ids, slots and indices only; every field is a set-to-value, so
       * applying it twice is harmless. The recorder has already
       * persisted it when the frontend sees it. Emitted after the last
       * `chunk` and before `stopped`, only when a segment moved.
       */
      type: "speakersReassigned";
      sessionId: StreamingSessionId;
      /** UUIDv7 of the plan; lets the UI detect a replay. */
      planId: string;
      created: SpeakerRef[];
      retired: SpeakerRetirement[];
      segments: SegmentSpeaker[];
      chunks: ChunkSpeaker[];
    };

/** A speaker the refine left without segments, and who took its speech. */
export type SpeakerRetirement = {
  speaker: SpeakerId;
  majorityRecipient: SpeakerId | null;
};

/** New speaker of one already-emitted segment. */
export type SegmentSpeaker = {
  segmentId: string;
  speakerId: SpeakerId | null;
};

/** New dominant speaker of one already-emitted chunk (live pane). */
export type ChunkSpeaker = {
  chunkIndex: number;
  speakerId: SpeakerId | null;
  speakerSlot: number | null;
};

/** A speaker referenced by a chunk: stable id plus arrival-order slot. */
export type SpeakerRef = {
  id: SpeakerId;
  slot: number;
};

/**
 * One offline-refinement merge: `absorbed` is collapsed into
 * `surviving`. Persisted segments referring to `absorbed` are
 * rewritten to point at `surviving`; the `absorbed` speaker row is
 * dropped from the meeting.
 */
export type SpeakerMerge = {
  absorbed: SpeakerId;
  surviving: SpeakerId;
};
