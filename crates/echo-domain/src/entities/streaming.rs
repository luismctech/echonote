//! Streaming-pipeline value objects.
//!
//! These types are produced by the application layer's
//! `StreamingPipeline` and crossed across every IPC boundary
//! (Tauri → React, CLI → stdout, future test harnesses). They sit in
//! `echo-domain` so every layer agrees on the wire shape without
//! pulling in framework dependencies.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::entities::segment::Segment;
use crate::entities::speaker::SpeakerId;
use crate::ports::audio::AudioFormat;

/// Identifier of a single streaming session. UUIDv7 keeps insertion
/// order ≈ creation order, useful for logs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, specta::Type)]
#[serde(transparent)]
pub struct StreamingSessionId(pub Uuid);

impl StreamingSessionId {
    /// Generate a new UUIDv7 id.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }
}

impl Default for StreamingSessionId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for StreamingSessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Knobs the caller can pass when starting a streaming pipeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, specta::Type)]
pub struct StreamingOptions {
    /// ISO-639-1 language hint for the ASR backend. `None` ⇒ auto.
    pub language: Option<String>,
    /// Maximum audio buffered before forcing a transcribe pass, in
    /// milliseconds. Default 5000 (5 s) — Whisper's sweet spot.
    pub chunk_ms: u32,
    /// Skip transcription of chunks whose RMS is below this threshold.
    /// `0.0` disables the gate. Default 0.02 (~ -34 dBFS), aligned
    /// with [`crate::EnergyVad`]'s `start_threshold` which was tuned
    /// for desk microphones in quiet-ish rooms (laptop coffeeshop /
    /// home office). A typical MacBook mic in a quiet room sits at
    /// 0.005–0.015 RMS when nobody is speaking; conversational speech
    /// at ~ -20 dBFS clocks in around 0.05–0.1 RMS. The previous
    /// defaults (0.005, 0.01) were too permissive and let background
    /// noise through, which fed Whisper near-silent chunks it then
    /// hallucinated YouTube outros / single-word "Gracias." over. If
    /// your mic environment is louder, raise it; if you have a soft
    /// speaker or a noise-cancelling headset, lower it.
    pub silence_rms_threshold: f32,
}

impl Default for StreamingOptions {
    fn default() -> Self {
        Self {
            language: None,
            chunk_ms: 5_000,
            silence_rms_threshold: 0.02,
        }
    }
}

/// One event emitted by the streaming pipeline.
///
/// The variant order on the wire intentionally matches the lifecycle:
/// `Started` → 0..N × `Chunk` → optional `Skipped` mixed in →
/// `Stopped` | `Failed`.
///
/// Wire format: tagged with `type` (lowercase) and field names in
/// `camelCase` for ergonomic consumption from TypeScript.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase", tag = "type")]
pub enum TranscriptEvent {
    /// Capture has begun. Reports the negotiated input format.
    #[serde(rename_all = "camelCase")]
    Started {
        /// Session id this event belongs to.
        session_id: StreamingSessionId,
        /// Format actually negotiated with the device.
        input_format: AudioFormat,
    },
    /// One transcribed chunk. Segments inside are timestamped relative
    /// to the start of the session (not the chunk).
    #[serde(rename_all = "camelCase")]
    Chunk {
        /// Session id.
        session_id: StreamingSessionId,
        /// 0-based chunk index in arrival order.
        chunk_index: u32,
        /// Offset of the chunk start, in milliseconds since `Started`.
        offset_ms: u32,
        /// Decoded segments, may be empty.
        segments: Vec<Segment>,
        /// Detected language for the chunk (if the backend reported it).
        language: Option<String>,
        /// Real-time factor for this chunk
        /// (`asr_elapsed / audio_duration`). Lower is faster.
        rtf: f32,
        /// Speaker the diarizer assigned to this chunk. `None` when no
        /// diarizer is wired into the pipeline (default), or when the
        /// chunk is too short to embed reliably. Persisted alongside
        /// every segment in the chunk.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        speaker_id: Option<SpeakerId>,
        /// Arrival-order slot of the assigned speaker, mirrored from
        /// `speaker_id` for the convenience of the UI palette (no
        /// need to round-trip through the speakers list to colour a
        /// chip). 0-based; `None` whenever `speaker_id` is `None`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        speaker_slot: Option<u32>,
    },
    /// A chunk was discarded by the silence gate before being sent to
    /// the ASR backend. Useful for UI / metrics.
    #[serde(rename_all = "camelCase")]
    Skipped {
        /// Session id.
        session_id: StreamingSessionId,
        /// 0-based chunk index in arrival order.
        chunk_index: u32,
        /// Offset of the skipped chunk, in milliseconds since `Started`.
        offset_ms: u32,
        /// Length of the skipped chunk in milliseconds.
        duration_ms: u32,
        /// RMS of the skipped chunk (for diagnostics).
        rms: f32,
    },
    /// The pipeline finished cleanly (caller stopped or stream EOF).
    #[serde(rename_all = "camelCase")]
    Stopped {
        /// Session id.
        session_id: StreamingSessionId,
        /// Total transcribed segments emitted across all chunks.
        total_segments: u32,
        /// Total wall-clock audio captured, in milliseconds.
        total_audio_ms: u32,
    },
    /// The pipeline aborted with an error. No further events follow.
    #[serde(rename_all = "camelCase")]
    Failed {
        /// Session id.
        session_id: StreamingSessionId,
        /// Human-readable error message. The structured cause is logged
        /// in the backend and not exposed across the wire.
        message: String,
    },
}

impl TranscriptEvent {
    /// Session id this event belongs to. Convenient for routing in the UI.
    #[must_use]
    pub fn session_id(&self) -> StreamingSessionId {
        match self {
            Self::Started { session_id, .. }
            | Self::Chunk { session_id, .. }
            | Self::Skipped { session_id, .. }
            | Self::Stopped { session_id, .. }
            | Self::Failed { session_id, .. } => *session_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn defaults_are_whisper_friendly() {
        let opts = StreamingOptions::default();
        assert_eq!(opts.chunk_ms, 5_000);
        assert!(opts.silence_rms_threshold > 0.0);
        assert!(opts.language.is_none());
    }

    #[test]
    fn event_serializes_with_kebab_friendly_camelcase_tag() {
        let id = StreamingSessionId::new();
        let evt = TranscriptEvent::Started {
            session_id: id,
            input_format: AudioFormat::WHISPER,
        };
        let json = serde_json::to_string(&evt).unwrap();
        assert!(json.contains("\"type\":\"started\""), "got: {json}");
        assert!(json.contains("\"sessionId\""), "got: {json}");
        assert!(json.contains("\"inputFormat\""), "got: {json}");
        // Regression: nested AudioFormat must also be camelCase or the
        // React `LivePane` subtitle renders "undefined Hz". See bug
        // fixed alongside the Silero VAD work.
        assert!(json.contains("\"sampleRateHz\""), "got: {json}");
        assert!(
            !json.contains("\"sample_rate_hz\""),
            "AudioFormat leaked snake_case to the wire: {json}"
        );
    }

    #[test]
    fn session_id_is_extractable_from_any_variant() {
        let id = StreamingSessionId::new();
        let evt = TranscriptEvent::Stopped {
            session_id: id,
            total_segments: 3,
            total_audio_ms: 12_000,
        };
        assert_eq!(evt.session_id(), id);
    }

    #[test]
    fn chunk_without_speaker_omits_field_on_the_wire() {
        let evt = TranscriptEvent::Chunk {
            session_id: StreamingSessionId::new(),
            chunk_index: 0,
            offset_ms: 0,
            segments: vec![],
            language: None,
            rtf: 0.1,
            speaker_id: None,
            speaker_slot: None,
        };
        let json = serde_json::to_string(&evt).unwrap();
        assert!(
            !json.contains("speakerId"),
            "speakerId leaked when None: {json}"
        );
        assert!(
            !json.contains("speakerSlot"),
            "speakerSlot leaked when None: {json}"
        );
    }

    #[test]
    fn chunk_with_speaker_serialises_both_fields() {
        use crate::entities::speaker::SpeakerId;
        let speaker = SpeakerId::new();
        let evt = TranscriptEvent::Chunk {
            session_id: StreamingSessionId::new(),
            chunk_index: 2,
            offset_ms: 10_000,
            segments: vec![],
            language: None,
            rtf: 0.2,
            speaker_id: Some(speaker),
            speaker_slot: Some(1),
        };
        let json = serde_json::to_string(&evt).unwrap();
        assert!(json.contains("\"speakerId\""), "got: {json}");
        assert!(json.contains("\"speakerSlot\":1"), "got: {json}");
    }

    #[test]
    fn chunk_without_speaker_fields_deserialises_via_serde_default() {
        // Older payloads (pre-Sprint 1 day 7) did not carry speaker
        // info; ensure they still round-trip through the new schema.
        let id = StreamingSessionId::new();
        let json = serde_json::json!({
            "type": "chunk",
            "sessionId": id,
            "chunkIndex": 0,
            "offsetMs": 0,
            "segments": [],
            "language": null,
            "rtf": 0.0,
        })
        .to_string();
        let parsed: TranscriptEvent = serde_json::from_str(&json).unwrap();
        match parsed {
            TranscriptEvent::Chunk {
                speaker_id,
                speaker_slot,
                ..
            } => {
                assert!(speaker_id.is_none());
                assert!(speaker_slot.is_none());
            }
            other => panic!("expected Chunk, got {other:?}"),
        }
    }
}
