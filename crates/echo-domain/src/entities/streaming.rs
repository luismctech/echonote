//! Streaming-pipeline value objects.
//!
//! These types are produced by the application layer's
//! `StreamingPipeline` and crossed across every IPC boundary
//! (Tauri → React, CLI → stdout, future test harnesses). They sit in
//! `echo-domain` so every layer agrees on the wire shape without
//! pulling in framework dependencies.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::entities::segment::{Segment, SegmentId};
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
    /// Fixed chunk length in milliseconds, used when no VAD is attached
    /// or `speech_aligned_chunks` is `false`. Default 5000 (5 s).
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
    /// Cut chunks at speech pauses instead of every `chunk_ms`. Only
    /// takes effect when a VAD is attached to the pipeline; without one
    /// the pipeline always uses fixed `chunk_ms` chunks and the RMS
    /// gate. Default `true`: Whisper is far more accurate on whole
    /// utterances than on fixed chunks that cut words.
    #[serde(default = "default_speech_aligned_chunks")]
    pub speech_aligned_chunks: bool,
    /// Speech-aligned mode: shortest chunk cut at a speech pause, in
    /// milliseconds, unless the utterance is followed by silence.
    /// Default 4000 (tuned in `docs/spikes/2026-10-vad-chunking-bench.md`).
    #[serde(default = "default_min_chunk_ms")]
    pub min_chunk_ms: u32,
    /// Speech-aligned mode: longest chunk, in milliseconds. Without a
    /// pause by then the chunk is cut at the quietest point shortly
    /// before it. Must be `>= min_chunk_ms`. Default 20000 (Whisper
    /// decodes up to 30 s windows).
    #[serde(default = "default_max_chunk_ms")]
    pub max_chunk_ms: u32,
    /// Raise quiet capture to a normal speech level before anything
    /// else sees it (VAD, ASR, diarizer, saved audio). The gain follows
    /// the speech level over the last seconds, never attenuates, is
    /// capped at about +30 dB and ignores steady noise, so loud input
    /// passes through unchanged. Default `true`: a mic at -54 dBFS made
    /// the VAD drop real speech as silence.
    #[serde(default = "default_auto_gain")]
    pub auto_gain: bool,
    /// Speech-aligned mode: while a chunk is still open, transcribe what
    /// it holds so far every this many milliseconds of new audio and
    /// emit it as a [`TranscriptEvent::Draft`], so text shows up long
    /// before the chunk closes at a pause. Drafts never change the
    /// final chunks and are skipped while transcription runs behind
    /// capture. `0` disables them; fixed chunks never draft. Default
    /// 4000.
    #[serde(default = "default_draft_interval_ms")]
    pub draft_interval_ms: u32,
}

/// Default for [`StreamingOptions::speech_aligned_chunks`].
const DEFAULT_SPEECH_ALIGNED_CHUNKS: bool = true;
/// Default for [`StreamingOptions::auto_gain`].
const DEFAULT_AUTO_GAIN: bool = true;
/// Default for [`StreamingOptions::draft_interval_ms`].
const DEFAULT_DRAFT_INTERVAL_MS: u32 = 4_000;
/// Default for [`StreamingOptions::min_chunk_ms`].
const DEFAULT_MIN_CHUNK_MS: u32 = 4_000;
/// Default for [`StreamingOptions::max_chunk_ms`].
const DEFAULT_MAX_CHUNK_MS: u32 = 20_000;

fn default_speech_aligned_chunks() -> bool {
    DEFAULT_SPEECH_ALIGNED_CHUNKS
}

fn default_auto_gain() -> bool {
    DEFAULT_AUTO_GAIN
}

fn default_draft_interval_ms() -> u32 {
    DEFAULT_DRAFT_INTERVAL_MS
}

fn default_min_chunk_ms() -> u32 {
    DEFAULT_MIN_CHUNK_MS
}

fn default_max_chunk_ms() -> u32 {
    DEFAULT_MAX_CHUNK_MS
}

impl Default for StreamingOptions {
    fn default() -> Self {
        Self {
            language: None,
            chunk_ms: 5_000,
            silence_rms_threshold: 0.02,
            speech_aligned_chunks: DEFAULT_SPEECH_ALIGNED_CHUNKS,
            min_chunk_ms: DEFAULT_MIN_CHUNK_MS,
            max_chunk_ms: DEFAULT_MAX_CHUNK_MS,
            auto_gain: DEFAULT_AUTO_GAIN,
            draft_interval_ms: DEFAULT_DRAFT_INTERVAL_MS,
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
        /// Every distinct speaker referenced by `segments` (each
        /// segment may belong to a different speaker than the
        /// chunk-dominant one above), so consumers can persist them
        /// before the segments. Defaults to empty so payloads from
        /// older producers still deserialise.
        #[serde(default)]
        speakers: Vec<SpeakerRef>,
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
    /// Offline diarization refinement after the streaming loop ended.
    ///
    /// Carries `(absorbed_id, surviving_id)` pairs the storage layer
    /// must apply to already-persisted segments: every reference to
    /// `absorbed_id` is rewritten to `surviving_id`, and the absorbed
    /// speaker row is deleted. Always emitted *before* [`Stopped`] so
    /// consumers can finalise the meeting with the correct speaker
    /// count. Empty `merges` is allowed — the recorder treats it as a
    /// no-op.
    #[serde(rename_all = "camelCase")]
    SpeakersMerged {
        /// Session id.
        session_id: StreamingSessionId,
        /// `(absorbed, surviving)` pairs produced by the diarizer's
        /// offline refinement pass.
        merges: Vec<SpeakerMerge>,
    },
    /// Offline re-segmentation after the streaming loop ended (ADR 0001).
    ///
    /// Moves already-emitted segments between speakers, creates
    /// speakers and retires speakers left without segments. Emitted
    /// after the last [`Chunk`](Self::Chunk) and before
    /// [`Stopped`](Self::Stopped), only when at least one segment
    /// changes speaker.
    ///
    /// Every field is a set-to-value, never a delta, so applying the
    /// event twice gives the same state. It carries ids, slots and
    /// indices only: no embeddings, scores or text.
    #[serde(rename_all = "camelCase")]
    SpeakersReassigned {
        /// Session id.
        session_id: StreamingSessionId,
        /// UUIDv7 of this plan; lets consumers detect a replay.
        plan_id: Uuid,
        /// Speakers the plan introduces. Persist them before `segments`.
        created: Vec<SpeakerRef>,
        /// Speakers left without segments.
        retired: Vec<SpeakerRetirement>,
        /// Segments whose speaker changed, with their new speaker.
        segments: Vec<SegmentSpeaker>,
        /// Chunks whose dominant speaker changed (live pane).
        chunks: Vec<ChunkSpeaker>,
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
    /// The pipeline was paused. Audio frames are discarded until resumed.
    #[serde(rename_all = "camelCase")]
    Paused {
        /// Session id.
        session_id: StreamingSessionId,
    },
    /// The pipeline was resumed after a pause.
    #[serde(rename_all = "camelCase")]
    Resumed {
        /// Session id.
        session_id: StreamingSessionId,
    },
    /// How far transcription runs behind capture. Emitted after every
    /// chunk or skipped span, so the UI can tell a slow transcriber
    /// (growing lag, no audio lost) apart from a stalled one. Not
    /// persisted.
    #[serde(rename_all = "camelCase")]
    Progress {
        /// Session id.
        session_id: StreamingSessionId,
        /// Active (non-paused) capture time minus the audio the
        /// pipeline has taken in (emitted or buffered for the open
        /// chunk), in milliseconds; never negative. An open speech chunk
        /// waiting for its pause is not lag.
        lag_ms: u32,
        /// Captured audio queued for the pipeline but not yet buffered
        /// for chunking, in milliseconds.
        backlog_ms: u32,
    },
    /// Provisional text of the speech chunk still open at the end of
    /// the captured audio (speech-aligned mode only). Each draft
    /// replaces the previous one; the [`TranscriptEvent::Chunk`] with
    /// the same `chunk_index` replaces it for good. Drafts carry no
    /// speaker, timestamps or segments and are never persisted.
    #[serde(rename_all = "camelCase")]
    Draft {
        /// Session id.
        session_id: StreamingSessionId,
        /// Index the final chunk of this speech will be emitted with.
        chunk_index: u32,
        /// Session-time start of that chunk, in milliseconds since
        /// `Started`.
        offset_ms: u32,
        /// Text transcribed so far, punctuated like final segments.
        text: String,
    },
}

/// A speaker referenced by a [`TranscriptEvent::Chunk`]: its stable id
/// plus the arrival-order slot the recorder persists it under.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerRef {
    /// Speaker identifier, as referenced by `Segment::speaker_id`.
    pub id: SpeakerId,
    /// Arrival-order slot (0-based) of the speaker.
    pub slot: u32,
}

/// One offline-refinement merge: `absorbed` is collapsed into
/// `surviving`. Persisted segments referring to `absorbed` must be
/// rewritten to point at `surviving`; the `absorbed` speaker row is
/// dropped from the meeting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerMerge {
    /// Speaker the offline pass decided was a duplicate.
    pub absorbed: SpeakerId,
    /// Speaker the absorbed one is folded into.
    pub surviving: SpeakerId,
}

/// A speaker the offline pass left without segments, and the speaker
/// that took most of its speech. The recipient may inherit its user
/// label (ADR 0001, Q1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerRetirement {
    /// Retired speaker.
    pub speaker: SpeakerId,
    /// Speaker that received the majority of its duration, if any.
    pub majority_recipient: Option<SpeakerId>,
}

/// New speaker of one already-emitted segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct SegmentSpeaker {
    /// Segment id, as emitted in a `Chunk` event.
    pub segment_id: SegmentId,
    /// Its speaker after the refine; `None` leaves it unattributed.
    pub speaker_id: Option<SpeakerId>,
}

/// New dominant speaker of one already-emitted chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct ChunkSpeaker {
    /// Chunk index, as emitted in a `Chunk` event.
    pub chunk_index: u32,
    /// Dominant speaker after the refine.
    pub speaker_id: Option<SpeakerId>,
    /// Arrival-order slot of `speaker_id`.
    pub speaker_slot: Option<u32>,
}

impl TranscriptEvent {
    /// Session id this event belongs to. Convenient for routing in the UI.
    #[must_use]
    pub fn session_id(&self) -> StreamingSessionId {
        match self {
            Self::Started { session_id, .. }
            | Self::Chunk { session_id, .. }
            | Self::Skipped { session_id, .. }
            | Self::SpeakersMerged { session_id, .. }
            | Self::SpeakersReassigned { session_id, .. }
            | Self::Stopped { session_id, .. }
            | Self::Failed { session_id, .. }
            | Self::Paused { session_id, .. }
            | Self::Resumed { session_id, .. }
            | Self::Progress { session_id, .. }
            | Self::Draft { session_id, .. } => *session_id,
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
        assert!(opts.speech_aligned_chunks);
        assert_eq!((opts.min_chunk_ms, opts.max_chunk_ms), (4_000, 20_000));
        assert!(opts.auto_gain);
        assert_eq!(opts.draft_interval_ms, 4_000);
    }

    #[test]
    fn options_without_chunk_bounds_deserialize_with_defaults() {
        let json = r#"{"language":null,"chunk_ms":5000,"silence_rms_threshold":0.02}"#;
        let opts: StreamingOptions = serde_json::from_str(json).unwrap();
        assert_eq!(opts, StreamingOptions::default());
    }

    #[test]
    fn progress_serializes_lag_and_backlog_in_camel_case() {
        let evt = TranscriptEvent::Progress {
            session_id: StreamingSessionId::new(),
            lag_ms: 12_000,
            backlog_ms: 3_500,
        };
        let json = serde_json::to_value(&evt).unwrap();
        assert_eq!(json["type"], "progress");
        assert_eq!(json["lagMs"], 12_000);
        assert_eq!(json["backlogMs"], 3_500);
        let back: TranscriptEvent = serde_json::from_value(json).unwrap();
        assert_eq!(back, evt);
    }

    #[test]
    fn draft_serializes_in_camel_case_and_round_trips() {
        let evt = TranscriptEvent::Draft {
            session_id: StreamingSessionId::new(),
            chunk_index: 3,
            offset_ms: 12_000,
            text: "hola a todos".into(),
        };
        let json = serde_json::to_value(&evt).unwrap();
        assert_eq!(json["type"], "draft");
        assert_eq!(json["chunkIndex"], 3);
        assert_eq!(json["offsetMs"], 12_000);
        assert_eq!(json["text"], "hola a todos");
        let back: TranscriptEvent = serde_json::from_value(json).unwrap();
        assert_eq!(back, evt);
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

    fn reassigned_event() -> TranscriptEvent {
        let (a, b) = (SpeakerId::new(), SpeakerId::new());
        TranscriptEvent::SpeakersReassigned {
            session_id: StreamingSessionId::new(),
            plan_id: Uuid::now_v7(),
            created: vec![SpeakerRef { id: b, slot: 4 }],
            retired: vec![SpeakerRetirement {
                speaker: SpeakerId::new(),
                majority_recipient: Some(a),
            }],
            segments: vec![SegmentSpeaker {
                segment_id: SegmentId::new(),
                speaker_id: Some(b),
            }],
            chunks: vec![ChunkSpeaker {
                chunk_index: 2,
                speaker_id: Some(b),
                speaker_slot: Some(4),
            }],
        }
    }

    #[test]
    fn speakers_reassigned_round_trips_in_camel_case() {
        let evt = reassigned_event();
        let json = serde_json::to_string(&evt).unwrap();
        for key in [
            "\"type\":\"speakersReassigned\"",
            "\"planId\"",
            "\"majorityRecipient\"",
            "\"segmentId\"",
            "\"chunkIndex\":2",
            "\"speakerSlot\":4",
        ] {
            assert!(json.contains(key), "missing {key}: {json}");
        }
        let back: TranscriptEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back, evt);
    }

    #[test]
    fn speakers_reassigned_carries_no_float_arrays() {
        // Privacy (ADR 0001): ids, slots and indices only, never
        // embeddings or scores.
        let json = serde_json::to_value(reassigned_event()).unwrap();
        fn has_float(v: &serde_json::Value) -> bool {
            match v {
                serde_json::Value::Number(n) => n.is_f64() && !n.is_u64() && !n.is_i64(),
                serde_json::Value::Array(a) => a.iter().any(has_float),
                serde_json::Value::Object(o) => o.values().any(has_float),
                _ => false,
            }
        }
        assert!(!has_float(&json), "float leaked: {json}");
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
            speakers: vec![],
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
            speakers: vec![],
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

    #[test]
    fn chunk_without_speakers_field_still_deserialises() {
        // Payload produced by a pre-`speakers` build (older frontend,
        // MCP sidecar, persisted fixture).
        let json = r#"{
            "type": "chunk",
            "sessionId": "0190a000-0000-7000-8000-000000000000",
            "chunkIndex": 3,
            "offsetMs": 1500,
            "segments": [],
            "language": "en",
            "rtf": 0.1
        }"#;
        let evt: TranscriptEvent = serde_json::from_str(json).expect("old payload");
        match evt {
            TranscriptEvent::Chunk { speakers, .. } => assert!(speakers.is_empty()),
            other => panic!("expected Chunk, got {other:?}"),
        }
    }

    #[test]
    fn chunk_speakers_round_trip_in_camel_case() {
        let id = SpeakerId::new();
        let evt = TranscriptEvent::Chunk {
            session_id: StreamingSessionId::new(),
            chunk_index: 0,
            offset_ms: 0,
            segments: vec![],
            language: None,
            rtf: 0.0,
            speaker_id: Some(id),
            speaker_slot: Some(0),
            speakers: vec![SpeakerRef { id, slot: 0 }],
        };
        let json = serde_json::to_value(&evt).unwrap();
        assert_eq!(json["speakers"][0]["slot"], 0);
        assert_eq!(
            json["speakers"][0]["id"],
            serde_json::json!(id.0.to_string())
        );
        let back: TranscriptEvent = serde_json::from_value(json).unwrap();
        assert_eq!(back, evt);
    }
}
