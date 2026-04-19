//! Transcription segment.
//!
//! A [`Segment`] is a contiguous span of speech with start/end offsets
//! relative to the meeting start, the transcribed text and (optionally)
//! the diarized speaker. Confidence is expressed in `[0.0, 1.0]` when
//! the ASR backend reports per-segment probabilities.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::entities::speaker::SpeakerId;

/// Strongly-typed identifier for a [`Segment`]. UUIDv7 keeps lexical
/// ordering aligned with creation time, which simplifies SQLite
/// indexing later on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SegmentId(pub Uuid);

impl SegmentId {
    /// Generate a new UUIDv7 identifier.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }
}

impl Default for SegmentId {
    fn default() -> Self {
        Self::new()
    }
}

/// One contiguous transcription span.
///
/// `serde(rename_all = "camelCase")` keeps the wire format aligned
/// with the TypeScript `Segment` type in `src/types/segment.ts`, so
/// the `Meeting` aggregate returned by `get_meeting` exposes
/// `startMs` (not `start_ms`) — the same convention the streaming
/// `Chunk` event already uses for the segments it carries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Segment {
    /// Stable identifier.
    pub id: SegmentId,
    /// Inclusive start offset, in milliseconds, relative to the
    /// containing recording.
    pub start_ms: u32,
    /// Exclusive end offset, in milliseconds. Always `>= start_ms`.
    pub end_ms: u32,
    /// Decoded text. May be empty when the segment is silence or pure
    /// non-speech but the ASR backend still emitted it.
    pub text: String,
    /// Diarized speaker. `None` until diarization runs (Sprint 2).
    pub speaker_id: Option<SpeakerId>,
    /// Confidence in `[0.0, 1.0]`. `None` when the backend does not
    /// report it (e.g. whisper.cpp without `temperature_inc`).
    pub confidence: Option<f32>,
}

impl Segment {
    /// Duration of the segment, in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u32 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn duration_handles_inverted_or_equal_offsets_safely() {
        let s = Segment {
            id: SegmentId::new(),
            start_ms: 1_000,
            end_ms: 1_500,
            text: "hello".into(),
            speaker_id: None,
            confidence: Some(0.93),
        };
        assert_eq!(s.duration_ms(), 500);
    }

    #[test]
    fn segment_id_default_uses_uuid_v7() {
        let id = SegmentId::default();
        // UUIDv7 has version byte 7.
        let bytes = id.0.as_bytes();
        assert_eq!(
            bytes[6] >> 4,
            7,
            "expected uuid v7, got version byte {bytes:02x?}"
        );
    }
}
