//! Meeting entity.
//!
//! A [`Meeting`] aggregates a recording session together with its
//! derived artifacts (segments, the negotiated audio format, the
//! detected language, …). Speakers, summary and chat history attach
//! later (Sprint 2 / Sprint 3) and live in their own tables; the
//! aggregate stays narrow on purpose so the storage adapter doesn't
//! need to load everything when listing.
//!
//! Meeting ids are UUIDv7 so the `(meeting_id, start_ms)` index used by
//! the SQLite adapter clusters rows by recency on disk.

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use uuid::Uuid;

use crate::entities::segment::Segment;
use crate::entities::speaker::Speaker;
use crate::ports::audio::AudioFormat;

/// Strongly-typed identifier for a [`Meeting`]. UUIDv7 keeps lexical
/// ordering aligned with creation time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MeetingId(pub Uuid);

impl MeetingId {
    /// Generate a new UUIDv7 identifier.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }
}

impl Default for MeetingId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for MeetingId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Lightweight projection used by listing endpoints. Holds only what
/// the UI needs to render a row in the sidebar — *not* the segments.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MeetingSummary {
    /// Stable identifier.
    pub id: MeetingId,
    /// Human-readable title. Defaults to `"Meeting <date>"` when the
    /// caller does not supply one.
    pub title: String,
    /// RFC 3339 instant the recording started.
    #[serde(with = "time::serde::rfc3339")]
    pub started_at: OffsetDateTime,
    /// RFC 3339 instant the recording stopped. `None` while the
    /// session is still running.
    #[serde(with = "time::serde::rfc3339::option")]
    pub ended_at: Option<OffsetDateTime>,
    /// Total audio captured, in milliseconds.
    pub duration_ms: u32,
    /// Dominant language reported by the ASR backend (mode of
    /// per-chunk languages). `None` until the first chunk lands.
    pub language: Option<String>,
    /// Number of segments persisted.
    pub segment_count: u32,
}

/// One hit returned by the search port. Carries the meeting summary
/// (so the sidebar can render it without an extra round-trip), the
/// FTS5 BM25 rank (smaller = better, by SQLite convention) and a
/// pre-rendered snippet around the strongest match. Snippet
/// boundaries are decided by SQLite's `snippet()` function and use
/// the markers chosen by the storage adapter — typically `<mark>` /
/// `</mark>` so the UI can render them as highlights.
///
/// `serde(rename_all = "camelCase")` keeps the wire format aligned
/// with the rest of the IPC surface.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MeetingSearchHit {
    /// Meeting matched by the query.
    pub meeting: MeetingSummary,
    /// Highlighted excerpt of the segment that matched.
    pub snippet: String,
    /// FTS5 BM25 rank — *smaller is better* (negative numbers are
    /// strongest matches). The UI should sort ascending; tests assert
    /// the same. We keep the raw value rather than mapping to a
    /// `[0,1]` score so consumers can decide their own normalisation.
    pub rank: f64,
}

/// Full meeting aggregate. Returned by point lookups and includes the
/// segment list and any diarized speakers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Meeting {
    /// Summary projection.
    #[serde(flatten)]
    pub summary: MeetingSummary,
    /// Audio format negotiated with the device.
    pub input_format: AudioFormat,
    /// Decoded segments, ordered by `start_ms`. Each segment may
    /// reference a `speaker_id` from `speakers`.
    pub segments: Vec<Segment>,
    /// Diarized speakers persisted for this meeting, ordered by
    /// `slot`. Empty when no diarizer was wired into the pipeline.
    /// `serde(default)` keeps older payloads (pre-Sprint 1 day 7)
    /// loadable without speakers.
    #[serde(default)]
    pub speakers: Vec<Speaker>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn meeting_id_default_uses_uuid_v7() {
        let id = MeetingId::default();
        let bytes = id.0.as_bytes();
        assert_eq!(
            bytes[6] >> 4,
            7,
            "expected uuid v7, got version byte {bytes:02x?}"
        );
    }

    #[test]
    fn summary_round_trips_through_serde_with_camelcase() {
        let s = MeetingSummary {
            id: MeetingId::new(),
            title: "Standup".into(),
            started_at: OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap(),
            ended_at: None,
            duration_ms: 5_000,
            language: Some("en".into()),
            segment_count: 3,
        };
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("\"startedAt\""), "got: {json}");
        assert!(json.contains("\"segmentCount\""), "got: {json}");
        let back: MeetingSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(back, s);
    }
}
