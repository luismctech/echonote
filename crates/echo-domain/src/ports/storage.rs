//! Storage port.
//!
//! The application layer persists meetings (header + segments) through
//! [`MeetingStore`]. The adapter (`echo-storage::SqliteMeetingStore`)
//! lives behind this trait so the domain stays free of `sqlx` knowledge.
//!
//! Concurrency contract: implementations MUST be safe to share across
//! tasks (`Send + Sync`) and serialize writes per `meeting_id` so that
//! `append_segments` calls from the streaming pipeline observe a
//! monotonically growing transcript.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::entities::meeting::{Meeting, MeetingId, MeetingSummary};
use crate::entities::segment::Segment;
use crate::ports::audio::AudioFormat;
use crate::DomainError;

/// Initial parameters used when opening a new meeting record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CreateMeeting {
    /// Stable identifier. Caller-supplied so the streaming pipeline can
    /// reuse its session id across the wire.
    pub id: MeetingId,
    /// Display title shown in the sidebar.
    pub title: String,
    /// Format negotiated with the capture device.
    pub input_format: AudioFormat,
}

/// Patch applied by `stop_recording` once the session ends. Every
/// field is optional so callers can also use this to update metadata
/// without finalizing.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FinalizeMeeting {
    /// RFC 3339 instant the recording ended. Set on stop.
    pub ended_at_rfc3339: Option<String>,
    /// Total audio captured, in milliseconds.
    pub duration_ms: Option<u32>,
    /// Dominant language inferred across chunks.
    pub language: Option<String>,
    /// New segment count after the final flush, if it changed.
    pub segment_count: Option<u32>,
}

/// Persistence port for meetings + their segments.
#[async_trait]
pub trait MeetingStore: Send + Sync {
    /// Create a new, empty meeting row. Returns the freshly inserted
    /// summary.
    async fn create(&self, params: CreateMeeting) -> Result<MeetingSummary, DomainError>;

    /// Append segments to an existing meeting in a single transaction.
    /// Implementations MUST de-duplicate by `Segment::id` so re-tries
    /// from the streaming pipeline are idempotent.
    async fn append_segments(
        &self,
        meeting_id: MeetingId,
        segments: &[Segment],
    ) -> Result<(), DomainError>;

    /// Update the meeting header. Used by `stop_recording` to mark a
    /// session as ended.
    async fn finalize(
        &self,
        meeting_id: MeetingId,
        patch: FinalizeMeeting,
    ) -> Result<MeetingSummary, DomainError>;

    /// List meetings ordered by `started_at` descending. `limit`
    /// caps the result; `0` means "no cap".
    async fn list(&self, limit: u32) -> Result<Vec<MeetingSummary>, DomainError>;

    /// Return the full meeting (with segments) or `None` when the id
    /// is unknown.
    async fn get(&self, meeting_id: MeetingId) -> Result<Option<Meeting>, DomainError>;

    /// Delete a meeting and its segments. Returns `false` when the id
    /// did not exist.
    async fn delete(&self, meeting_id: MeetingId) -> Result<bool, DomainError>;
}
