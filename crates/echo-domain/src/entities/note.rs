//! Meeting note entity.
//!
//! A [`Note`] is a user-created text annotation attached to a meeting,
//! timestamped relative to the recording start. Users add notes while
//! a session is running so they can mark important moments without
//! interrupting the transcription flow.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::entities::meeting::MeetingId;

/// Strongly-typed identifier for a [`Note`]. UUIDv7 keeps lexical
/// ordering aligned with creation time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, specta::Type)]
#[serde(transparent)]
pub struct NoteId(pub Uuid);

impl NoteId {
    /// Generate a new UUIDv7 identifier.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }
}

impl Default for NoteId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for NoteId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A user-created text annotation within a meeting.
///
/// `timestamp_ms` is relative to the meeting start (not wall-clock),
/// matching the same timeline as [`Segment::start_ms`]. This allows
/// notes and transcript segments to be displayed together on a unified
/// timeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct Note {
    /// Stable identifier.
    pub id: NoteId,
    /// The meeting this note belongs to.
    pub meeting_id: MeetingId,
    /// User-written text content.
    pub text: String,
    /// Offset in milliseconds from the meeting start at which the
    /// note was created.
    pub timestamp_ms: u32,
    /// RFC 3339 instant the note was persisted (wall-clock, for
    /// auditing only — the functional timestamp is `timestamp_ms`).
    pub created_at: String,
}
