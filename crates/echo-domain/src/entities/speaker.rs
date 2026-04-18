//! Speaker entity.
//!
//! A `Speaker` is the clustered identity of one voice in a meeting.
//! Speakers start anonymous (`remote_01`, `remote_02`) and may be
//! renamed by the user or matched to a participant hint.
//!
//! Sprint 0 day 6 only ships the [`SpeakerId`] newtype so [`Segment`]s
//! can reference it; the full entity (embedding, label, color) is
//! populated in Sprint 2 alongside the diarization adapter.
//!
//! [`Segment`]: crate::entities::segment::Segment

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Strongly-typed speaker identifier. UUIDv7 keeps insertion-time
/// ordering aligned with creation order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SpeakerId(pub Uuid);

impl SpeakerId {
    /// Generate a new UUIDv7 identifier.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }
}

impl Default for SpeakerId {
    fn default() -> Self {
        Self::new()
    }
}
