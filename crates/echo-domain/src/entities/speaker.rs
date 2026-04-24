//! Speaker entity.
//!
//! A `Speaker` is the clustered identity of one voice in a meeting.
//! Speakers start anonymous (numbered by arrival order — `Speaker 1`,
//! `Speaker 2`, …) and can later be renamed by the user or matched to
//! a participant hint.
//!
//! ## What "speaker" means in EchoNote
//!
//! - **Per-meeting scope.** Two recordings on different days may map
//!   "Alice" to different `SpeakerId`s. Cross-meeting identity is a
//!   separate concern (Sprint 4 — speaker enrollment).
//! - **Per-track for now.** The Sprint 1 MVP clusters microphone and
//!   system-output independently. A speaker that talks on both sides
//!   ends up with two ids; cross-track unification is a follow-up.
//! - **Stable color via slot.** UI assigns a color from a fixed
//!   palette indexed by [`Speaker::slot`], so the visual identity
//!   persists across renames and across reloads of the same meeting.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Strongly-typed speaker identifier. UUIDv7 keeps insertion-time
/// ordering aligned with creation order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, specta::Type)]
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

impl std::fmt::Display for SpeakerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// One clustered voice within a meeting.
///
/// `slot` is the 0-based index of the speaker within the meeting,
/// assigned in arrival order. It drives the UI palette and the
/// default display name; it is **not** the database primary key
/// (that's [`SpeakerId`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, specta::Type)]
pub struct Speaker {
    /// Stable identifier.
    pub id: SpeakerId,
    /// 0-based arrival order within the meeting. Drives the default
    /// display name (`Speaker 1` for slot 0, etc.) and the color
    /// palette index in the UI.
    pub slot: u32,
    /// User-assigned label, if any. `None` ⇒ render as
    /// "Speaker {slot+1}". Limited to 80 characters at the storage
    /// layer; the domain itself does not impose a length cap.
    pub label: Option<String>,
}

impl Speaker {
    /// Build an anonymous speaker for a slot. The `id` is freshly
    /// allocated; pass an explicit one if persistence already has
    /// the row.
    #[must_use]
    pub fn anonymous(slot: u32) -> Self {
        Self {
            id: SpeakerId::new(),
            slot,
            label: None,
        }
    }

    /// Human-readable name. Returns the user-supplied label when
    /// present, otherwise a deterministic `Speaker {slot+1}` so the
    /// UI never has to deal with missing names.
    #[must_use]
    pub fn display_name(&self) -> String {
        match &self.label {
            Some(name) if !name.trim().is_empty() => name.clone(),
            _ => format!("Speaker {}", self.slot + 1),
        }
    }

    /// Returns a copy with the label rewritten. Trims whitespace and
    /// collapses empty strings to `None` so the UI can offer a
    /// "clear name" affordance via empty input.
    #[must_use]
    pub fn renamed(mut self, label: impl Into<String>) -> Self {
        let trimmed = label.into().trim().to_string();
        self.label = if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        };
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn anonymous_default_name_uses_slot_plus_one() {
        let s = Speaker::anonymous(0);
        assert_eq!(s.display_name(), "Speaker 1");
        let s = Speaker::anonymous(4);
        assert_eq!(s.display_name(), "Speaker 5");
    }

    #[test]
    fn label_overrides_default_name() {
        let s = Speaker::anonymous(0).renamed("Alice");
        assert_eq!(s.display_name(), "Alice");
    }

    #[test]
    fn whitespace_only_label_falls_back_to_default() {
        let s = Speaker::anonymous(2).renamed("   ");
        assert_eq!(s.label, None);
        assert_eq!(s.display_name(), "Speaker 3");
    }

    #[test]
    fn renaming_trims_whitespace() {
        let s = Speaker::anonymous(0).renamed("  Bob  ");
        assert_eq!(s.label.as_deref(), Some("Bob"));
        assert_eq!(s.display_name(), "Bob");
    }
}
