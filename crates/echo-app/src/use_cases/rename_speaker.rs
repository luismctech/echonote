//! `rename_speaker` use case (CU-04).
//!
//! Lets the user attach a human-readable name to a diarized speaker
//! (or clear the label back to anonymous). The use case is a thin
//! validator on top of [`MeetingStore::rename_speaker`]: it normalises
//! the input (trim + collapse-empty-to-None + length cap) and surfaces
//! a typed error for the "speaker not found" case so the IPC layer can
//! turn it into a 404-equivalent without parsing string messages.
//!
//! The 80-char cap matches the documented soft limit on
//! [`echo_domain::Speaker::label`]; the domain itself does not enforce
//! it because storage is the natural place to defend the column width.

use std::sync::Arc;

use thiserror::Error;

use echo_domain::{DomainError, MeetingId, MeetingStore, SpeakerId};

/// Maximum number of UTF-8 characters allowed in a speaker label.
/// Past this the UI starts wrapping awkwardly and there is no real
/// use case for very long names — pick a sensible upper bound and
/// reject anything beyond it at the application boundary.
pub const MAX_LABEL_CHARS: usize = 80;

/// Errors the use case can fail with. `NotFound` is expected (user
/// renamed a speaker that was already deleted on another device);
/// `Invalid` covers caller mistakes; `Storage` wraps anything the
/// adapter throws so the caller doesn't have to import `DomainError`.
#[derive(Debug, Error)]
pub enum RenameSpeakerError {
    /// The (meeting, speaker) pair did not exist in the store.
    #[error("speaker {speaker_id} not found in meeting {meeting_id}")]
    NotFound {
        /// Meeting that was searched.
        meeting_id: MeetingId,
        /// Speaker that the caller asked to rename.
        speaker_id: SpeakerId,
    },
    /// The supplied label failed validation (too long).
    #[error("invalid label: {0}")]
    Invalid(String),
    /// Storage layer error, surfaced unchanged.
    #[error(transparent)]
    Storage(#[from] DomainError),
}

/// Use case handler. Holding the store as `Arc<dyn …>` keeps tests
/// independent of any concrete adapter.
pub struct RenameSpeaker {
    store: Arc<dyn MeetingStore>,
}

impl RenameSpeaker {
    /// Wire the use case against a concrete store.
    #[must_use]
    pub fn new(store: Arc<dyn MeetingStore>) -> Self {
        Self { store }
    }

    /// Apply a rename. `label = None` clears the user-assigned name
    /// and the speaker reverts to its slot-derived `Speaker N`
    /// rendering. Whitespace-only labels are normalised to `None` so
    /// the UI can offer a "clear" affordance via empty input without
    /// a separate command.
    pub async fn execute(
        &self,
        meeting_id: MeetingId,
        speaker_id: SpeakerId,
        label: Option<String>,
    ) -> Result<(), RenameSpeakerError> {
        let normalised = normalise_label(label.as_deref())?;
        let renamed = self
            .store
            .rename_speaker(meeting_id, speaker_id, normalised.as_deref())
            .await?;
        if !renamed {
            return Err(RenameSpeakerError::NotFound {
                meeting_id,
                speaker_id,
            });
        }
        Ok(())
    }
}

/// Strip outer whitespace and collapse the empty case to `None`.
/// Pulled out of `execute` so the rules are unit-testable without an
/// in-memory store; the domain `Speaker::renamed` uses the same
/// rules for in-memory mutation, so the contract stays consistent
/// across layers.
fn normalise_label(input: Option<&str>) -> Result<Option<String>, RenameSpeakerError> {
    let Some(raw) = input else { return Ok(None) };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    if trimmed.chars().count() > MAX_LABEL_CHARS {
        return Err(RenameSpeakerError::Invalid(format!(
            "label exceeds {MAX_LABEL_CHARS} characters"
        )));
    }
    Ok(Some(trimmed.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    use async_trait::async_trait;
    use pretty_assertions::assert_eq;
    use time::OffsetDateTime;

    use echo_domain::{
        AudioFormat, CreateMeeting, FinalizeMeeting, Meeting, MeetingSummary, Segment, Speaker,
    };

    /// Minimal in-memory store. Only implements the methods the use
    /// case touches; the rest panic so a future change that adds new
    /// calls fails loudly during the test run.
    #[derive(Default)]
    struct FakeStore {
        meetings: Mutex<Vec<Meeting>>,
    }

    #[async_trait]
    impl MeetingStore for FakeStore {
        async fn create(&self, _: CreateMeeting) -> Result<MeetingSummary, DomainError> {
            unreachable!("rename_speaker tests don't create meetings via the port")
        }
        async fn append_segments(&self, _: MeetingId, _: &[Segment]) -> Result<(), DomainError> {
            unreachable!()
        }
        async fn upsert_speaker(&self, _: MeetingId, _: &Speaker) -> Result<(), DomainError> {
            unreachable!()
        }
        async fn list_speakers(&self, _: MeetingId) -> Result<Vec<Speaker>, DomainError> {
            unreachable!()
        }
        async fn rename_speaker(
            &self,
            meeting_id: MeetingId,
            speaker_id: SpeakerId,
            label: Option<&str>,
        ) -> Result<bool, DomainError> {
            let mut guard = self.meetings.lock().unwrap();
            let Some(m) = guard.iter_mut().find(|m| m.summary.id == meeting_id) else {
                return Ok(false);
            };
            let Some(s) = m.speakers.iter_mut().find(|s| s.id == speaker_id) else {
                return Ok(false);
            };
            s.label = label.map(str::to_string);
            Ok(true)
        }
        async fn finalize(
            &self,
            _: MeetingId,
            _: FinalizeMeeting,
        ) -> Result<MeetingSummary, DomainError> {
            unreachable!()
        }
        async fn list(&self, _: u32) -> Result<Vec<MeetingSummary>, DomainError> {
            unreachable!()
        }
        async fn get(&self, meeting_id: MeetingId) -> Result<Option<Meeting>, DomainError> {
            Ok(self
                .meetings
                .lock()
                .unwrap()
                .iter()
                .find(|m| m.summary.id == meeting_id)
                .cloned())
        }
        async fn delete(&self, _: MeetingId) -> Result<bool, DomainError> {
            unreachable!()
        }
    }

    fn seed_meeting(store: &FakeStore) -> (MeetingId, SpeakerId) {
        let meeting_id = MeetingId::new();
        let s = Speaker::anonymous(0);
        let speaker_id = s.id;
        store.meetings.lock().unwrap().push(Meeting {
            summary: MeetingSummary {
                id: meeting_id,
                title: "T".into(),
                started_at: OffsetDateTime::now_utc(),
                ended_at: None,
                duration_ms: 0,
                language: None,
                segment_count: 0,
            },
            input_format: AudioFormat::WHISPER,
            segments: vec![],
            speakers: vec![s],
        });
        (meeting_id, speaker_id)
    }

    #[tokio::test]
    async fn sets_and_clears_label() {
        let store = Arc::new(FakeStore::default());
        let (mid, sid) = seed_meeting(&store);
        let uc = RenameSpeaker::new(store.clone());

        uc.execute(mid, sid, Some("Alice".into())).await.unwrap();
        let snap = store.get(mid).await.unwrap().unwrap();
        assert_eq!(snap.speakers[0].label.as_deref(), Some("Alice"));

        // Empty string clears the label so the UI can use a single
        // text input to both rename and reset.
        uc.execute(mid, sid, Some(String::new())).await.unwrap();
        let snap = store.get(mid).await.unwrap().unwrap();
        assert!(snap.speakers[0].label.is_none());

        // Explicit None also clears.
        uc.execute(mid, sid, Some("Bob".into())).await.unwrap();
        uc.execute(mid, sid, None).await.unwrap();
        let snap = store.get(mid).await.unwrap().unwrap();
        assert!(snap.speakers[0].label.is_none());
    }

    #[tokio::test]
    async fn whitespace_label_is_normalised_to_none() {
        let store = Arc::new(FakeStore::default());
        let (mid, sid) = seed_meeting(&store);
        let uc = RenameSpeaker::new(store.clone());
        uc.execute(mid, sid, Some("Alice".into())).await.unwrap();
        uc.execute(mid, sid, Some("   ".into())).await.unwrap();
        let snap = store.get(mid).await.unwrap().unwrap();
        assert!(snap.speakers[0].label.is_none());
    }

    #[tokio::test]
    async fn label_exceeding_cap_is_rejected() {
        let store = Arc::new(FakeStore::default());
        let (mid, sid) = seed_meeting(&store);
        let uc = RenameSpeaker::new(store.clone());
        let too_long = "a".repeat(MAX_LABEL_CHARS + 1);
        let err = uc.execute(mid, sid, Some(too_long)).await.unwrap_err();
        assert!(matches!(err, RenameSpeakerError::Invalid(_)), "{err:?}");
    }

    #[tokio::test]
    async fn unknown_speaker_returns_not_found() {
        let store = Arc::new(FakeStore::default());
        let (mid, _) = seed_meeting(&store);
        let uc = RenameSpeaker::new(store.clone());
        let err = uc
            .execute(mid, SpeakerId::new(), Some("ghost".into()))
            .await
            .unwrap_err();
        assert!(
            matches!(err, RenameSpeakerError::NotFound { .. }),
            "{err:?}"
        );
    }
}
