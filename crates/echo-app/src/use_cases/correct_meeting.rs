//! `correct_meeting` use case: user corrections to a saved transcript.
//!
//! After automatic diarization the user can fix a saved meeting in three
//! ways: merge two speakers that are the same person, move segments to
//! another speaker, and delete segments (for example everything after the
//! meeting really ended). Each operation validates its ids against the
//! meeting and delegates to one [`MeetingStore`] transaction, so a bad id
//! writes nothing.
//!
//! Merge label rule: the surviving speaker keeps its label; when it has
//! none it inherits the absorbed speaker's label. The absorbed speaker row
//! is always removed, because the user explicitly chose which name stays.
//! This differs from the automatic refinement, which keeps a labelled
//! absorbed row with zero segments so it never discards a user label it was
//! not asked about.

use std::sync::Arc;

use thiserror::Error;

use echo_domain::{
    DomainError, MeetingId, MeetingStore, SegmentId, SegmentSpeaker, SpeakerId, SpeakerReassignment,
};

/// Errors a correction can fail with, already split the way the IPC layer
/// reports them.
#[derive(Debug, Error)]
pub enum CorrectMeetingError {
    /// The meeting or one of the named speakers does not exist.
    #[error("{0}")]
    NotFound(String),
    /// The request is malformed: an id belongs to another meeting, or a
    /// speaker is merged into itself.
    #[error("{0}")]
    Invalid(String),
    /// Storage layer error, surfaced unchanged.
    #[error(transparent)]
    Storage(DomainError),
}

impl From<DomainError> for CorrectMeetingError {
    /// The store reports ids of another meeting as
    /// [`DomainError::Invariant`]; for a user request that is invalid
    /// input, not an internal failure.
    fn from(e: DomainError) -> Self {
        match e {
            DomainError::NotFound { .. } => Self::NotFound(e.to_string()),
            DomainError::Invariant(msg) => Self::Invalid(msg),
            other => Self::Storage(other),
        }
    }
}

/// Use case handler for saved-meeting corrections.
pub struct CorrectMeeting {
    store: Arc<dyn MeetingStore>,
}

impl CorrectMeeting {
    /// Wire the use case against a concrete store.
    #[must_use]
    pub fn new(store: Arc<dyn MeetingStore>) -> Self {
        Self { store }
    }

    /// Merge `absorbed` into `surviving`: every segment of `absorbed`
    /// moves to `surviving` and the `absorbed` row is removed, under the
    /// label rule in the module docs. Returns `true` when anything changed.
    pub async fn merge_speakers(
        &self,
        meeting_id: MeetingId,
        absorbed: SpeakerId,
        surviving: SpeakerId,
    ) -> Result<bool, CorrectMeetingError> {
        if absorbed == surviving {
            return Err(CorrectMeetingError::Invalid(format!(
                "speaker {absorbed} cannot be merged into itself"
            )));
        }
        let speakers = self.store.list_speakers(meeting_id).await?;
        let find = |id: SpeakerId| {
            speakers.iter().find(|s| s.id == id).ok_or_else(|| {
                CorrectMeetingError::NotFound(format!(
                    "speaker {id} not found in meeting {meeting_id}"
                ))
            })
        };
        let absorbed_row = find(absorbed)?;
        let surviving_row = find(surviving)?;
        let both_labelled = absorbed_row.label.is_some() && surviving_row.label.is_some();

        // One transaction moves the segments and retires `absorbed`,
        // handing its label to an unlabelled survivor.
        let mut changed = self
            .store
            .merge_speakers(meeting_id, absorbed, surviving)
            .await?;
        if both_labelled {
            // The store keeps a labelled absorbed row when the survivor has
            // a label of its own. Clearing that label and merging again
            // deletes the now segment-less row. If this cleanup fails the
            // merge itself is already committed and consistent.
            self.store
                .rename_speaker(meeting_id, absorbed, None)
                .await?;
            changed |= self
                .store
                .merge_speakers(meeting_id, absorbed, surviving)
                .await?;
        }
        Ok(changed)
    }

    /// Point every listed segment at `speaker_id` (`None` leaves them
    /// unattributed). Returns how many segments actually changed speaker.
    pub async fn reassign_segments(
        &self,
        meeting_id: MeetingId,
        segment_ids: &[SegmentId],
        speaker_id: Option<SpeakerId>,
    ) -> Result<u32, CorrectMeetingError> {
        let plan = SpeakerReassignment {
            segments: segment_ids
                .iter()
                .map(|&segment_id| SegmentSpeaker {
                    segment_id,
                    speaker_id,
                })
                .collect(),
            ..SpeakerReassignment::default()
        };
        let outcome = self.store.reassign_speakers(meeting_id, &plan).await?;
        Ok(outcome.segments_updated)
    }

    /// Delete the listed segments. Returns how many rows were deleted;
    /// already-deleted ids count as zero. Speakers left without segments
    /// stay (the meeting view hides them).
    pub async fn delete_segments(
        &self,
        meeting_id: MeetingId,
        segment_ids: &[SegmentId],
    ) -> Result<u32, CorrectMeetingError> {
        Ok(self.store.delete_segments(meeting_id, segment_ids).await?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use echo_domain::{AudioFormat, CreateMeeting, Meeting, Segment, Speaker};
    use echo_storage::SqliteMeetingStore;
    use pretty_assertions::assert_eq;

    struct Fixture {
        store: Arc<SqliteMeetingStore>,
        use_case: CorrectMeeting,
        meeting: MeetingId,
        a: Speaker,
        b: Speaker,
        seg_a: Segment,
        seg_b: Segment,
    }

    fn segment(start_ms: u32, text: &str, speaker: SpeakerId) -> Segment {
        Segment {
            id: SegmentId::new(),
            start_ms,
            end_ms: start_ms + 1_000,
            text: text.into(),
            speaker_id: Some(speaker),
            confidence: None,
        }
    }

    async fn seed(
        store: &SqliteMeetingStore,
        label_a: Option<&str>,
        label_b: Option<&str>,
    ) -> (MeetingId, Speaker, Speaker, Segment, Segment) {
        let meeting = MeetingId::new();
        store
            .create(CreateMeeting {
                id: meeting,
                title: "corrections".into(),
                input_format: AudioFormat::WHISPER,
            })
            .await
            .unwrap();
        let a = Speaker {
            label: label_a.map(str::to_owned),
            ..Speaker::anonymous(0)
        };
        let b = Speaker {
            label: label_b.map(str::to_owned),
            ..Speaker::anonymous(1)
        };
        store.upsert_speaker(meeting, &a).await.unwrap();
        store.upsert_speaker(meeting, &b).await.unwrap();
        let seg_a = segment(0, "uno", a.id);
        let seg_b = segment(1_000, "dos", b.id);
        store
            .append_segments(meeting, &[seg_a.clone(), seg_b.clone()])
            .await
            .unwrap();
        (meeting, a, b, seg_a, seg_b)
    }

    async fn fixture(label_a: Option<&str>, label_b: Option<&str>) -> Fixture {
        let store = Arc::new(SqliteMeetingStore::open_in_memory().await.unwrap());
        let (meeting, a, b, seg_a, seg_b) = seed(&store, label_a, label_b).await;
        Fixture {
            use_case: CorrectMeeting::new(store.clone()),
            store,
            meeting,
            a,
            b,
            seg_a,
            seg_b,
        }
    }

    async fn load(f: &Fixture, meeting: MeetingId) -> Meeting {
        f.store.get(meeting).await.unwrap().unwrap()
    }

    fn labels(m: &Meeting) -> Vec<(SpeakerId, Option<String>)> {
        m.speakers.iter().map(|s| (s.id, s.label.clone())).collect()
    }

    fn speakers_of(m: &Meeting) -> Vec<Option<SpeakerId>> {
        m.segments.iter().map(|s| s.speaker_id).collect()
    }

    // ---------- merge ---------------------------------------------------

    #[tokio::test]
    async fn merge_moves_segments_and_survivor_keeps_its_label() {
        let f = fixture(Some("Ana"), Some("Beto")).await;

        assert!(f
            .use_case
            .merge_speakers(f.meeting, f.b.id, f.a.id)
            .await
            .unwrap());

        let m = load(&f, f.meeting).await;
        assert_eq!(labels(&m), vec![(f.a.id, Some("Ana".into()))]);
        assert_eq!(speakers_of(&m), vec![Some(f.a.id), Some(f.a.id)]);
    }

    #[tokio::test]
    async fn merge_into_unlabelled_survivor_inherits_absorbed_label() {
        let f = fixture(None, Some("Beto")).await;

        assert!(f
            .use_case
            .merge_speakers(f.meeting, f.b.id, f.a.id)
            .await
            .unwrap());

        let m = load(&f, f.meeting).await;
        assert_eq!(labels(&m), vec![(f.a.id, Some("Beto".into()))]);
        assert_eq!(speakers_of(&m), vec![Some(f.a.id), Some(f.a.id)]);
    }

    #[tokio::test]
    async fn merge_of_unlabelled_speakers_removes_absorbed_row() {
        let f = fixture(None, None).await;

        assert!(f
            .use_case
            .merge_speakers(f.meeting, f.a.id, f.b.id)
            .await
            .unwrap());

        let m = load(&f, f.meeting).await;
        assert_eq!(labels(&m), vec![(f.b.id, None)]);
        assert_eq!(speakers_of(&m), vec![Some(f.b.id), Some(f.b.id)]);
    }

    #[tokio::test]
    async fn merge_rejects_self_merge_and_unknown_or_foreign_speakers() {
        let f = fixture(Some("Ana"), None).await;
        let (other, x, _, _, _) = seed(&f.store, None, None).await;
        let before = load(&f, f.meeting).await;

        assert!(matches!(
            f.use_case.merge_speakers(f.meeting, f.a.id, f.a.id).await,
            Err(CorrectMeetingError::Invalid(_))
        ));
        assert!(matches!(
            f.use_case.merge_speakers(f.meeting, x.id, f.a.id).await,
            Err(CorrectMeetingError::NotFound(_))
        ));
        assert!(matches!(
            f.use_case.merge_speakers(f.meeting, f.b.id, x.id).await,
            Err(CorrectMeetingError::NotFound(_))
        ));
        assert!(matches!(
            f.use_case
                .merge_speakers(MeetingId::new(), f.b.id, f.a.id)
                .await,
            Err(CorrectMeetingError::NotFound(_))
        ));
        assert_eq!(load(&f, f.meeting).await, before);
        assert_eq!(f.store.list_speakers(other).await.unwrap().len(), 2);
    }

    // ---------- reassign ------------------------------------------------

    #[tokio::test]
    async fn reassign_moves_segments_and_counts_only_changes() {
        let f = fixture(None, None).await;

        let changed = f
            .use_case
            .reassign_segments(f.meeting, &[f.seg_a.id, f.seg_b.id], Some(f.b.id))
            .await
            .unwrap();
        assert_eq!(changed, 1, "seg_b already belonged to b");
        assert_eq!(
            speakers_of(&load(&f, f.meeting).await),
            vec![Some(f.b.id), Some(f.b.id)]
        );

        let cleared = f
            .use_case
            .reassign_segments(f.meeting, &[f.seg_a.id], None)
            .await
            .unwrap();
        assert_eq!(cleared, 1);
        let m = load(&f, f.meeting).await;
        assert_eq!(speakers_of(&m), vec![None, Some(f.b.id)]);
        // Speaker `a` lost its last segment but its row stays.
        assert_eq!(m.speakers.len(), 2);
    }

    #[tokio::test]
    async fn reassign_rejects_foreign_ids_and_writes_nothing() {
        let f = fixture(None, None).await;
        let (_, x, _, seg_x, _) = seed(&f.store, None, None).await;
        let before = load(&f, f.meeting).await;

        assert!(matches!(
            f.use_case
                .reassign_segments(f.meeting, &[f.seg_a.id, seg_x.id], Some(f.b.id))
                .await,
            Err(CorrectMeetingError::Invalid(_))
        ));
        assert!(matches!(
            f.use_case
                .reassign_segments(f.meeting, &[f.seg_a.id], Some(x.id))
                .await,
            Err(CorrectMeetingError::Invalid(_))
        ));
        assert!(matches!(
            f.use_case
                .reassign_segments(MeetingId::new(), &[f.seg_a.id], None)
                .await,
            Err(CorrectMeetingError::NotFound(_))
        ));
        assert_eq!(load(&f, f.meeting).await, before);
    }

    // ---------- delete --------------------------------------------------

    #[tokio::test]
    async fn delete_removes_segments_and_rejects_foreign_ids() {
        let f = fixture(None, Some("Beto")).await;
        let (other, _, _, seg_x, _) = seed(&f.store, None, None).await;
        let before = load(&f, f.meeting).await;

        assert!(matches!(
            f.use_case
                .delete_segments(f.meeting, &[f.seg_b.id, seg_x.id])
                .await,
            Err(CorrectMeetingError::Invalid(_))
        ));
        assert_eq!(load(&f, f.meeting).await, before);
        assert_eq!(load(&f, other).await.segments.len(), 2);

        assert_eq!(
            f.use_case
                .delete_segments(f.meeting, &[f.seg_b.id])
                .await
                .unwrap(),
            1
        );
        assert_eq!(
            f.use_case
                .delete_segments(f.meeting, &[f.seg_b.id])
                .await
                .unwrap(),
            0
        );
        let m = load(&f, f.meeting).await;
        assert_eq!(m.summary.segment_count, 1);
        assert_eq!(m.segments.len(), 1);
        // The emptied speaker keeps its row and label.
        assert_eq!(labels(&m), labels(&before));

        assert!(matches!(
            f.use_case
                .delete_segments(MeetingId::new(), &[f.seg_a.id])
                .await,
            Err(CorrectMeetingError::NotFound(_))
        ));
    }
}
