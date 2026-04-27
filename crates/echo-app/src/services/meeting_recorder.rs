//! `MeetingRecorder` — bridges streaming events into the [`MeetingStore`].
//!
//! Lifecycle:
//!
//! ```text
//!   StreamingPipeline                  Frontend channel
//!         │                                  │
//!         ▼                                  ▲
//!     TranscriptEvent ────► MeetingRecorder ─┘
//!                                  │
//!                                  ▼
//!                           MeetingStore (SQLite)
//! ```
//!
//! The recorder is *event-driven*: callers feed it every event coming
//! out of the pipeline (typically inside the same drain loop that
//! forwards events to the IPC channel). It opens a meeting on
//! `Started`, persists segments on each `Chunk` (in a single
//! transaction so a crash leaves the DB consistent), and finalizes the
//! meeting on `Stopped` / `Failed`.
//!
//! The recorder owns no I/O of its own beyond the [`MeetingStore`]
//! port, so unit tests can swap a `Mutex<Vec<…>>`-backed store in
//! without touching SQLite.

use std::sync::Arc;

use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;
use tracing::{debug, warn};

use echo_domain::{
    CreateMeeting, DomainError, FinalizeMeeting, MeetingId, MeetingStore, Speaker,
    StreamingSessionId, TranscriptEvent,
};

/// Stateful recorder. One instance per session is the simplest pattern,
/// but the type also handles multiple interleaved sessions safely
/// because state is keyed by [`StreamingSessionId`].
pub struct MeetingRecorder {
    store: Arc<dyn MeetingStore>,
    title_template: String,
    /// Per-session running stats. Populated lazily on `Started`.
    sessions: tokio::sync::Mutex<std::collections::HashMap<StreamingSessionId, SessionStats>>,
}

#[derive(Debug, Default)]
struct SessionStats {
    meeting_id: MeetingId,
    duration_ms: u32,
    segment_count: u32,
    /// Tally of detected languages so we can pick the dominant one on stop.
    language_votes: std::collections::HashMap<String, u32>,
}

impl SessionStats {
    fn dominant_language(&self) -> Option<String> {
        self.language_votes
            .iter()
            .max_by_key(|(_, n)| *n)
            .map(|(lang, _)| lang.clone())
    }
}

impl MeetingRecorder {
    /// Wire the recorder against a concrete [`MeetingStore`].
    /// `title_template` is used as the meeting title; a literal
    /// `{date}` placeholder is replaced with the local date.
    pub fn new(store: Arc<dyn MeetingStore>, title_template: impl Into<String>) -> Self {
        Self {
            store,
            title_template: title_template.into(),
            sessions: tokio::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Convenience constructor with a sensible default title.
    pub fn with_default_title(store: Arc<dyn MeetingStore>) -> Self {
        Self::new(store, "Meeting {date}")
    }

    /// Feed one event. Returns the `MeetingId` associated with this
    /// session once it is known (i.e. after the first `Started`).
    pub async fn record(&self, event: &TranscriptEvent) -> Result<Option<MeetingId>, DomainError> {
        match event {
            TranscriptEvent::Started {
                session_id,
                input_format,
            } => {
                let meeting_id = MeetingId::new();
                let title = self.render_title();
                self.store
                    .create(CreateMeeting {
                        id: meeting_id,
                        title,
                        input_format: *input_format,
                    })
                    .await?;
                let mut guard = self.sessions.lock().await;
                guard.insert(
                    *session_id,
                    SessionStats {
                        meeting_id,
                        ..Default::default()
                    },
                );
                debug!(%session_id, %meeting_id, "recorder: meeting created");
                Ok(Some(meeting_id))
            }
            TranscriptEvent::Chunk {
                session_id,
                segments,
                language,
                speaker_id,
                speaker_slot,
                ..
            } => {
                let meeting_id = self
                    .update_chunk_stats(session_id, segments, language)
                    .await;
                if let Some(meeting_id) = meeting_id {
                    // Persist the speaker BEFORE the segments so the
                    // `segments.speaker_id` foreign key always finds
                    // its row, even on the very first chunk for this
                    // slot. Anonymous (label = None) — the rename use
                    // case populates the label later, and the COALESCE
                    // upsert keeps it from being clobbered by the next
                    // recorder pass.
                    if let (Some(sid), Some(slot)) = (speaker_id, speaker_slot) {
                        self.store
                            .upsert_speaker(
                                meeting_id,
                                &Speaker {
                                    id: *sid,
                                    slot: *slot,
                                    label: None,
                                },
                            )
                            .await?;
                    }
                    if !segments.is_empty() {
                        self.store.append_segments(meeting_id, segments).await?;
                    }
                    Ok(Some(meeting_id))
                } else {
                    warn!(%session_id, "recorder: chunk for unknown session");
                    Ok(None)
                }
            }
            TranscriptEvent::Skipped {
                session_id,
                duration_ms,
                ..
            } => {
                let mut guard = self.sessions.lock().await;
                if let Some(stats) = guard.get_mut(session_id) {
                    stats.duration_ms = stats.duration_ms.saturating_add(*duration_ms);
                    Ok(Some(stats.meeting_id))
                } else {
                    Ok(None)
                }
            }
            TranscriptEvent::Stopped {
                session_id,
                total_segments,
                total_audio_ms,
            } => {
                let stats = self.sessions.lock().await.remove(session_id);
                let Some(stats) = stats else {
                    return Ok(None);
                };
                let now = OffsetDateTime::now_utc()
                    .format(&Rfc3339)
                    .map_err(|e| DomainError::Invariant(format!("format ended_at: {e}")))?;
                self.store
                    .finalize(
                        stats.meeting_id,
                        FinalizeMeeting {
                            ended_at_rfc3339: Some(now),
                            duration_ms: Some(*total_audio_ms),
                            language: stats.dominant_language(),
                            segment_count: Some(*total_segments),
                        },
                    )
                    .await?;
                debug!(
                    %session_id, meeting_id = %stats.meeting_id,
                    total_segments, total_audio_ms,
                    "recorder: meeting finalized"
                );
                Ok(Some(stats.meeting_id))
            }
            TranscriptEvent::Failed {
                session_id,
                message,
            } => {
                let stats = self.sessions.lock().await.remove(session_id);
                if let Some(stats) = stats {
                    let now = OffsetDateTime::now_utc()
                        .format(&Rfc3339)
                        .map_err(|e| DomainError::Invariant(format!("format ended_at: {e}")))?;
                    if let Err(e) = self
                        .store
                        .finalize(
                            stats.meeting_id,
                            FinalizeMeeting {
                                ended_at_rfc3339: Some(now),
                                duration_ms: Some(stats.duration_ms),
                                language: stats.dominant_language(),
                                segment_count: Some(stats.segment_count),
                            },
                        )
                        .await
                    {
                        // The session already failed upstream — surfacing
                        // a second error to the caller would mask the
                        // original cause. Log loudly so the operator can
                        // still see we couldn't even mark the meeting
                        // ended; the meeting will appear as "in flight"
                        // in the sidebar until manually cleaned up.
                        warn!(
                            %session_id, meeting_id = %stats.meeting_id, error = %e,
                            "recorder: failed to mark failed meeting as ended"
                        );
                    }
                    warn!(%session_id, meeting_id = %stats.meeting_id, %message, "recorder: meeting failed");
                    Ok(Some(stats.meeting_id))
                } else {
                    Ok(None)
                }
            }
            // Paused / Resumed are informational UI events — the
            // recorder doesn't persist anything for them.
            TranscriptEvent::Paused { .. } | TranscriptEvent::Resumed { .. } => Ok(None),
        }
    }

    async fn update_chunk_stats(
        &self,
        session_id: &StreamingSessionId,
        segments: &[echo_domain::Segment],
        language: &Option<String>,
    ) -> Option<MeetingId> {
        let mut guard = self.sessions.lock().await;
        let stats = guard.get_mut(session_id)?;
        let max_end = segments.iter().map(|s| s.end_ms).max().unwrap_or(0);
        if max_end > stats.duration_ms {
            stats.duration_ms = max_end;
        }
        stats.segment_count = stats.segment_count.saturating_add(segments.len() as u32);
        if let Some(lang) = language {
            *stats.language_votes.entry(lang.clone()).or_insert(0) += 1;
        }
        Some(stats.meeting_id)
    }

    fn render_title(&self) -> String {
        let date = OffsetDateTime::now_utc().date().to_string(); // YYYY-MM-DD
        self.title_template.replace("{date}", &date)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    use async_trait::async_trait;
    use pretty_assertions::assert_eq;

    use echo_domain::{AudioFormat, Meeting, MeetingSummary, Segment, SegmentId, SpeakerId};

    /// In-memory store keyed by id; threadsafe via a single Mutex.
    #[derive(Default)]
    struct FakeStore {
        meetings: Mutex<Vec<Meeting>>,
    }

    impl FakeStore {
        fn snapshot(&self) -> Vec<Meeting> {
            self.meetings.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl MeetingStore for FakeStore {
        async fn create(&self, params: CreateMeeting) -> Result<MeetingSummary, DomainError> {
            let summary = MeetingSummary {
                id: params.id,
                title: params.title.clone(),
                started_at: OffsetDateTime::now_utc(),
                ended_at: None,
                duration_ms: 0,
                language: None,
                segment_count: 0,
            };
            self.meetings.lock().unwrap().push(Meeting {
                summary: summary.clone(),
                input_format: params.input_format,
                segments: vec![],
                speakers: vec![],
            });
            Ok(summary)
        }
        async fn append_segments(
            &self,
            meeting_id: MeetingId,
            segments: &[Segment],
        ) -> Result<(), DomainError> {
            let mut guard = self.meetings.lock().unwrap();
            let m = guard
                .iter_mut()
                .find(|m| m.summary.id == meeting_id)
                .ok_or_else(|| DomainError::NotFound {
                    entity: "meeting",
                    id: meeting_id.to_string(),
                })?;
            for s in segments {
                if !m.segments.iter().any(|x| x.id == s.id) {
                    m.segments.push(s.clone());
                }
            }
            m.summary.segment_count = m.segments.len() as u32;
            Ok(())
        }
        async fn upsert_speaker(
            &self,
            meeting_id: MeetingId,
            speaker: &Speaker,
        ) -> Result<(), DomainError> {
            let mut guard = self.meetings.lock().unwrap();
            let m = guard
                .iter_mut()
                .find(|m| m.summary.id == meeting_id)
                .ok_or_else(|| DomainError::NotFound {
                    entity: "meeting",
                    id: meeting_id.to_string(),
                })?;
            // Mirror the SQLite COALESCE behaviour: keep the existing
            // SpeakerId on slot conflict, only refresh label when the
            // caller passed Some.
            if let Some(existing) = m.speakers.iter_mut().find(|s| s.slot == speaker.slot) {
                if let Some(label) = &speaker.label {
                    existing.label = Some(label.clone());
                }
            } else {
                m.speakers.push(speaker.clone());
            }
            Ok(())
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
            s.label = label.map(|l| l.to_string());
            Ok(true)
        }
        async fn list_speakers(&self, meeting_id: MeetingId) -> Result<Vec<Speaker>, DomainError> {
            let guard = self.meetings.lock().unwrap();
            Ok(guard
                .iter()
                .find(|m| m.summary.id == meeting_id)
                .map(|m| {
                    let mut v = m.speakers.clone();
                    v.sort_by_key(|s| s.slot);
                    v
                })
                .unwrap_or_default())
        }
        async fn finalize(
            &self,
            meeting_id: MeetingId,
            patch: FinalizeMeeting,
        ) -> Result<MeetingSummary, DomainError> {
            let mut guard = self.meetings.lock().unwrap();
            let m = guard
                .iter_mut()
                .find(|m| m.summary.id == meeting_id)
                .ok_or_else(|| DomainError::NotFound {
                    entity: "meeting",
                    id: meeting_id.to_string(),
                })?;
            if let Some(d) = patch.duration_ms {
                m.summary.duration_ms = d;
            }
            if patch.language.is_some() {
                m.summary.language = patch.language;
            }
            if let Some(c) = patch.segment_count {
                m.summary.segment_count = c;
            }
            if patch.ended_at_rfc3339.is_some() {
                m.summary.ended_at = Some(OffsetDateTime::now_utc());
            }
            Ok(m.summary.clone())
        }
        async fn list(&self, _limit: u32) -> Result<Vec<MeetingSummary>, DomainError> {
            Ok(self
                .meetings
                .lock()
                .unwrap()
                .iter()
                .map(|m| m.summary.clone())
                .collect())
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
        async fn delete(&self, meeting_id: MeetingId) -> Result<bool, DomainError> {
            let mut guard = self.meetings.lock().unwrap();
            let len_before = guard.len();
            guard.retain(|m| m.summary.id != meeting_id);
            Ok(guard.len() != len_before)
        }
        async fn search(
            &self,
            _query: &str,
            _limit: u32,
        ) -> Result<Vec<echo_domain::MeetingSearchHit>, DomainError> {
            // FakeStore is only used to test the recorder, not search.
            Ok(Vec::new())
        }
        async fn upsert_summary(&self, _summary: &echo_domain::Summary) -> Result<(), DomainError> {
            // The recorder never writes summaries; the SummarizeMeeting
            // use case owns that path.
            unreachable!("recorder tests don't persist summaries")
        }
        async fn get_summary(
            &self,
            _meeting_id: MeetingId,
        ) -> Result<Option<echo_domain::Summary>, DomainError> {
            unreachable!()
        }
    }

    fn segment(start_ms: u32, end_ms: u32, text: &str) -> Segment {
        Segment {
            id: SegmentId::new(),
            start_ms,
            end_ms,
            text: text.into(),
            speaker_id: None,
            confidence: None,
        }
    }

    #[tokio::test]
    async fn full_session_lifecycle_persists_meeting() {
        let store = Arc::new(FakeStore::default());
        let recorder = MeetingRecorder::with_default_title(store.clone());
        let session = StreamingSessionId::new();

        recorder
            .record(&TranscriptEvent::Started {
                session_id: session,
                input_format: AudioFormat::WHISPER,
            })
            .await
            .unwrap();
        let snap = store.snapshot();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].summary.segment_count, 0);

        recorder
            .record(&TranscriptEvent::Chunk {
                session_id: session,
                chunk_index: 0,
                offset_ms: 0,
                segments: vec![segment(0, 1_000, "hello")],
                language: Some("en".into()),
                rtf: 0.1,
                speaker_id: None,
                speaker_slot: None,
            })
            .await
            .unwrap();
        recorder
            .record(&TranscriptEvent::Chunk {
                session_id: session,
                chunk_index: 1,
                offset_ms: 1_000,
                segments: vec![segment(1_000, 2_000, "world")],
                language: Some("en".into()),
                rtf: 0.1,
                speaker_id: None,
                speaker_slot: None,
            })
            .await
            .unwrap();

        recorder
            .record(&TranscriptEvent::Stopped {
                session_id: session,
                total_segments: 2,
                total_audio_ms: 2_000,
            })
            .await
            .unwrap();

        let snap = store.snapshot();
        assert_eq!(snap.len(), 1);
        let m = &snap[0];
        assert_eq!(m.summary.segment_count, 2);
        assert_eq!(m.summary.duration_ms, 2_000);
        assert_eq!(m.summary.language.as_deref(), Some("en"));
        assert!(m.summary.ended_at.is_some());
        assert_eq!(m.segments.len(), 2);
    }

    #[tokio::test]
    async fn dominant_language_wins_on_finalize() {
        let store = Arc::new(FakeStore::default());
        let recorder = MeetingRecorder::with_default_title(store.clone());
        let session = StreamingSessionId::new();

        recorder
            .record(&TranscriptEvent::Started {
                session_id: session,
                input_format: AudioFormat::WHISPER,
            })
            .await
            .unwrap();
        for lang in ["en", "es", "en", "en", "fr"] {
            recorder
                .record(&TranscriptEvent::Chunk {
                    session_id: session,
                    chunk_index: 0,
                    offset_ms: 0,
                    segments: vec![segment(0, 1_000, "x")],
                    language: Some(lang.into()),
                    rtf: 0.1,
                    speaker_id: None,
                    speaker_slot: None,
                })
                .await
                .unwrap();
        }
        recorder
            .record(&TranscriptEvent::Stopped {
                session_id: session,
                total_segments: 5,
                total_audio_ms: 5_000,
            })
            .await
            .unwrap();
        let snap = store.snapshot();
        assert_eq!(snap[0].summary.language.as_deref(), Some("en"));
    }

    #[tokio::test]
    async fn chunks_with_speakers_persist_speaker_rows_before_segments() {
        let store = Arc::new(FakeStore::default());
        let recorder = MeetingRecorder::with_default_title(store.clone());
        let session = StreamingSessionId::new();

        recorder
            .record(&TranscriptEvent::Started {
                session_id: session,
                input_format: AudioFormat::WHISPER,
            })
            .await
            .unwrap();

        let alice = SpeakerId::new();
        let bob = SpeakerId::new();
        // Alternate two speakers across chunks. Each upsert is
        // idempotent, so re-seeing slot 0 must NOT spawn a duplicate
        // row, and the speaker must exist before the chunk's segment
        // is appended (otherwise the speakers vec would be empty
        // when get() returns the meeting).
        for (idx, (sid, slot)) in [(alice, 0u32), (bob, 1), (alice, 0), (bob, 1)]
            .into_iter()
            .enumerate()
        {
            let i = idx as u32;
            let mut seg = segment(i * 1_000, (i + 1) * 1_000, "x");
            seg.speaker_id = Some(sid);
            recorder
                .record(&TranscriptEvent::Chunk {
                    session_id: session,
                    chunk_index: i,
                    offset_ms: i * 1_000,
                    segments: vec![seg],
                    language: Some("en".into()),
                    rtf: 0.1,
                    speaker_id: Some(sid),
                    speaker_slot: Some(slot),
                })
                .await
                .unwrap();
        }

        let snap = store.snapshot();
        let meeting = &snap[0];
        assert_eq!(meeting.segments.len(), 4);
        assert_eq!(
            meeting.speakers.len(),
            2,
            "exactly one row per slot, idempotent across re-upserts"
        );
        let slots: Vec<u32> = meeting.speakers.iter().map(|s| s.slot).collect();
        assert_eq!(slots, vec![0, 1]);
        // Original SpeakerIds survive — segments still resolve to a real speaker row.
        assert!(meeting
            .speakers
            .iter()
            .any(|s| s.id == alice && s.slot == 0));
        assert!(meeting.speakers.iter().any(|s| s.id == bob && s.slot == 1));
        assert!(
            meeting.speakers.iter().all(|s| s.label.is_none()),
            "recorder leaves labels anonymous; rename use case sets them"
        );
    }

    #[tokio::test]
    async fn unknown_session_chunks_are_ignored() {
        let store = Arc::new(FakeStore::default());
        let recorder = MeetingRecorder::with_default_title(store.clone());
        let result = recorder
            .record(&TranscriptEvent::Chunk {
                session_id: StreamingSessionId::new(),
                chunk_index: 0,
                offset_ms: 0,
                segments: vec![segment(0, 1_000, "ghost")],
                language: None,
                rtf: 0.0,
                speaker_id: None,
                speaker_slot: None,
            })
            .await
            .unwrap();
        assert_eq!(result, None);
        assert!(store.snapshot().is_empty());
    }
}
