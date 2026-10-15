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

use crate::entities::meeting::{Meeting, MeetingId, MeetingSearchHit, MeetingSummary};
use crate::entities::note::Note;
use crate::entities::segment::{Segment, SegmentId};
use crate::entities::speaker::{Speaker, SpeakerId};
use crate::entities::streaming::{SegmentSpeaker, SpeakerRef, SpeakerRetirement};
use crate::entities::summary::Summary;
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

/// Speaker changes from one offline re-segmentation, applied by
/// [`MeetingStore::reassign_speakers`]. Same shapes as
/// `TranscriptEvent::SpeakersReassigned`; ids only.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpeakerReassignment {
    /// Speakers to insert (unlabelled) before any segment points at them.
    pub created: Vec<SpeakerRef>,
    /// Speakers left without segments, with their majority recipient.
    pub retired: Vec<SpeakerRetirement>,
    /// Segments to point at a new speaker.
    pub segments: Vec<SegmentSpeaker>,
}

/// What [`MeetingStore::reassign_speakers`] changed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReassignOutcome {
    /// Segments whose speaker actually changed.
    pub segments_updated: u32,
    /// Speaker rows inserted (0 on replay).
    pub speakers_created: u32,
    /// Retired speaker rows deleted.
    pub speakers_deleted: u32,
    /// User labels handed to a majority recipient.
    pub labels_moved: u32,
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

    /// Persist or refresh a speaker row for the meeting.
    ///
    /// Idempotent on `(meeting_id, slot)`: calling it twice with the
    /// same slot keeps the existing row's `id` (so any
    /// `segments.speaker_id` foreign keys stay valid). The `label`
    /// follows COALESCE semantics — a `Some(label)` overwrites the
    /// stored value, while a `None` preserves it. That lets both the
    /// streaming recorder (always `label = None`, just registering
    /// the speaker exists) and the rename use case (always
    /// `label = Some(...)`) call this same method without stomping
    /// on each other.
    ///
    /// Implementations MUST insert the speaker row before any
    /// segment that references its `id`; the `MeetingRecorder` orders
    /// the calls accordingly.
    async fn upsert_speaker(
        &self,
        meeting_id: MeetingId,
        speaker: &Speaker,
    ) -> Result<(), DomainError>;

    /// Snapshot of all speakers persisted for a meeting, ordered by
    /// `slot` ascending. Returns an empty vec when the meeting is
    /// unknown or has no diarized speakers yet.
    async fn list_speakers(&self, meeting_id: MeetingId) -> Result<Vec<Speaker>, DomainError>;

    /// Set or clear a speaker's user-visible label. Distinct from
    /// [`MeetingStore::upsert_speaker`] because the rename flow needs
    /// to address the row by `SpeakerId` (so the UI can rename a
    /// speaker without knowing its slot) and must be able to clear
    /// the label back to `None` (which the COALESCE upsert cannot
    /// express). Returns `false` when the (meeting, speaker) pair
    /// was not found, so the use case can surface a 404 to the UI.
    async fn rename_speaker(
        &self,
        meeting_id: MeetingId,
        speaker_id: SpeakerId,
        label: Option<&str>,
    ) -> Result<bool, DomainError>;

    /// Collapse one speaker into another for this meeting.
    ///
    /// Rewrites every `segments.speaker_id` referencing `absorbed` to
    /// `surviving`, then retires the `absorbed` row without ever losing a
    /// user label (ADR 0001, Q1): an unlabelled row is deleted; a labelled
    /// row hands its label to `surviving` when that row has none, and is
    /// otherwise kept with zero segments. Such a labelled, segment-less
    /// row stays in [`MeetingStore::list_speakers`] and in the meeting's
    /// speakers, so consumers must tolerate a speaker that owns no
    /// segment (the desktop speakers panel shows it at 0 % talk time,
    /// where it can still be renamed). Used by the
    /// offline diarization refinement pass invoked at the end of a
    /// streaming session to fix online over-splits.
    ///
    /// Idempotent: if `absorbed` no longer exists the call succeeds
    /// with `Ok(false)`. Returns `Ok(true)` when at least one
    /// speaker or segment row was modified. If `surviving` does not
    /// exist the call is a no-op (returns `Ok(false)`) — callers
    /// should keep both ids in sync with the diarizer's view of the
    /// meeting.
    async fn merge_speakers(
        &self,
        meeting_id: MeetingId,
        absorbed: SpeakerId,
        surviving: SpeakerId,
    ) -> Result<bool, DomainError>;

    /// Apply an offline re-segmentation in one all-or-nothing
    /// transaction (ADR 0001, D5).
    ///
    /// In order: insert `created` speakers (replay-safe), point every
    /// listed segment at its new speaker, then retire each `retired`
    /// speaker that has no segment left, under the same label rule as
    /// [`MeetingStore::merge_speakers`]: a user label is never
    /// overwritten or destroyed.
    ///
    /// Returns [`DomainError::NotFound`] when the meeting does not
    /// exist and [`DomainError::Invariant`] when a listed segment, a
    /// referenced speaker, a retired speaker or a created speaker id
    /// belongs to another meeting; nothing is written in either case.
    /// Applying the same plan twice leaves the same state.
    async fn reassign_speakers(
        &self,
        meeting_id: MeetingId,
        plan: &SpeakerReassignment,
    ) -> Result<ReassignOutcome, DomainError>;

    /// Delete segments of one meeting in a single all-or-nothing
    /// transaction, used when the user trims a saved transcript.
    ///
    /// Keeps derived state consistent: the full-text index drops the
    /// deleted text and the meeting's `segment_count` is recomputed.
    /// Speakers left without segments are kept, like after
    /// [`MeetingStore::merge_speakers`], so deleting text never destroys
    /// a user label.
    ///
    /// Returns the number of rows deleted. Ids unknown to every meeting
    /// (for example already deleted) are skipped, so a retry returns
    /// `Ok(0)`. Returns [`DomainError::NotFound`] when the meeting does
    /// not exist and [`DomainError::Invariant`] when an id belongs to
    /// another meeting; nothing is deleted in either case.
    async fn delete_segments(
        &self,
        meeting_id: MeetingId,
        segment_ids: &[SegmentId],
    ) -> Result<u32, DomainError>;

    /// Update the display title of a meeting. Returns `false` when
    /// the meeting id was not found.
    async fn rename_meeting(&self, meeting_id: MeetingId, title: &str)
        -> Result<bool, DomainError>;

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

    /// Return the full meeting (with segments + speakers) or `None`
    /// when the id is unknown.
    async fn get(&self, meeting_id: MeetingId) -> Result<Option<Meeting>, DomainError>;

    /// Full-text search over segment text. Implementations return at
    /// most one hit per meeting (collapsed on the strongest segment),
    /// ordered by FTS5 BM25 rank ascending — i.e. best match first.
    /// `limit` caps the result; `0` means "no cap".
    ///
    /// The query is treated as raw user input. Implementations MUST
    /// escape FTS5 syntax characters (`"`, `*`, `(`, `)`, `^`, `:`,
    /// `+`, `-`, `~`, `NEAR`, `AND`, `OR`, `NOT`) so a stray double
    /// quote can never become a parser error or, worse, a CYK
    /// injection through the trigger machinery. The simplest path —
    /// and the one the SQLite adapter takes — is to wrap each
    /// whitespace-separated token in double quotes after escaping
    /// embedded quotes; that turns the query into a plain
    /// "match all of these phrases" search.
    ///
    /// Returns an empty vec for empty / whitespace-only queries
    /// instead of erroring, so the UI can wire `onChange` directly.
    async fn search(&self, query: &str, limit: u32) -> Result<Vec<MeetingSearchHit>, DomainError>;

    /// Delete a meeting and its segments. Returns `false` when the id
    /// did not exist.
    async fn delete(&self, meeting_id: MeetingId) -> Result<bool, DomainError>;

    /// Persist (or replace) the LLM summary attached to a meeting.
    ///
    /// Re-running the summarizer overwrites the previous row for the
    /// same `(meeting_id, template)` pair so a meeting only ever has
    /// one current summary per template. Implementations MUST set
    /// the summary's `created_at` as the source of truth — the value
    /// on the input is treated as authoritative and stored verbatim.
    ///
    /// Returns [`DomainError::NotFound`] when `summary.meeting_id`
    /// does not exist (so the caller can distinguish "the user
    /// deleted the meeting while we were summarizing" from a real
    /// storage failure).
    async fn upsert_summary(&self, summary: &Summary) -> Result<(), DomainError>;

    /// Most-recent summary attached to a meeting, or `None` when
    /// no summary has been generated yet. The MVP only ships the
    /// "general" template, so this returns at most one row; once
    /// other templates land we'll add a `template` parameter.
    async fn get_summary(&self, meeting_id: MeetingId) -> Result<Option<Summary>, DomainError>;

    /// Add a user note to a meeting. The note is persisted immediately
    /// so no data is lost if the app crashes. Returns the persisted
    /// note (with its generated `id` and `created_at`).
    async fn add_note(
        &self,
        meeting_id: MeetingId,
        text: &str,
        timestamp_ms: u32,
    ) -> Result<Note, DomainError>;

    /// List all notes for a meeting, ordered by `timestamp_ms`
    /// ascending. Returns an empty vec when no notes exist.
    async fn list_notes(&self, meeting_id: MeetingId) -> Result<Vec<Note>, DomainError>;

    /// Delete a single note by id. Returns `false` when the note did
    /// not exist.
    async fn delete_note(
        &self,
        note_id: crate::entities::note::NoteId,
    ) -> Result<bool, DomainError>;

    /// Flush WAL frames to the main database file. Called before
    /// shutdown so data written since the last automatic checkpoint is
    /// not lost when the process exits.
    ///
    /// Default impl is a no-op — only the SQLite adapter does real work.
    async fn checkpoint(&self) -> Result<(), DomainError> {
        Ok(())
    }

    /// Release any expensive resources (database pool, file handles)
    /// the adapter holds. Called once on app shutdown so the underlying
    /// storage layer can checkpoint cleanly.
    ///
    /// Default impl is a no-op so in-memory or test-only implementations
    /// don't need to care. The SQLite adapter overrides this to close
    /// its connection pool and flush WAL frames.
    async fn close(&self) {}
}
