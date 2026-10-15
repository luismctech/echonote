//! Meeting CRUD commands.

use tauri::State;

use crate::ipc_error::IpcError;

use echo_app::{CorrectMeeting, CorrectMeetingError};
use echo_domain::{
    Meeting, MeetingId, MeetingSearchHit, MeetingSummary, Note, NoteId, SegmentId, SpeakerId,
};

use super::{session_audio, AppState};

/// List meetings, newest first.
#[tauri::command]
#[specta::specta]
pub async fn list_meetings(
    state: State<'_, AppState>,
    limit: Option<u32>,
) -> Result<Vec<MeetingSummary>, IpcError> {
    state
        .store
        .list(limit.unwrap_or(0))
        .await
        .map_err(|e| IpcError::storage(format!("list meetings: {e}")))
}

/// Fetch a single meeting (header + segments). Returns `null` when
/// the id does not exist.
#[tauri::command]
#[specta::specta]
pub async fn get_meeting(
    state: State<'_, AppState>,
    id: MeetingId,
) -> Result<Option<Meeting>, IpcError> {
    state
        .store
        .get(id)
        .await
        .map_err(|e| IpcError::storage(format!("get meeting: {e}")))
}

/// Delete a meeting, its segments and its saved audio, if any.
/// Returns `true` when the row existed and was removed.
#[tauri::command]
#[specta::specta]
pub async fn delete_meeting(state: State<'_, AppState>, id: MeetingId) -> Result<bool, IpcError> {
    let deleted = state
        .store
        .delete(id)
        .await
        .map_err(|e| IpcError::storage(format!("delete meeting: {e}")))?;
    // The meeting is already gone; a leftover file is logged, not
    // reported as a failed delete.
    if let Err(e) = session_audio::delete_meeting_audio(&state.recordings_dir, id) {
        tracing::warn!(error = %e, meeting_id = %id, "could not delete the meeting audio");
    }
    Ok(deleted)
}

/// Full-text search over segment text. Returns one hit per meeting,
/// ordered by FTS5 BM25 rank (best match first). Empty / whitespace-
/// only queries return an empty vec — the frontend wires this to
/// `onChange` debounced, so the initial empty state never errors.
///
/// `limit` defaults to 20 (a comfortable sidebar page); `0` means no
/// cap. The query is sanitised inside the storage layer, so the
/// frontend can pass raw user input without worrying about FTS5
/// syntax characters.
#[tauri::command]
#[specta::specta]
pub async fn search_meetings(
    state: State<'_, AppState>,
    query: String,
    limit: Option<u32>,
) -> Result<Vec<MeetingSearchHit>, IpcError> {
    state
        .store
        .search(&query, limit.unwrap_or(20))
        .await
        .map_err(|e| IpcError::storage(format!("search meetings: {e}")))
}

/// Rename a meeting's title. Returns the updated summary so the
/// sidebar and detail view can re-render without an extra round-trip.
#[tauri::command]
#[specta::specta]
pub async fn rename_meeting(
    state: State<'_, AppState>,
    id: MeetingId,
    title: String,
) -> Result<MeetingSummary, IpcError> {
    let found = state
        .store
        .rename_meeting(id, &title)
        .await
        .map_err(|e| IpcError::storage(format!("rename meeting: {e}")))?;
    if !found {
        return Err(IpcError::not_found(format!("meeting {id} not found")));
    }
    // Return updated summary list entry so sidebar stays in sync.
    state
        .store
        .get(id)
        .await
        .map_err(|e| IpcError::storage(format!("reload meeting: {e}")))?
        .map(|m| m.summary)
        .ok_or_else(|| IpcError::not_found(format!("meeting {id} disappeared after rename")))
}

/// Rename a diarized speaker (or clear the label back to anonymous
/// by passing `null`/empty). Returns the freshly-loaded meeting so
/// the frontend can re-render speakers + segment chips from a single
/// source of truth without an extra `get_meeting` round-trip.
#[tauri::command]
#[specta::specta]
pub async fn rename_speaker(
    state: State<'_, AppState>,
    meeting_id: MeetingId,
    speaker_id: SpeakerId,
    label: Option<String>,
) -> Result<Meeting, IpcError> {
    state
        .rename_speaker
        .execute(meeting_id, speaker_id, label)
        .await
        .map_err(IpcError::from)?;
    state
        .store
        .get(meeting_id)
        .await
        .map_err(|e| IpcError::storage(format!("reload meeting: {e}")))?
        .ok_or_else(|| {
            IpcError::not_found(format!("meeting {meeting_id} disappeared after rename"))
        })
}

/// Map a correction failure to the IPC error the frontend expects.
/// Lives here rather than in `ipc_error` because only these commands
/// produce it.
fn correction_error(e: CorrectMeetingError) -> IpcError {
    match &e {
        CorrectMeetingError::NotFound(_) => IpcError::not_found(e.to_string()),
        CorrectMeetingError::Invalid(_) => IpcError::invalid_input(e.to_string()),
        CorrectMeetingError::Storage(_) => IpcError::storage(e.to_string()),
    }
}

/// Merge speaker `absorbed` into `surviving` in a saved meeting: all of
/// its segments move to `surviving` and its row is removed. The survivor
/// keeps its label, or inherits the absorbed label when it had none.
/// Irreversible; the UI confirms first. Resolves to `true` when anything
/// changed.
#[tauri::command]
#[specta::specta]
pub async fn merge_meeting_speakers(
    state: State<'_, AppState>,
    meeting_id: MeetingId,
    absorbed: SpeakerId,
    surviving: SpeakerId,
) -> Result<bool, IpcError> {
    CorrectMeeting::new(state.store.clone())
        .merge_speakers(meeting_id, absorbed, surviving)
        .await
        .map_err(correction_error)
}

/// Point the given segments of a saved meeting at `speaker_id`, or clear
/// their speaker with `null`. All ids must belong to the meeting; a
/// foreign id fails with `invalid_input` and writes nothing. Resolves to
/// the number of segments whose speaker changed.
#[tauri::command]
#[specta::specta]
pub async fn reassign_segments(
    state: State<'_, AppState>,
    meeting_id: MeetingId,
    segment_ids: Vec<SegmentId>,
    speaker_id: Option<SpeakerId>,
) -> Result<u32, IpcError> {
    CorrectMeeting::new(state.store.clone())
        .reassign_segments(meeting_id, &segment_ids, speaker_id)
        .await
        .map_err(correction_error)
}

/// Delete segments of a saved meeting (search index and segment count
/// stay consistent). Speakers left without segments are kept. A foreign
/// id fails with `invalid_input` and deletes nothing; already-deleted ids
/// are skipped. Irreversible; the UI confirms first. Resolves to the
/// number of segments deleted.
#[tauri::command]
#[specta::specta]
pub async fn delete_segments(
    state: State<'_, AppState>,
    meeting_id: MeetingId,
    segment_ids: Vec<SegmentId>,
) -> Result<u32, IpcError> {
    CorrectMeeting::new(state.store.clone())
        .delete_segments(meeting_id, &segment_ids)
        .await
        .map_err(correction_error)
}

/// Add a timestamped note to a meeting. `timestamp_ms` is relative to
/// the recording start (same timeline as segment offsets).
#[tauri::command]
#[specta::specta]
pub async fn add_note(
    state: State<'_, AppState>,
    meeting_id: MeetingId,
    text: String,
    timestamp_ms: u32,
) -> Result<Note, IpcError> {
    state
        .store
        .add_note(meeting_id, &text, timestamp_ms)
        .await
        .map_err(|e| IpcError::storage(format!("add note: {e}")))
}

/// List all notes for a meeting, ordered by recording timestamp.
#[tauri::command]
#[specta::specta]
pub async fn list_notes(
    state: State<'_, AppState>,
    meeting_id: MeetingId,
) -> Result<Vec<Note>, IpcError> {
    state
        .store
        .list_notes(meeting_id)
        .await
        .map_err(|e| IpcError::storage(format!("list notes: {e}")))
}

/// Delete a single note by id.
#[tauri::command]
#[specta::specta]
pub async fn delete_note(state: State<'_, AppState>, note_id: NoteId) -> Result<bool, IpcError> {
    state
        .store
        .delete_note(note_id)
        .await
        .map_err(|e| IpcError::storage(format!("delete note: {e}")))
}
