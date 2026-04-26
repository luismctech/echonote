//! Meeting CRUD commands.

use tauri::State;

use crate::ipc_error::IpcError;

use echo_domain::{Meeting, MeetingId, MeetingSearchHit, MeetingSummary, SpeakerId};

use super::AppState;

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

/// Delete a meeting and its segments. Returns `true` when the row
/// existed and was removed.
#[tauri::command]
#[specta::specta]
pub async fn delete_meeting(state: State<'_, AppState>, id: MeetingId) -> Result<bool, IpcError> {
    state
        .store
        .delete(id)
        .await
        .map_err(|e| IpcError::storage(format!("delete meeting: {e}")))
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
