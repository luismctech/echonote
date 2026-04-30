//! LLM commands: summarize, get summary, chat.

use tauri::ipc::Channel;
use tauri::State;

use crate::ipc_error::IpcError;

use echo_app::{AskAboutMeeting, AskAboutMeetingEvent, SummarizeEvent, SummarizeMeeting};
use echo_domain::{ChatMessage, CustomTemplateId, MeetingId, Summary};
use futures::stream::StreamExt;

use super::AppState;

/// Generate (or regenerate) the local-LLM summary for a meeting.
///
/// `template` selects the prompt template: `"general"` (default),
/// `"oneOnOne"`, `"sprintReview"`, `"interview"`, `"salesCall"`, or
/// `"lecture"`. Passing `null` or omitting the field defaults to
/// `"general"`.
#[tauri::command]
#[specta::specta]
pub async fn summarize_meeting(
    state: State<'_, AppState>,
    meeting_id: MeetingId,
    template: Option<String>,
    include_notes: bool,
) -> Result<Summary, IpcError> {
    let llm = state.ensure_llm().await?;
    let use_case = SummarizeMeeting::new(llm, state.store.clone());
    let tmpl = template.as_deref().unwrap_or("general");
    use_case
        .execute(meeting_id, tmpl, include_notes)
        .await
        .map_err(IpcError::from)
}

/// Fetch the most recent summary for a meeting, or `null` when none
/// has been generated yet. The frontend uses this on `MeetingDetail`
/// mount so the panel can render either the existing summary or the
/// "Generate" affordance without a redundant generate call.
#[tauri::command]
#[specta::specta]
pub async fn get_summary(
    state: State<'_, AppState>,
    meeting_id: MeetingId,
) -> Result<Option<Summary>, IpcError> {
    state
        .store
        .get_summary(meeting_id)
        .await
        .map_err(|e| IpcError::storage(format!("get summary: {e}")))
}

/// Generate a summary using a user-defined custom template.
///
/// Loads the custom template by `template_id`, then runs the LLM with
/// the user's prompt. The result is stored as
/// [`echo_domain::SummaryContent::Custom`].
#[tauri::command]
#[specta::specta]
pub async fn summarize_with_custom_template(
    state: State<'_, AppState>,
    meeting_id: MeetingId,
    template_id: CustomTemplateId,
    include_notes: bool,
) -> Result<Summary, IpcError> {
    // Load the custom template from disk.
    let templates = super::templates::read_templates_from(&state)?;
    let custom = templates
        .iter()
        .find(|t| t.id == template_id)
        .ok_or_else(|| IpcError::not_found(format!("custom template {template_id} not found")))?;

    let llm = state.ensure_llm().await?;
    let use_case = SummarizeMeeting::new(llm, state.store.clone());
    use_case
        .execute_custom(meeting_id, custom, include_notes)
        .await
        .map_err(IpcError::from)
}

/// Run one chat turn against a meeting's transcript.
///
/// Each invocation streams the assistant's reply token-by-token through
/// `on_event` and resolves the IPC promise once the stream terminates
/// (with [`AskAboutMeetingEvent::Finished`] on success or
/// [`AskAboutMeetingEvent::Failed`] on a mid-decode error).
///
/// ## Lifecycle
///
/// 1. Pre-stream errors (meeting not found, empty question, model
///    not loaded) come back as `Err(String)` so the UI can show a
///    toast without parsing event variants.
/// 2. Once the stream is open, **every** terminal condition travels
///    as a final event on the channel: success → `Finished`,
///    mid-decode failure → `Failed`. The promise then resolves
///    `Ok(())`.
/// 3. Cancellation is automatic: when the React component unmounts
///    (or the user closes the chat panel), Tauri drops the
///    `Channel`. The next `on_event.send` returns `Err(_)`, this
///    function exits its drain loop, the `BoxStream` is dropped, the
///    underlying `mpsc::Receiver` closes and the blocking decoder
///    thread inside `LlamaCppChat` exits on the next
///    `tx.blocking_send` failure. No explicit `cancel_chat` command
///    is needed — symmetric with `start_streaming` was deliberately
///    not chosen because chat is bounded (one Q&A turn ≤ ~10s on
///    Qwen 14B, vs streaming sessions that can run hours).
///
/// ## Why we don't spawn a background task
///
/// Unlike `start_streaming`, which returns a session id and runs
/// indefinitely until `stop_streaming`, a chat turn is a single
/// bounded interaction. Draining the stream inline keeps the IPC
/// surface minimal (one command, one channel, one promise) and means
/// any error gets observed by the calling `await` instead of escaping
/// into a detached `tokio::spawn`.
#[tauri::command]
#[specta::specta]
pub async fn ask_about_meeting(
    state: State<'_, AppState>,
    meeting_id: MeetingId,
    history: Option<Vec<ChatMessage>>,
    question: String,
    on_event: Channel<AskAboutMeetingEvent>,
) -> Result<(), IpcError> {
    let chat = state.ensure_chat().await?;
    let use_case = AskAboutMeeting::new(chat, state.store.clone());

    let mut stream = use_case
        .execute(meeting_id, history.unwrap_or_default(), question)
        .await
        .map_err(IpcError::from)?;

    while let Some(event) = stream.next().await {
        if let Err(e) = on_event.send(event) {
            tracing::warn!(
                error = %e,
                %meeting_id,
                "ask_about_meeting channel send failed; aborting drain",
            );
            break;
        }
    }
    Ok(())
}

/// Streaming variant of [`summarize_meeting`]. Sends tokens as they
/// are decoded so the UI can render them incrementally — same UX as
/// the chat feature.
///
/// The stream finishes with `SummarizeEvent::Completed` carrying the
/// persisted [`Summary`], or `SummarizeEvent::Failed` on error. The
/// IPC promise resolves `Ok(())` in both cases; the frontend reads
/// the terminal event to decide success or failure.
#[tauri::command]
#[specta::specta]
pub async fn summarize_meeting_stream(
    state: State<'_, AppState>,
    meeting_id: MeetingId,
    template: Option<String>,
    include_notes: bool,
    language: Option<String>,
    on_event: Channel<SummarizeEvent>,
) -> Result<(), IpcError> {
    let llm = state.ensure_llm().await?;
    let use_case = SummarizeMeeting::new(llm, state.store.clone());
    let tmpl = template.as_deref().unwrap_or("general");

    let mut stream = use_case
        .execute_stream(meeting_id, tmpl, include_notes, language.as_deref())
        .await
        .map_err(IpcError::from)?;

    while let Some(event) = stream.next().await {
        if let Err(e) = on_event.send(event) {
            tracing::warn!(
                error = %e,
                %meeting_id,
                "summarize_meeting_stream channel send failed; aborting drain",
            );
            break;
        }
    }
    Ok(())
}
