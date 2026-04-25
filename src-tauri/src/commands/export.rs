//! Meeting export command.
//!
//! Path validation and file I/O live here (shell concern).
//! Rendering is delegated to [`echo_app::use_cases::export`].

use std::path::PathBuf;

use serde::Deserialize;
use tauri::State;

use crate::ipc_error::{ErrorCode, IpcError};

use echo_app::use_cases::export as rendering;
use echo_domain::MeetingId;

use super::AppState;

/// Supported export formats (mirrored for specta/serde on the IPC boundary).
#[derive(Debug, Clone, Copy, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub enum ExportFormat {
    Markdown,
    Txt,
}

impl From<ExportFormat> for rendering::ExportFormat {
    fn from(f: ExportFormat) -> Self {
        match f {
            ExportFormat::Markdown => rendering::ExportFormat::Markdown,
            ExportFormat::Txt => rendering::ExportFormat::Txt,
        }
    }
}

/// Export a meeting (with optional summary) to a file at `dest_path`.
///
/// The frontend is responsible for showing the save-file dialog (via
/// `@tauri-apps/plugin-dialog`) and passing the chosen path here. This
/// command generates the formatted content and writes it atomically.
///
/// **Security:** `dest_path` is validated to be inside the user's home
/// directory and must not contain path-traversal components (`..`).
#[tauri::command]
#[specta::specta]
pub async fn export_meeting(
    state: State<'_, AppState>,
    meeting_id: MeetingId,
    format: ExportFormat,
    dest_path: String,
) -> Result<(), IpcError> {
    // ── Path validation ──────────────────────────────────────────
    let dest = PathBuf::from(&dest_path);
    if !dest.is_absolute() {
        return Err(IpcError::new(
            ErrorCode::InvalidInput,
            "export path must be absolute",
        ));
    }
    // Reject explicit traversal components before canonicalizing.
    if dest
        .components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return Err(IpcError::new(
            ErrorCode::InvalidInput,
            "export path must not contain '..' components",
        ));
    }
    let home =
        dirs::home_dir().ok_or_else(|| IpcError::internal("cannot determine home directory"))?;
    // Canonicalize the parent (the file itself may not exist yet).
    let parent = dest.parent().ok_or_else(|| {
        IpcError::new(
            ErrorCode::InvalidInput,
            "export path has no parent directory",
        )
    })?;
    let canonical_parent = parent
        .canonicalize()
        .map_err(|e| IpcError::storage(format!("invalid export directory: {e}")))?;
    if !canonical_parent.starts_with(&home) {
        return Err(IpcError::new(
            ErrorCode::InvalidInput,
            "export path must be within the home directory",
        ));
    }
    let safe_dest = canonical_parent
        .join(dest.file_name().ok_or_else(|| {
            IpcError::new(ErrorCode::InvalidInput, "export path has no filename")
        })?);

    // ── Generate + write ─────────────────────────────────────────
    let meeting = state
        .store
        .get(meeting_id)
        .await
        .map_err(|e| IpcError::storage(format!("get meeting: {e}")))?
        .ok_or_else(|| IpcError::not_found(format!("meeting {meeting_id} not found")))?;

    let summary = state
        .store
        .get_summary(meeting_id)
        .await
        .map_err(|e| IpcError::storage(format!("get summary: {e}")))?;

    let content = rendering::render(&meeting, summary.as_ref(), format.into());

    tokio::fs::write(&safe_dest, content.as_bytes())
        .await
        .map_err(|e| IpcError::storage(format!("write file: {e}")))?;

    Ok(())
}
