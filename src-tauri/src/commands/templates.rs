//! Custom template CRUD commands.
//!
//! User-defined summary templates are stored as a JSON array in
//! `<data_root>/custom_templates.json`. The file is read/written
//! atomically on each operation — the list is small (tens of items
//! at most) so there's no need for a SQLite table.

use tauri::State;

use crate::ipc_error::{ErrorCode, IpcError};

use echo_domain::{CustomTemplate, CustomTemplateId};

use super::AppState;

/// Hard ceiling on the number of custom templates a user can create.
/// Prevents unbounded growth of the JSON file on disk.
const MAX_TEMPLATES: usize = 50;

/// Maximum length (in chars) for a template name.
const MAX_NAME_LEN: usize = 200;

/// Maximum length (in chars) for a template prompt.
const MAX_PROMPT_LEN: usize = 10_000;

/// Path to the custom templates JSON file.
fn templates_path(state: &AppState) -> std::path::PathBuf {
    state.data_root.join("custom_templates.json")
}

/// Read all custom templates from disk.
fn read_templates(state: &AppState) -> Result<Vec<CustomTemplate>, IpcError> {
    read_templates_from(state)
}

/// Read all custom templates — public within the crate so other
/// command modules (e.g. `llm::summarize_with_custom_template`)
/// can look up a template by id.
pub(crate) fn read_templates_from(state: &AppState) -> Result<Vec<CustomTemplate>, IpcError> {
    let path = templates_path(state);
    if !path.exists() {
        return Ok(Vec::new());
    }
    let data = std::fs::read_to_string(&path)
        .map_err(|e| IpcError::storage(format!("read custom templates: {e}")))?;
    serde_json::from_str(&data)
        .map_err(|e| IpcError::storage(format!("parse custom templates: {e}")))
}

/// Write all custom templates to disk atomically.
fn write_templates(state: &AppState, templates: &[CustomTemplate]) -> Result<(), IpcError> {
    let path = templates_path(state);
    let data = serde_json::to_string_pretty(templates)
        .map_err(|e| IpcError::storage(format!("serialize custom templates: {e}")))?;
    // Write to a temp file first, then rename for atomicity.
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, data.as_bytes())
        .map_err(|e| IpcError::storage(format!("write custom templates: {e}")))?;
    std::fs::rename(&tmp, &path)
        .map_err(|e| IpcError::storage(format!("rename custom templates: {e}")))?;
    Ok(())
}

/// List all user-defined custom templates.
#[tauri::command]
#[specta::specta]
pub fn list_custom_templates(state: State<'_, AppState>) -> Result<Vec<CustomTemplate>, IpcError> {
    read_templates(&state)
}

/// Create a new custom template.
///
/// The `name` and `prompt` fields are required and must be non-empty.
/// Returns the newly created template with its generated id.
#[tauri::command]
#[specta::specta]
pub fn create_custom_template(
    state: State<'_, AppState>,
    name: String,
    prompt: String,
) -> Result<CustomTemplate, IpcError> {
    let name = name.trim().to_string();
    let prompt = prompt.trim().to_string();
    if name.is_empty() {
        return Err(IpcError::new(
            ErrorCode::InvalidInput,
            "template name cannot be empty".to_string(),
        ));
    }
    if prompt.is_empty() {
        return Err(IpcError::new(
            ErrorCode::InvalidInput,
            "template prompt cannot be empty".to_string(),
        ));
    }
    if name.len() > MAX_NAME_LEN {
        return Err(IpcError::new(
            ErrorCode::InvalidInput,
            format!("template name exceeds {MAX_NAME_LEN} characters"),
        ));
    }
    if prompt.len() > MAX_PROMPT_LEN {
        return Err(IpcError::new(
            ErrorCode::InvalidInput,
            format!("template prompt exceeds {MAX_PROMPT_LEN} characters"),
        ));
    }

    let template = CustomTemplate {
        id: CustomTemplateId::new(),
        name,
        prompt,
    };

    let mut templates = read_templates(&state)?;
    if templates.len() >= MAX_TEMPLATES {
        return Err(IpcError::new(
            ErrorCode::InvalidInput,
            format!("maximum number of custom templates ({MAX_TEMPLATES}) reached"),
        ));
    }
    templates.push(template.clone());
    write_templates(&state, &templates)?;

    Ok(template)
}

/// Update an existing custom template's name and/or prompt.
#[tauri::command]
#[specta::specta]
pub fn update_custom_template(
    state: State<'_, AppState>,
    id: CustomTemplateId,
    name: String,
    prompt: String,
) -> Result<CustomTemplate, IpcError> {
    let name = name.trim().to_string();
    let prompt = prompt.trim().to_string();
    if name.is_empty() {
        return Err(IpcError::new(
            ErrorCode::InvalidInput,
            "template name cannot be empty".to_string(),
        ));
    }
    if prompt.is_empty() {
        return Err(IpcError::new(
            ErrorCode::InvalidInput,
            "template prompt cannot be empty".to_string(),
        ));
    }
    if name.len() > MAX_NAME_LEN {
        return Err(IpcError::new(
            ErrorCode::InvalidInput,
            format!("template name exceeds {MAX_NAME_LEN} characters"),
        ));
    }
    if prompt.len() > MAX_PROMPT_LEN {
        return Err(IpcError::new(
            ErrorCode::InvalidInput,
            format!("template prompt exceeds {MAX_PROMPT_LEN} characters"),
        ));
    }

    let mut templates = read_templates(&state)?;
    let tmpl = templates
        .iter_mut()
        .find(|t| t.id == id)
        .ok_or_else(|| IpcError::not_found(format!("custom template {id} not found")))?;

    tmpl.name = name;
    tmpl.prompt = prompt;
    let updated = tmpl.clone();
    write_templates(&state, &templates)?;

    Ok(updated)
}

/// Delete a custom template by id.
#[tauri::command]
#[specta::specta]
pub fn delete_custom_template(
    state: State<'_, AppState>,
    id: CustomTemplateId,
) -> Result<(), IpcError> {
    let mut templates = read_templates(&state)?;
    let before = templates.len();
    templates.retain(|t| t.id != id);
    if templates.len() == before {
        return Err(IpcError::not_found(format!(
            "custom template {id} not found"
        )));
    }
    write_templates(&state, &templates)?;
    Ok(())
}
