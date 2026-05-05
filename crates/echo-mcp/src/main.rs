//! EchoNote MCP Server — exposes meeting data to LLM clients via the
//! Model Context Protocol (JSON-RPC over stdio).
//!
//! This binary opens the EchoNote SQLite database and serves tools
//! that allow MCP clients (Claude Desktop, VS Code Copilot, Cursor,
//! etc.) to query, annotate and export meetings, transcripts,
//! summaries, speakers, notes, and custom templates.
//!
//! Usage:
//!   echo-mcp --db /path/to/echonote.db

use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use echo_app::use_cases::export::{self, ExportFormat};
use echo_domain::{CustomTemplate, MeetingId, MeetingStore, NoteId, SpeakerId};
use echo_storage::SqliteMeetingStore;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// JSON-RPC types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Serialize)]
struct JsonRpcResponse {
    jsonrpc: &'static str,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

impl JsonRpcResponse {
    fn ok(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: Some(result),
            error: None,
        }
    }
    fn err(id: Value, code: i32, msg: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: msg.into(),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// MCP protocol types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ServerInfo {
    name: String,
    version: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct InitializeResult {
    protocol_version: String,
    capabilities: Capabilities,
    server_info: ServerInfo,
}

#[derive(Serialize)]
struct Capabilities {
    tools: ToolsCapability,
}

#[derive(Serialize)]
struct ToolsCapability {}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ToolDef {
    name: String,
    description: String,
    input_schema: Value,
}

#[derive(Serialize)]
struct ToolsListResult {
    tools: Vec<ToolDef>,
}

#[derive(Serialize)]
struct ToolCallResult {
    content: Vec<ContentBlock>,
    #[serde(rename = "isError", skip_serializing_if = "Option::is_none")]
    is_error: Option<bool>,
}

#[derive(Serialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    kind: String,
    text: String,
}

fn text_content(text: String) -> Vec<ContentBlock> {
    vec![ContentBlock {
        kind: "text".into(),
        text,
    }]
}

// ---------------------------------------------------------------------------
// Tool definitions
// ---------------------------------------------------------------------------

fn tool_definitions() -> Vec<ToolDef> {
    vec![
        // ---- Read: Meetings ------------------------------------------------
        ToolDef {
            name: "echonote_list_meetings".into(),
            description: "List recent meetings with title, date, duration and segment count. Returns newest first.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max meetings to return (0 = no limit, default 20)"
                    }
                }
            }),
        },
        ToolDef {
            name: "echonote_get_meeting".into(),
            description: "Get the full transcript of a meeting including all segments with timestamps and speaker labels.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "meeting_id": {
                        "type": "string",
                        "description": "UUID of the meeting"
                    }
                },
                "required": ["meeting_id"]
            }),
        },
        ToolDef {
            name: "echonote_search_meetings".into(),
            description: "Full-text search across all meeting transcripts. Returns matching meetings with highlighted snippets.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (words or phrases)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 10)"
                    }
                },
                "required": ["query"]
            }),
        },
        ToolDef {
            name: "echonote_get_summary".into(),
            description: "Get the AI-generated summary of a meeting, if one exists.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "meeting_id": {
                        "type": "string",
                        "description": "UUID of the meeting"
                    }
                },
                "required": ["meeting_id"]
            }),
        },
        ToolDef {
            name: "echonote_list_notes".into(),
            description: "List user-written notes for a meeting, with timestamps relative to the recording start.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "meeting_id": {
                        "type": "string",
                        "description": "UUID of the meeting"
                    }
                },
                "required": ["meeting_id"]
            }),
        },
        // ---- Read: Speakers ------------------------------------------------
        ToolDef {
            name: "echonote_get_speakers".into(),
            description: "List all diarized speakers for a meeting with their id, slot number, and optional label.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "meeting_id": {
                        "type": "string",
                        "description": "UUID of the meeting"
                    }
                },
                "required": ["meeting_id"]
            }),
        },
        // ---- Read: Export --------------------------------------------------
        ToolDef {
            name: "echonote_export_meeting".into(),
            description: "Export a meeting (with optional summary) as formatted Markdown or plain text. Returns the rendered content as a string.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "meeting_id": {
                        "type": "string",
                        "description": "UUID of the meeting"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["markdown", "txt"],
                        "description": "Export format (default: markdown)"
                    }
                },
                "required": ["meeting_id"]
            }),
        },
        // ---- Read: Templates -----------------------------------------------
        ToolDef {
            name: "echonote_list_templates".into(),
            description: "List user-defined custom summary templates (id, name, prompt).".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
        // ---- Read: Status --------------------------------------------------
        ToolDef {
            name: "echonote_get_status".into(),
            description: "Get EchoNote status: total meetings count and available template ids.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
        // ---- Write: Meetings -----------------------------------------------
        ToolDef {
            name: "echonote_rename_meeting".into(),
            description: "Change the display title of a meeting.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "meeting_id": {
                        "type": "string",
                        "description": "UUID of the meeting"
                    },
                    "title": {
                        "type": "string",
                        "description": "New title for the meeting"
                    }
                },
                "required": ["meeting_id", "title"]
            }),
        },
        ToolDef {
            name: "echonote_delete_meeting".into(),
            description: "Permanently delete a meeting and all its segments, speakers, notes and summary. This action cannot be undone.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "meeting_id": {
                        "type": "string",
                        "description": "UUID of the meeting to delete"
                    }
                },
                "required": ["meeting_id"]
            }),
        },
        // ---- Write: Speakers -----------------------------------------------
        ToolDef {
            name: "echonote_rename_speaker".into(),
            description: "Assign a display name to a diarized speaker (e.g. 'Speaker 1' → 'Alice'). Pass null as label to reset to the default.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "meeting_id": {
                        "type": "string",
                        "description": "UUID of the meeting"
                    },
                    "speaker_id": {
                        "type": "string",
                        "description": "UUID of the speaker"
                    },
                    "label": {
                        "type": ["string", "null"],
                        "description": "New display name, or null to reset to default"
                    }
                },
                "required": ["meeting_id", "speaker_id", "label"]
            }),
        },
        // ---- Write: Notes --------------------------------------------------
        ToolDef {
            name: "echonote_add_note".into(),
            description: "Create a timestamped text note on a meeting. The timestamp is relative to the meeting start in milliseconds.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "meeting_id": {
                        "type": "string",
                        "description": "UUID of the meeting"
                    },
                    "text": {
                        "type": "string",
                        "description": "Note text content"
                    },
                    "timestamp_ms": {
                        "type": "integer",
                        "description": "Offset in milliseconds from meeting start (default 0)"
                    }
                },
                "required": ["meeting_id", "text"]
            }),
        },
        ToolDef {
            name: "echonote_delete_note".into(),
            description: "Delete a single note by its id.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "note_id": {
                        "type": "string",
                        "description": "UUID of the note to delete"
                    }
                },
                "required": ["note_id"]
            }),
        },
    ]
}

// ---------------------------------------------------------------------------
// Tool execution
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse a `meeting_id` string parameter into a [`MeetingId`].
fn parse_meeting_id(args: &Value) -> Result<MeetingId, ToolCallResult> {
    let id_str = args
        .get("meeting_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ToolCallResult {
            content: text_content("Missing required parameter: meeting_id".into()),
            is_error: Some(true),
        })?;
    Uuid::parse_str(id_str)
        .map(MeetingId)
        .map_err(|_| ToolCallResult {
            content: text_content(format!("Invalid UUID: {id_str}")),
            is_error: Some(true),
        })
}

/// Resolve the app data directory (same path convention as the Tauri app).
fn data_root() -> Option<PathBuf> {
    dirs::data_dir().map(|d| d.join("app.echonote.desktop"))
}

/// Read custom templates from the JSON file on disk.
fn read_custom_templates() -> Vec<CustomTemplate> {
    let Some(root) = data_root() else {
        return Vec::new();
    };
    let path = root.join("custom_templates.json");
    if !path.exists() {
        return Vec::new();
    }
    let Ok(data) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    serde_json::from_str(&data).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Tool execution
// ---------------------------------------------------------------------------

async fn call_tool(store: &SqliteMeetingStore, name: &str, args: &Value) -> ToolCallResult {
    match name {
        "echonote_list_meetings" => {
            let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(20) as u32;
            match store.list(limit).await {
                Ok(meetings) => {
                    let out: Vec<Value> = meetings
                        .iter()
                        .map(|m| {
                            serde_json::json!({
                                "id": m.id.to_string(),
                                "title": m.title,
                                "started_at": m.started_at.to_string(),
                                "duration_ms": m.duration_ms,
                                "segment_count": m.segment_count,
                                "language": m.language,
                            })
                        })
                        .collect();
                    ToolCallResult {
                        content: text_content(
                            serde_json::to_string_pretty(&out).unwrap_or_default(),
                        ),
                        is_error: None,
                    }
                }
                Err(e) => ToolCallResult {
                    content: text_content(format!("Error: {e}")),
                    is_error: Some(true),
                },
            }
        }
        "echonote_get_meeting" => {
            let id = match parse_meeting_id(args) {
                Ok(id) => id,
                Err(e) => return e,
            };
            match store.get(id).await {
                Ok(Some(meeting)) => {
                    let mut transcript = String::new();
                    transcript.push_str(&format!("# {}\n", meeting.summary.title));
                    transcript.push_str(&format!("Started: {}\n", meeting.summary.started_at));
                    transcript
                        .push_str(&format!("Duration: {} ms\n\n", meeting.summary.duration_ms));

                    let speaker_names: HashMap<_, _> = meeting
                        .speakers
                        .iter()
                        .map(|s| {
                            (
                                s.id,
                                s.label
                                    .clone()
                                    .unwrap_or_else(|| format!("Speaker {}", s.slot)),
                            )
                        })
                        .collect();

                    for seg in &meeting.segments {
                        let speaker = seg
                            .speaker_id
                            .and_then(|sid| speaker_names.get(&sid))
                            .map(|s| s.as_str())
                            .unwrap_or("Unknown");
                        let start_s = seg.start_ms as f64 / 1000.0;
                        let end_s = seg.end_ms as f64 / 1000.0;
                        transcript.push_str(&format!(
                            "[{start_s:.1}s–{end_s:.1}s] {speaker}: {}\n",
                            seg.text
                        ));
                    }

                    ToolCallResult {
                        content: text_content(transcript),
                        is_error: None,
                    }
                }
                Ok(None) => ToolCallResult {
                    content: text_content(format!("Meeting not found: {}", id.0)),
                    is_error: Some(true),
                },
                Err(e) => ToolCallResult {
                    content: text_content(format!("Error: {e}")),
                    is_error: Some(true),
                },
            }
        }
        "echonote_search_meetings" => {
            let Some(query) = args.get("query").and_then(|v| v.as_str()) else {
                return ToolCallResult {
                    content: text_content("Missing required parameter: query".into()),
                    is_error: Some(true),
                };
            };
            let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as u32;
            match store.search(query, limit).await {
                Ok(hits) => {
                    let out: Vec<Value> = hits
                        .iter()
                        .map(|h| {
                            serde_json::json!({
                                "id": h.meeting.id.to_string(),
                                "title": h.meeting.title,
                                "started_at": h.meeting.started_at.to_string(),
                                "snippet": h.snippet,
                                "rank": h.rank,
                            })
                        })
                        .collect();
                    ToolCallResult {
                        content: text_content(
                            serde_json::to_string_pretty(&out).unwrap_or_default(),
                        ),
                        is_error: None,
                    }
                }
                Err(e) => ToolCallResult {
                    content: text_content(format!("Error: {e}")),
                    is_error: Some(true),
                },
            }
        }
        "echonote_get_summary" => {
            let id = match parse_meeting_id(args) {
                Ok(id) => id,
                Err(e) => return e,
            };
            match store.get_summary(id).await {
                Ok(Some(summary)) => {
                    let text = serde_json::to_string_pretty(&serde_json::json!({
                        "model": summary.model,
                        "language": summary.language,
                        "created_at": summary.created_at.to_string(),
                        "content": format!("{:?}", summary.content),
                    }))
                    .unwrap_or_default();
                    ToolCallResult {
                        content: text_content(text),
                        is_error: None,
                    }
                }
                Ok(None) => ToolCallResult {
                    content: text_content(format!("No summary found for meeting {}", id.0)),
                    is_error: None,
                },
                Err(e) => ToolCallResult {
                    content: text_content(format!("Error: {e}")),
                    is_error: Some(true),
                },
            }
        }
        "echonote_list_notes" => {
            let id = match parse_meeting_id(args) {
                Ok(id) => id,
                Err(e) => return e,
            };
            match store.list_notes(id).await {
                Ok(notes) => {
                    let out: Vec<Value> = notes
                        .iter()
                        .map(|n| {
                            serde_json::json!({
                                "id": n.id.to_string(),
                                "text": n.text,
                                "timestamp_ms": n.timestamp_ms,
                                "created_at": n.created_at,
                            })
                        })
                        .collect();
                    ToolCallResult {
                        content: text_content(
                            serde_json::to_string_pretty(&out).unwrap_or_default(),
                        ),
                        is_error: None,
                    }
                }
                Err(e) => ToolCallResult {
                    content: text_content(format!("Error: {e}")),
                    is_error: Some(true),
                },
            }
        }
        // ---- NEW: Get speakers ---------------------------------------------
        "echonote_get_speakers" => {
            let id = match parse_meeting_id(args) {
                Ok(id) => id,
                Err(e) => return e,
            };
            match store.list_speakers(id).await {
                Ok(speakers) => {
                    let out: Vec<Value> = speakers.iter().map(|s| serde_json::json!({
                        "id": s.id.to_string(),
                        "slot": s.slot,
                        "label": s.label.as_deref().unwrap_or(&format!("Speaker {}", s.slot + 1)),
                    })).collect();
                    ToolCallResult {
                        content: text_content(
                            serde_json::to_string_pretty(&out).unwrap_or_default(),
                        ),
                        is_error: None,
                    }
                }
                Err(e) => ToolCallResult {
                    content: text_content(format!("Error: {e}")),
                    is_error: Some(true),
                },
            }
        }
        // ---- NEW: Export meeting --------------------------------------------
        "echonote_export_meeting" => {
            let id = match parse_meeting_id(args) {
                Ok(id) => id,
                Err(e) => return e,
            };
            let format = match args.get("format").and_then(|v| v.as_str()) {
                Some("txt") => ExportFormat::Txt,
                _ => ExportFormat::Markdown,
            };
            let meeting = match store.get(id).await {
                Ok(Some(m)) => m,
                Ok(None) => {
                    return ToolCallResult {
                        content: text_content(format!("Meeting not found: {}", id.0)),
                        is_error: Some(true),
                    }
                }
                Err(e) => {
                    return ToolCallResult {
                        content: text_content(format!("Error: {e}")),
                        is_error: Some(true),
                    }
                }
            };
            let summary = store.get_summary(id).await.ok().flatten();
            let rendered = export::render(&meeting, summary.as_ref(), format);
            ToolCallResult {
                content: text_content(rendered),
                is_error: None,
            }
        }
        // ---- NEW: List templates -------------------------------------------
        "echonote_list_templates" => {
            let templates = read_custom_templates();
            let out: Vec<Value> = templates
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "id": t.id.to_string(),
                        "name": t.name,
                        "prompt": t.prompt,
                    })
                })
                .collect();
            ToolCallResult {
                content: text_content(serde_json::to_string_pretty(&out).unwrap_or_default()),
                is_error: None,
            }
        }
        // ---- NEW: Get status -----------------------------------------------
        "echonote_get_status" => {
            let meeting_count = store.list(0).await.map(|v| v.len()).unwrap_or(0);
            let templates = read_custom_templates();
            let built_in: Vec<&str> = echo_domain::TEMPLATE_IDS.to_vec();
            let out = serde_json::json!({
                "version": env!("CARGO_PKG_VERSION"),
                "total_meetings": meeting_count,
                "built_in_templates": built_in,
                "custom_templates_count": templates.len(),
            });
            ToolCallResult {
                content: text_content(serde_json::to_string_pretty(&out).unwrap_or_default()),
                is_error: None,
            }
        }
        // ---- NEW: Rename meeting -------------------------------------------
        "echonote_rename_meeting" => {
            let id = match parse_meeting_id(args) {
                Ok(id) => id,
                Err(e) => return e,
            };
            let Some(title) = args.get("title").and_then(|v| v.as_str()) else {
                return ToolCallResult {
                    content: text_content("Missing required parameter: title".into()),
                    is_error: Some(true),
                };
            };
            if title.trim().is_empty() {
                return ToolCallResult {
                    content: text_content("Title cannot be empty".into()),
                    is_error: Some(true),
                };
            }
            match store.rename_meeting(id, title).await {
                Ok(true) => ToolCallResult {
                    content: text_content(format!("Meeting renamed to: {title}")),
                    is_error: None,
                },
                Ok(false) => ToolCallResult {
                    content: text_content(format!("Meeting not found: {}", id.0)),
                    is_error: Some(true),
                },
                Err(e) => ToolCallResult {
                    content: text_content(format!("Error: {e}")),
                    is_error: Some(true),
                },
            }
        }
        // ---- NEW: Delete meeting -------------------------------------------
        "echonote_delete_meeting" => {
            let id = match parse_meeting_id(args) {
                Ok(id) => id,
                Err(e) => return e,
            };
            match store.delete(id).await {
                Ok(true) => ToolCallResult {
                    content: text_content(format!("Meeting {} deleted", id.0)),
                    is_error: None,
                },
                Ok(false) => ToolCallResult {
                    content: text_content(format!("Meeting not found: {}", id.0)),
                    is_error: Some(true),
                },
                Err(e) => ToolCallResult {
                    content: text_content(format!("Error: {e}")),
                    is_error: Some(true),
                },
            }
        }
        // ---- NEW: Rename speaker -------------------------------------------
        "echonote_rename_speaker" => {
            let id = match parse_meeting_id(args) {
                Ok(id) => id,
                Err(e) => return e,
            };
            let Some(sid_str) = args.get("speaker_id").and_then(|v| v.as_str()) else {
                return ToolCallResult {
                    content: text_content("Missing required parameter: speaker_id".into()),
                    is_error: Some(true),
                };
            };
            let Ok(speaker_id) = Uuid::parse_str(sid_str).map(SpeakerId) else {
                return ToolCallResult {
                    content: text_content(format!("Invalid speaker UUID: {sid_str}")),
                    is_error: Some(true),
                };
            };
            let label = match args.get("label") {
                Some(Value::String(s)) => Some(s.as_str()),
                Some(Value::Null) | None => None,
                _ => {
                    return ToolCallResult {
                        content: text_content("Parameter 'label' must be a string or null".into()),
                        is_error: Some(true),
                    }
                }
            };
            match store.rename_speaker(id, speaker_id, label).await {
                Ok(true) => {
                    let msg = match label {
                        Some(l) => format!("Speaker renamed to: {l}"),
                        None => "Speaker label reset to default".into(),
                    };
                    ToolCallResult {
                        content: text_content(msg),
                        is_error: None,
                    }
                }
                Ok(false) => ToolCallResult {
                    content: text_content(format!(
                        "Speaker {sid_str} not found in meeting {}",
                        id.0
                    )),
                    is_error: Some(true),
                },
                Err(e) => ToolCallResult {
                    content: text_content(format!("Error: {e}")),
                    is_error: Some(true),
                },
            }
        }
        // ---- NEW: Add note -------------------------------------------------
        "echonote_add_note" => {
            let id = match parse_meeting_id(args) {
                Ok(id) => id,
                Err(e) => return e,
            };
            let Some(text) = args.get("text").and_then(|v| v.as_str()) else {
                return ToolCallResult {
                    content: text_content("Missing required parameter: text".into()),
                    is_error: Some(true),
                };
            };
            if text.trim().is_empty() {
                return ToolCallResult {
                    content: text_content("Note text cannot be empty".into()),
                    is_error: Some(true),
                };
            }
            let ts = args
                .get("timestamp_ms")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            match store.add_note(id, text, ts).await {
                Ok(note) => {
                    let out = serde_json::json!({
                        "id": note.id.to_string(),
                        "text": note.text,
                        "timestamp_ms": note.timestamp_ms,
                        "created_at": note.created_at,
                    });
                    ToolCallResult {
                        content: text_content(
                            serde_json::to_string_pretty(&out).unwrap_or_default(),
                        ),
                        is_error: None,
                    }
                }
                Err(e) => ToolCallResult {
                    content: text_content(format!("Error: {e}")),
                    is_error: Some(true),
                },
            }
        }
        // ---- NEW: Delete note ----------------------------------------------
        "echonote_delete_note" => {
            let Some(nid_str) = args.get("note_id").and_then(|v| v.as_str()) else {
                return ToolCallResult {
                    content: text_content("Missing required parameter: note_id".into()),
                    is_error: Some(true),
                };
            };
            let Ok(note_id) = Uuid::parse_str(nid_str).map(NoteId) else {
                return ToolCallResult {
                    content: text_content(format!("Invalid UUID: {nid_str}")),
                    is_error: Some(true),
                };
            };
            match store.delete_note(note_id).await {
                Ok(true) => ToolCallResult {
                    content: text_content(format!("Note {nid_str} deleted")),
                    is_error: None,
                },
                Ok(false) => ToolCallResult {
                    content: text_content(format!("Note not found: {nid_str}")),
                    is_error: Some(true),
                },
                Err(e) => ToolCallResult {
                    content: text_content(format!("Error: {e}")),
                    is_error: Some(true),
                },
            }
        }
        _ => ToolCallResult {
            content: text_content(format!("Unknown tool: {name}")),
            is_error: Some(true),
        },
    }
}

// ---------------------------------------------------------------------------
// Request dispatch
// ---------------------------------------------------------------------------

async fn handle_request(
    store: &SqliteMeetingStore,
    req: JsonRpcRequest,
) -> Option<JsonRpcResponse> {
    let id = req.id.clone().unwrap_or(Value::Null);

    match req.method.as_str() {
        "initialize" => {
            let result = InitializeResult {
                protocol_version: "2024-11-05".into(),
                capabilities: Capabilities {
                    tools: ToolsCapability {},
                },
                server_info: ServerInfo {
                    name: "echonote".into(),
                    version: env!("CARGO_PKG_VERSION").into(),
                },
            };
            Some(JsonRpcResponse::ok(
                id,
                serde_json::to_value(result).unwrap(),
            ))
        }
        "notifications/initialized" | "initialized" => {
            // Notification — no response
            None
        }
        "tools/list" => {
            let result = ToolsListResult {
                tools: tool_definitions(),
            };
            Some(JsonRpcResponse::ok(
                id,
                serde_json::to_value(result).unwrap(),
            ))
        }
        "tools/call" => {
            let name = req
                .params
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let args = req
                .params
                .get("arguments")
                .cloned()
                .unwrap_or(Value::Object(Default::default()));
            let result = call_tool(store, name, &args).await;
            Some(JsonRpcResponse::ok(
                id,
                serde_json::to_value(result).unwrap(),
            ))
        }
        "ping" => Some(JsonRpcResponse::ok(id, serde_json::json!({}))),
        _ => Some(JsonRpcResponse::err(
            id,
            -32601,
            format!("Method not found: {}", req.method),
        )),
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn parse_args() -> PathBuf {
    let args: Vec<String> = std::env::args().collect();
    for i in 0..args.len() {
        if args[i] == "--db" {
            if let Some(path) = args.get(i + 1) {
                return PathBuf::from(path);
            }
        }
    }
    // Default: platform-standard location
    if let Some(dir) = dirs_next() {
        return dir.join("echonote.db");
    }
    PathBuf::from("echonote.db")
}

/// Resolve the OS-standard app data directory for EchoNote.
fn dirs_next() -> Option<PathBuf> {
    #[cfg(target_os = "macos")]
    {
        dirs::data_dir().map(|d| d.join("app.echonote.desktop"))
    }
    #[cfg(target_os = "windows")]
    {
        dirs::data_dir().map(|d| d.join("app.echonote.desktop"))
    }
    #[cfg(target_os = "linux")]
    {
        dirs::data_dir().map(|d| d.join("app.echonote.desktop"))
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
    {
        None
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_writer(io::stderr)
        .with_env_filter("echo_storage=info,echo_mcp=info")
        .init();

    let db_path = parse_args();
    tracing::info!(db = %db_path.display(), "opening database");

    let store = match SqliteMeetingStore::open(&db_path).await {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to open database at {}: {e}", db_path.display());
            std::process::exit(1);
        }
    };

    let stdin = io::stdin();
    let stdout = io::stdout();
    let reader = stdin.lock();

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let req: JsonRpcRequest = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(e) => {
                let resp = JsonRpcResponse::err(Value::Null, -32700, format!("Parse error: {e}"));
                let out = serde_json::to_string(&resp).unwrap();
                let mut stdout = stdout.lock();
                let _ = writeln!(stdout, "{out}");
                let _ = stdout.flush();
                continue;
            }
        };

        if let Some(resp) = handle_request(&store, req).await {
            let out = serde_json::to_string(&resp).unwrap();
            let mut stdout = stdout.lock();
            let _ = writeln!(stdout, "{out}");
            let _ = stdout.flush();
        }
    }
}
