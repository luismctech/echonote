//! MCP client detection and configuration installer.
//!
//! Provides Tauri commands that:
//! 1. Detect which MCP-capable clients are installed on the user's machine.
//! 2. Install (or remove) the EchoNote MCP server config for each client.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tauri::State;

use super::AppState;
use crate::ipc_error::{ErrorCode, IpcError};

// ---------------------------------------------------------------------------
// Types exposed over IPC
// ---------------------------------------------------------------------------

/// An MCP client that can be configured to connect to EchoNote.
#[derive(Debug, Clone, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct McpClient {
    /// Machine identifier: `claude-desktop`, `vscode`, `cursor`, etc.
    pub id: String,
    /// Human-readable name shown in the UI.
    pub label: String,
    /// Whether the client application appears to be installed.
    pub detected: bool,
    /// Whether the EchoNote MCP entry is already present in the config.
    pub installed: bool,
    /// Path to the configuration file (for transparency).
    pub config_path: Option<String>,
}

/// Result of an install/uninstall operation.
#[derive(Debug, Clone, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct McpInstallResult {
    pub success: bool,
    pub message: String,
}

// ---------------------------------------------------------------------------
// Client registry
// ---------------------------------------------------------------------------

struct ClientSpec {
    id: &'static str,
    label: &'static str,
    /// Returns the config file path. `None` = not applicable on this OS.
    config_path: fn() -> Option<PathBuf>,
    /// JSON key that holds MCP servers in the config file.
    /// Most clients use `"mcpServers"`, VS Code mcp.json uses top-level `"servers"`.
    key_style: KeyStyle,
    /// Returns `true` when the application appears installed.
    detect: fn() -> bool,
}

#[derive(Clone, Copy)]
enum KeyStyle {
    /// `{ "mcpServers": { "echonote": { ... } } }`
    McpServers,
    /// `{ "servers": { "echonote": { ... } } }`  (VS Code mcp.json)
    Servers,
}

fn home() -> Option<PathBuf> {
    dirs::home_dir()
}

fn data_dir() -> Option<PathBuf> {
    dirs::data_dir()
}

// ── Config path resolvers ────────────────────────────────────────────────

fn claude_desktop_config() -> Option<PathBuf> {
    #[cfg(target_os = "macos")]
    {
        data_dir().map(|d| d.join("Claude").join("claude_desktop_config.json"))
    }
    #[cfg(target_os = "windows")]
    {
        std::env::var("APPDATA").ok().map(|d| {
            PathBuf::from(d)
                .join("Claude")
                .join("claude_desktop_config.json")
        })
    }
    #[cfg(target_os = "linux")]
    {
        home().map(|h| {
            h.join(".config")
                .join("Claude")
                .join("claude_desktop_config.json")
        })
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
    {
        None
    }
}

fn claude_code_config() -> Option<PathBuf> {
    home().map(|h| h.join(".claude.json"))
}

fn vscode_config() -> Option<PathBuf> {
    #[cfg(target_os = "macos")]
    {
        data_dir().map(|d| d.join("Code").join("User").join("mcp.json"))
    }
    #[cfg(target_os = "windows")]
    {
        std::env::var("APPDATA")
            .ok()
            .map(|d| PathBuf::from(d).join("Code").join("User").join("mcp.json"))
    }
    #[cfg(target_os = "linux")]
    {
        home().map(|h| h.join(".config").join("Code").join("User").join("mcp.json"))
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
    {
        None
    }
}

fn cursor_config() -> Option<PathBuf> {
    home().map(|h| h.join(".cursor").join("mcp.json"))
}

fn windsurf_config() -> Option<PathBuf> {
    home().map(|h| h.join(".codeium").join("windsurf").join("mcp_config.json"))
}

// ── Detection helpers ────────────────────────────────────────────────────

fn detect_claude_desktop() -> bool {
    #[cfg(target_os = "macos")]
    {
        PathBuf::from("/Applications/Claude.app").exists()
    }
    #[cfg(target_os = "windows")]
    {
        std::env::var("LOCALAPPDATA")
            .map(|d| {
                PathBuf::from(d)
                    .join("Programs")
                    .join("claude")
                    .join("Claude.exe")
                    .exists()
            })
            .unwrap_or(false)
    }
    #[cfg(target_os = "linux")]
    {
        which_exists("claude")
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
    {
        false
    }
}

fn detect_claude_code() -> bool {
    which_exists("claude")
}

fn detect_vscode() -> bool {
    #[cfg(target_os = "macos")]
    {
        PathBuf::from("/Applications/Visual Studio Code.app").exists() || which_exists("code")
    }
    #[cfg(not(target_os = "macos"))]
    {
        which_exists("code")
    }
}

fn detect_cursor() -> bool {
    #[cfg(target_os = "macos")]
    {
        PathBuf::from("/Applications/Cursor.app").exists() || which_exists("cursor")
    }
    #[cfg(not(target_os = "macos"))]
    {
        which_exists("cursor")
    }
}

fn detect_windsurf() -> bool {
    #[cfg(target_os = "macos")]
    {
        PathBuf::from("/Applications/Windsurf.app").exists() || which_exists("windsurf")
    }
    #[cfg(not(target_os = "macos"))]
    {
        which_exists("windsurf")
    }
}

fn which_exists(cmd: &str) -> bool {
    std::process::Command::new("which")
        .arg(cmd)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

static CLIENTS: &[ClientSpec] = &[
    ClientSpec {
        id: "claude-desktop",
        label: "Claude Desktop",
        config_path: claude_desktop_config,
        key_style: KeyStyle::McpServers,
        detect: detect_claude_desktop,
    },
    ClientSpec {
        id: "claude-code",
        label: "Claude Code",
        config_path: claude_code_config,
        key_style: KeyStyle::McpServers,
        detect: detect_claude_code,
    },
    ClientSpec {
        id: "vscode",
        label: "VS Code",
        config_path: vscode_config,
        key_style: KeyStyle::Servers,
        detect: detect_vscode,
    },
    ClientSpec {
        id: "cursor",
        label: "Cursor",
        config_path: cursor_config,
        key_style: KeyStyle::McpServers,
        detect: detect_cursor,
    },
    ClientSpec {
        id: "windsurf",
        label: "Windsurf",
        config_path: windsurf_config,
        key_style: KeyStyle::McpServers,
        detect: detect_windsurf,
    },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolve the absolute path to the echo-mcp binary.
fn mcp_binary_path(_state: &AppState) -> PathBuf {
    // In release builds, the binary is bundled next to the main executable.
    // In debug builds, it's in target/debug/.
    if cfg!(debug_assertions) {
        // Development: point to workspace target
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.pop(); // src-tauri -> workspace root
        p.push("target");
        p.push("debug");
        #[cfg(target_os = "windows")]
        p.push("echo-mcp.exe");
        #[cfg(not(target_os = "windows"))]
        p.push("echo-mcp");
        p
    } else {
        // Production: next to the main binary
        let exe = std::env::current_exe().unwrap_or_default();
        let dir = exe.parent().unwrap_or(std::path::Path::new("."));
        #[cfg(target_os = "windows")]
        {
            dir.join("echo-mcp.exe")
        }
        #[cfg(not(target_os = "windows"))]
        {
            dir.join("echo-mcp")
        }
    }
}

/// Resolve the database path the MCP server should use.
fn mcp_db_path(state: &AppState) -> PathBuf {
    // Reuse the same db_path logic from AppState.
    // The data_root is already resolved; the db is at the standard location.
    if cfg!(debug_assertions) {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.pop();
        p.push("echonote.db");
        p
    } else {
        state.data_root.join("echonote.db")
    }
}

/// Build the server entry JSON for the EchoNote MCP server.
fn server_entry(state: &AppState) -> serde_json::Value {
    let bin = mcp_binary_path(state);
    let db = mcp_db_path(state);
    serde_json::json!({
        "command": bin.to_string_lossy(),
        "args": ["--db", db.to_string_lossy()]
    })
}

/// Check if "echonote" is already configured in a config file.
fn is_installed(config_path: &PathBuf, key_style: KeyStyle) -> bool {
    let Ok(content) = std::fs::read_to_string(config_path) else {
        return false;
    };
    let Ok(json): Result<serde_json::Value, _> = serde_json::from_str(&content) else {
        return false;
    };

    match key_style {
        KeyStyle::McpServers => json
            .get("mcpServers")
            .and_then(|v| v.get("echonote"))
            .is_some(),
        KeyStyle::Servers => {
            // VS Code mcp.json: top-level "servers" key
            json.get("servers")
                .and_then(|v| v.get("echonote"))
                .is_some()
        }
    }
}

// ---------------------------------------------------------------------------
// Tauri commands
// ---------------------------------------------------------------------------

/// Detect all known MCP clients and their installation status.
#[tauri::command]
#[specta::specta]
pub fn detect_mcp_clients(_state: State<'_, AppState>) -> Vec<McpClient> {
    CLIENTS
        .iter()
        .map(|spec| {
            let config_path = (spec.config_path)();
            let detected = (spec.detect)();
            let installed = config_path
                .as_ref()
                .map(|p| is_installed(p, spec.key_style))
                .unwrap_or(false);
            McpClient {
                id: spec.id.to_string(),
                label: spec.label.to_string(),
                detected,
                installed,
                config_path: config_path.map(|p| p.to_string_lossy().to_string()),
            }
        })
        .collect()
}

/// Install the EchoNote MCP server entry into a specific client's config.
#[tauri::command]
#[specta::specta]
pub fn install_mcp_client(
    state: State<'_, AppState>,
    client_id: String,
) -> Result<McpInstallResult, IpcError> {
    let spec = CLIENTS
        .iter()
        .find(|s| s.id == client_id)
        .ok_or_else(|| IpcError {
            code: ErrorCode::InvalidInput,
            message: format!("Unknown client: {client_id}"),
            retriable: false,
        })?;

    let config_path = (spec.config_path)().ok_or_else(|| IpcError {
        code: ErrorCode::Internal,
        message: "Config path not available on this OS".into(),
        retriable: false,
    })?;

    // Ensure parent directory exists
    if let Some(parent) = config_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| IpcError {
            code: ErrorCode::Storage,
            message: format!("Failed to create config dir: {e}"),
            retriable: true,
        })?;
    }

    // Read or create the config file
    let mut json: serde_json::Value = if config_path.exists() {
        let content = std::fs::read_to_string(&config_path).map_err(|e| IpcError {
            code: ErrorCode::Storage,
            message: format!("Failed to read config: {e}"),
            retriable: true,
        })?;
        serde_json::from_str(&content).unwrap_or_else(|_| serde_json::json!({}))
    } else {
        serde_json::json!({})
    };

    let entry = server_entry(&state);

    match spec.key_style {
        KeyStyle::McpServers => {
            let servers = json
                .as_object_mut()
                .unwrap()
                .entry("mcpServers")
                .or_insert_with(|| serde_json::json!({}));
            servers
                .as_object_mut()
                .unwrap()
                .insert("echonote".into(), entry);
        }
        KeyStyle::Servers => {
            // VS Code mcp.json: top-level "servers" with required "type" field
            let servers = json
                .as_object_mut()
                .unwrap()
                .entry("servers")
                .or_insert_with(|| serde_json::json!({}));
            servers.as_object_mut().unwrap().insert(
                "echonote".into(),
                serde_json::json!({
                    "type": "stdio",
                    "command": entry["command"],
                    "args": entry["args"]
                }),
            );
        }
    }

    let formatted = serde_json::to_string_pretty(&json).map_err(|e| IpcError {
        code: ErrorCode::Internal,
        message: format!("Failed to serialize config: {e}"),
        retriable: false,
    })?;

    std::fs::write(&config_path, formatted).map_err(|e| IpcError {
        code: ErrorCode::Storage,
        message: format!("Failed to write config: {e}"),
        retriable: true,
    })?;

    tracing::info!(client = client_id, path = %config_path.display(), "MCP config installed");

    Ok(McpInstallResult {
        success: true,
        message: format!("Installed to {}", config_path.display()),
    })
}

/// Remove the EchoNote MCP server entry from a specific client's config.
#[tauri::command]
#[specta::specta]
pub fn uninstall_mcp_client(
    _state: State<'_, AppState>,
    client_id: String,
) -> Result<McpInstallResult, IpcError> {
    let spec = CLIENTS
        .iter()
        .find(|s| s.id == client_id)
        .ok_or_else(|| IpcError {
            code: ErrorCode::InvalidInput,
            message: format!("Unknown client: {client_id}"),
            retriable: false,
        })?;

    let config_path = (spec.config_path)().ok_or_else(|| IpcError {
        code: ErrorCode::Internal,
        message: "Config path not available on this OS".into(),
        retriable: false,
    })?;

    if !config_path.exists() {
        return Ok(McpInstallResult {
            success: true,
            message: "Config file doesn't exist — nothing to remove.".into(),
        });
    }

    let content = std::fs::read_to_string(&config_path).map_err(|e| IpcError {
        code: ErrorCode::Storage,
        message: format!("Failed to read config: {e}"),
        retriable: true,
    })?;

    let mut json: serde_json::Value =
        serde_json::from_str(&content).unwrap_or_else(|_| serde_json::json!({}));

    match spec.key_style {
        KeyStyle::McpServers => {
            if let Some(servers) = json.get_mut("mcpServers").and_then(|v| v.as_object_mut()) {
                servers.remove("echonote");
            }
        }
        KeyStyle::Servers => {
            // VS Code mcp.json: top-level "servers"
            if let Some(servers) = json.get_mut("servers").and_then(|v| v.as_object_mut()) {
                servers.remove("echonote");
            }
        }
    }

    let formatted = serde_json::to_string_pretty(&json).map_err(|e| IpcError {
        code: ErrorCode::Internal,
        message: format!("Failed to serialize config: {e}"),
        retriable: false,
    })?;

    std::fs::write(&config_path, formatted).map_err(|e| IpcError {
        code: ErrorCode::Storage,
        message: format!("Failed to write config: {e}"),
        retriable: true,
    })?;

    tracing::info!(client = client_id, path = %config_path.display(), "MCP config removed");

    Ok(McpInstallResult {
        success: true,
        message: "Removed EchoNote MCP entry.".into(),
    })
}

/// Generate the MCP config JSON snippet for manual copy-paste.
#[tauri::command]
#[specta::specta]
pub fn get_mcp_config_snippet(state: State<'_, AppState>) -> String {
    let entry = server_entry(&state);
    let snippet = serde_json::json!({
        "echonote": entry
    });
    serde_json::to_string_pretty(&snippet).unwrap_or_default()
}
