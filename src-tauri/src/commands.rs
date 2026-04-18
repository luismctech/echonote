//! Tauri IPC commands exposed to the frontend.
//!
//! Each command here mirrors a typed contract in
//! `src/lib/ipc.ts`. When the surface grows beyond a handful, switch to
//! `tauri-specta` code generation — see ADR note in
//! `docs/adr/0002-rust-plus-react-stack.md`.

use serde::Serialize;
use time::{format_description::well_known::Rfc3339, OffsetDateTime};

/// Result returned by [`health_check`]. Mirrors `HealthStatus` on the TS side.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HealthStatus {
    /// RFC 3339 timestamp of the probe.
    pub timestamp: String,
    /// Backend semver, from Cargo at compile time.
    pub version: String,
    /// Target triple the backend was compiled for.
    pub target: String,
    /// Short git hash, `unknown` when `.git` is missing at build time.
    pub commit: String,
}

/// Lightweight probe the frontend calls on mount to confirm the bridge is live.
#[tauri::command]
pub fn health_check() -> HealthStatus {
    let timestamp = OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_else(|_| "unknown".to_string());

    HealthStatus {
        timestamp,
        version: env!("CARGO_PKG_VERSION").to_string(),
        target: env!("TAURI_ENV_TARGET_TRIPLE").to_string(),
        commit: env!("ECHO_GIT_HASH").to_string(),
    }
}
