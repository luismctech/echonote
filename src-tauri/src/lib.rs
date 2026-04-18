//! EchoNote Tauri host shell.
//!
//! This crate is intentionally thin: it wires the application layer
//! (`echo-app`) and the selected infrastructure adapters (`echo-audio`,
//! `echo-asr`, ...) into a Tauri application. Domain rules live in
//! `echo-domain` and must not be imported from command handlers.

mod commands;

use tauri::generate_handler;

/// Entry point invoked by `main.rs`. Kept library-friendly so mobile
/// targets (Tauri 2 iOS/Android) can reuse the same builder.
pub fn run() {
    echo_telemetry::init();

    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        commit = env!("ECHO_GIT_HASH"),
        "echo-shell starting"
    );

    tauri::Builder::default()
        .plugin(tauri_plugin_log::Builder::new().build())
        .invoke_handler(generate_handler![commands::health_check])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
