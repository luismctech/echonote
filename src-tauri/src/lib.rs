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

    // NOTE: `tauri-plugin-log` conflicts with `echo_telemetry::init`,
    // which already installs a global `log → tracing` bridge. Logging
    // is delivered through that subscriber. Reintroduce the plugin only
    // if we move telemetry off `tracing-subscriber`.
    tauri::Builder::default()
        .manage(commands::AppState::new())
        .invoke_handler(generate_handler![
            commands::health_check,
            commands::start_streaming,
            commands::stop_streaming,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
