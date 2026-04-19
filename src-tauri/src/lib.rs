//! EchoNote Tauri host shell.
//!
//! This crate is intentionally thin: it wires the application layer
//! (`echo-app`) and the selected infrastructure adapters (`echo-audio`,
//! `echo-asr`, `echo-storage`, ...) into a Tauri application. Domain
//! rules live in `echo-domain` and must not be imported from command
//! handlers.

mod commands;

use tauri::{generate_handler, Manager};

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
        .setup(|app| {
            // `AppState::initialize` is async (opens SQLite and runs
            // migrations). Tauri's setup hook is sync but we have a
            // tokio runtime available via the handle, so we block_on
            // long enough for the DB to come up. This is fine — the
            // window has not been shown yet.
            let handle = app.handle().clone();
            let state = tauri::async_runtime::block_on(commands::AppState::initialize())
                .expect("failed to initialize echo-shell state");
            handle.manage(state);
            Ok(())
        })
        .invoke_handler(generate_handler![
            commands::health_check,
            commands::start_streaming,
            commands::stop_streaming,
            commands::list_meetings,
            commands::get_meeting,
            commands::delete_meeting,
            commands::rename_speaker,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
