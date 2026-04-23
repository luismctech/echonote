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
    let app = tauri::Builder::default()
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
            commands::search_meetings,
            commands::summarize_meeting,
            commands::get_summary,
            commands::ask_about_meeting,
            commands::get_model_status,
            commands::download_model,
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|app_handle, event| {
        if matches!(event, tauri::RunEvent::Exit) {
            // Run our ordered shutdown *before* the ggml workaround
            // below: drain in-flight streaming sessions (so the final
            // Stopped/Failed events get persisted) and close the
            // SQLite pool (so WAL frames are checkpointed). We block
            // on the existing async runtime — Tauri keeps it alive
            // through the Exit event.
            let state = app_handle.state::<commands::AppState>();
            tauri::async_runtime::block_on(state.shutdown());

            // Workaround: ggml/whisper.cpp registers a C++ atexit
            // handler that frees the global Metal device. On macOS that
            // destructor (`ggml_metal_rsets_free`) calls `ggml_abort`
            // because a background GCD block (`__ggml_metal_rsets_init`)
            // is still alive when the process tears down, producing a
            // SIGABRT every time the user closes the window.
            //
            // We sidestep it by jumping straight to `_exit`, which
            // returns to the kernel without running C/C++ static
            // destructors. We do this *after* `state.shutdown()` so
            // every Rust resource we can flush in-process is flushed
            // first; only the C++ globals are skipped.
            tracing::info!("echo-shell shutting down (bypassing atexit)");
            #[cfg(unix)]
            unsafe {
                libc::_exit(0);
            }
            #[cfg(not(unix))]
            std::process::exit(0);
        }
    });
}
