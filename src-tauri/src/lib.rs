//! EchoNote Tauri host shell.
//!
//! This crate is intentionally thin: it wires the application layer
//! (`echo-app`) and the selected infrastructure adapters (`echo-audio`,
//! `echo-asr`, `echo-storage`, ...) into a Tauri application. Domain
//! rules live in `echo-domain` and must not be imported from command
//! handlers.

mod commands;
mod ipc_error;

use tauri::menu::{MenuBuilder, MenuItemBuilder};
use tauri::tray::TrayIconEvent;
use tauri::Manager;

/// Entry point invoked by `main.rs`. Kept library-friendly so mobile
/// targets (Tauri 2 iOS/Android) can reuse the same builder.
pub fn run() {
    echo_telemetry::init();

    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        commit = env!("ECHO_GIT_HASH"),
        "echo-shell starting"
    );

    // Build the tauri-specta binding layer. This collects all
    // `#[specta::specta]`-annotated commands and their types so they
    // can be (a) fed to Tauri's invoke handler and (b) exported to
    // TypeScript at dev time.
    let specta_builder =
        tauri_specta::Builder::<tauri::Wry>::new().commands(tauri_specta::collect_commands![
            commands::health_check,
            commands::streaming::start_streaming,
            commands::streaming::stop_streaming,
            commands::streaming::pause_streaming,
            commands::streaming::resume_streaming,
            commands::streaming::get_meeting_id,
            commands::meetings::list_meetings,
            commands::meetings::get_meeting,
            commands::meetings::delete_meeting,
            commands::meetings::rename_meeting,
            commands::meetings::rename_speaker,
            commands::meetings::search_meetings,
            commands::meetings::add_note,
            commands::meetings::list_notes,
            commands::meetings::delete_note,
            commands::llm::summarize_meeting,
            commands::llm::summarize_meeting_stream,
            commands::llm::get_summary,
            commands::llm::ask_about_meeting,
            commands::export::export_meeting,
            commands::models::get_model_status,
            commands::models::download_model,
            commands::models::unload_model,
            commands::models::cancel_download,
            commands::models::delete_model,
            commands::models::set_active_llm,
            commands::models::get_active_llm,
            commands::models::set_active_asr,
            commands::models::get_active_asr,
            commands::models::set_active_embedder,
            commands::models::get_active_embedder,
            commands::templates::list_custom_templates,
            commands::templates::create_custom_template,
            commands::templates::update_custom_template,
            commands::templates::delete_custom_template,
            commands::llm::summarize_with_custom_template,
            commands::hardware::get_hardware_profile,
            commands::recommendation::get_model_recommendation,
            commands::mcp::detect_mcp_clients,
            commands::mcp::install_mcp_client,
            commands::mcp::uninstall_mcp_client,
            commands::mcp::get_mcp_config_snippet,
        ]);

    // In dev builds, export TypeScript bindings so the frontend can
    // import fully typed command wrappers and domain types from a
    // single generated file. The file is written at app startup, NOT
    // on every hot-reload, because we gate on `debug_assertions`.
    #[cfg(debug_assertions)]
    {
        let bindings_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../src/ipc/bindings.ts");
        specta_builder
            .export(
                specta_typescript::Typescript::default()
                    .bigint(specta_typescript::BigIntExportBehavior::Number),
                &bindings_path,
            )
            .expect("failed to export tauri-specta TypeScript bindings");
    }

    // NOTE: `tauri-plugin-log` conflicts with `echo_telemetry::init`,
    // which already installs a global `log → tracing` bridge. Logging
    // is delivered through that subscriber. Reintroduce the plugin only
    // if we move telemetry off `tracing-subscriber`.
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            // Another instance tried to launch — bring the existing
            // window to the front instead of spawning a duplicate.
            if let Some(w) = app.get_webview_window("main") {
                let _ = w.show();
                let _ = w.unminimize();
                let _ = w.set_focus();
            }
        }))
        .setup(|app| {
            // ── System tray menu ─────────────────────────────────
            let show = MenuItemBuilder::with_id("show", "Show EchoNote").build(app)?;
            let quit = MenuItemBuilder::with_id("quit", "Quit").build(app)?;
            let menu = MenuBuilder::new(app).items(&[&show, &quit]).build()?;

            if let Some(tray) = app.tray_by_id("main-tray") {
                tray.set_menu(Some(menu))?;
                tray.on_menu_event(move |app, event| match event.id().as_ref() {
                    "show" => {
                        if let Some(w) = app.get_webview_window("main") {
                            let _ = w.show();
                            let _ = w.unminimize();
                            let _ = w.set_focus();
                        }
                    }
                    "quit" => {
                        app.exit(0);
                    }
                    _ => {}
                });
                tray.on_tray_icon_event(|tray, event| {
                    // Only react to left-click. On Windows the OS opens
                    // the context menu on right-click; intercepting all
                    // clicks steals focus and prevents the menu from
                    // appearing.
                    if let TrayIconEvent::Click {
                        button: tauri::tray::MouseButton::Left,
                        button_state: tauri::tray::MouseButtonState::Up,
                        ..
                    } = event
                    {
                        if let Some(w) = tray.app_handle().get_webview_window("main") {
                            let _ = w.show();
                            let _ = w.unminimize();
                            let _ = w.set_focus();
                        }
                    }
                });
            }

            // ── App state (SQLite + adapters) ────────────────────
            let handle = app.handle().clone();

            // Resolve the OS-appropriate app-data directory and ensure
            // it exists before the store tries to open the database.
            let app_data_dir = handle.path().app_data_dir().ok();
            if let Some(ref dir) = app_data_dir {
                std::fs::create_dir_all(dir).expect("failed to create app data directory");
            }

            let state =
                tauri::async_runtime::block_on(commands::AppState::initialize(app_data_dir))
                    .expect("failed to initialize echo-shell state");
            handle.manage(state);
            Ok(())
        })
        .invoke_handler(specta_builder.invoke_handler())
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|app_handle, event| {
        match event {
            tauri::RunEvent::WindowEvent {
                label,
                event: tauri::WindowEvent::CloseRequested { api, .. },
                ..
            } if label == "main" => {
                // Hide to tray instead of quitting — the user can
                // restore the window from the tray icon or menu.
                api.prevent_close();
                if let Some(w) = app_handle.get_webview_window("main") {
                    let _ = w.hide();
                }
            }
            tauri::RunEvent::Exit => {
                // Ordered shutdown: drain streaming sessions, close SQLite.
                let state = app_handle.state::<commands::AppState>();
                tauri::async_runtime::block_on(state.shutdown());

                // Workaround: ggml/whisper.cpp registers a C++ atexit
                // handler that frees the global Metal device. On macOS
                // that destructor calls `ggml_abort` because a
                // background GCD block is still alive at teardown,
                // producing SIGABRT. `_exit` skips C++ destructors.
                tracing::info!("echo-shell shutting down (bypassing atexit)");
                #[cfg(unix)]
                unsafe {
                    libc::_exit(0);
                }
                #[cfg(not(unix))]
                std::process::exit(0);
            }
            _ => {}
        }
    });
}
