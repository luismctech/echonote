//! Model catalog, status, and download commands.

use serde::Serialize;
use tauri::ipc::Channel;
use tauri::State;

use crate::ipc_error::{ErrorCode, IpcError};

use futures::stream::StreamExt;

use super::AppState;

/// A downloadable model known to the app.
#[derive(Debug, Clone, Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    /// Machine-readable id (e.g. `"asr-large-v3-turbo"`).
    pub id: String,
    /// Human label shown in the UI.
    pub label: String,
    /// Category: `"asr"`, `"llm"`, `"vad"`, or `"embedder"`.
    pub kind: String,
    /// Whether the file exists on disk right now.
    pub present: bool,
    /// Approximate download size in bytes (for progress UI).
    pub size_bytes: u64,
}

/// Catalog of models the app can download, with their HF URLs and
/// expected sizes. Only includes models the priority resolvers know.
///
/// Each entry carries an optional SHA-256 hex digest. When present,
/// downloads are verified after completion and rejected on mismatch.
/// Populate hashes by running `shasum -a 256 <model_file>`.
fn model_catalog(root: &std::path::Path) -> Vec<(ModelInfo, &'static str, Option<&'static str>)> {
    let present = |rel: &str| root.join(rel).exists();

    vec![
        (
            ModelInfo {
                id: "asr-large-v3-turbo".into(),
                label: "Whisper Large V3 Turbo (multilingual, 1.5 GB)".into(),
                kind: "asr".into(),
                present: present("models/asr/ggml-large-v3-turbo.bin"),
                size_bytes: 1_600_000_000,
            },
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
            None, // TODO: populate sha256
        ),
        (
            ModelInfo {
                id: "asr-large-v3-turbo-q5".into(),
                label: "Whisper Large V3 Turbo Q5 (multilingual, 574 MB)".into(),
                kind: "asr".into(),
                present: present("models/asr/ggml-large-v3-turbo-q5_0.bin"),
                size_bytes: 574_000_000,
            },
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin",
            None, // TODO: populate sha256
        ),
        (
            ModelInfo {
                id: "asr-distil-large-v3".into(),
                label: "Distil-Whisper Large V3 (English, 756 MB, 5x faster)".into(),
                kind: "asr".into(),
                present: present("models/asr/ggml-distil-large-v3.bin"),
                size_bytes: 756_000_000,
            },
            "https://huggingface.co/distil-whisper/distil-large-v3-ggml/resolve/main/ggml-distil-large-v3.bin",
            None, // TODO: populate sha256
        ),
        (
            ModelInfo {
                id: "asr-medium".into(),
                label: "Whisper Medium (multilingual, 1.5 GB)".into(),
                kind: "asr".into(),
                present: present("models/asr/ggml-medium.bin"),
                size_bytes: 1_530_000_000,
            },
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
            None, // TODO: populate sha256
        ),
        (
            ModelInfo {
                id: "asr-small".into(),
                label: "Whisper Small (multilingual, 488 MB)".into(),
                kind: "asr".into(),
                present: present("models/asr/ggml-small.bin"),
                size_bytes: 488_000_000,
            },
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
            None, // TODO: populate sha256
        ),
        (
            ModelInfo {
                id: "asr-base".into(),
                label: "Whisper Base (multilingual, 142 MB)".into(),
                kind: "asr".into(),
                present: present("models/asr/ggml-base.bin"),
                size_bytes: 148_000_000,
            },
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
            None, // TODO: populate sha256
        ),
        (
            ModelInfo {
                id: "llm-qwen3-14b".into(),
                label: "Qwen 3 14B Q4_K_M (9 GB)".into(),
                kind: "llm".into(),
                present: present("models/llm/Qwen3-14B-Q4_K_M.gguf"),
                size_bytes: 9_200_000_000,
            },
            "https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q4_K_M.gguf",
            None, // TODO: populate sha256
        ),
        (
            ModelInfo {
                id: "llm-qwen3-8b".into(),
                label: "Qwen 3 8B Q4_K_M (5 GB)".into(),
                kind: "llm".into(),
                present: present("models/llm/Qwen3-8B-Q4_K_M.gguf"),
                size_bytes: 5_200_000_000,
            },
            "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
            None, // TODO: populate sha256
        ),
        (
            ModelInfo {
                id: "llm-qwen3-4b".into(),
                label: "Qwen 3 4B Q4_K_M (2.5 GB) — for <8 GB RAM".into(),
                kind: "llm".into(),
                present: present("models/llm/Qwen3-4B-Q4_K_M.gguf"),
                size_bytes: 2_600_000_000,
            },
            "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf",
            None, // TODO: populate sha256
        ),
        (
            ModelInfo {
                id: "vad-silero".into(),
                label: "Silero VAD v5 (1.2 MB, simplified for tract)".into(),
                kind: "vad".into(),
                present: present("models/vad/silero_vad.onnx"),
                size_bytes: 1_200_000,
            },
            // Pre-simplified ONNX (If nodes inlined for 16 kHz, BASIC-optimised).
            // The upstream file has ONNX `If` ops that tract cannot execute.
            "https://github.com/luismctech/echonote/releases/download/v0.2.1/silero_vad.onnx",
            Some("d224cf508fbaf8bb1a49f333120b536dbaa1ed2b0ab49bed059d6e44a4f8305c"),
        ),
        (
            ModelInfo {
                id: "embedder-eres2net".into(),
                label: "ERes2Net Speaker Embedder (26 MB)".into(),
                kind: "embedder".into(),
                present: present("models/embedder/eres2net_en_voxceleb.onnx"),
                size_bytes: 26_000_000,
            },
            "https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main/3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx",
            None,
        ),
        (
            ModelInfo {
                id: "embedder-camplusplus".into(),
                label: "CAM++ Speaker Embedder — recommended for Spanish (28 MB)".into(),
                kind: "embedder".into(),
                present: present("models/embedder/campplus_en_voxceleb.onnx"),
                size_bytes: 28_000_000,
            },
            "https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
            None,
        ),
        (
            ModelInfo {
                id: "segmenter-pyannote".into(),
                label: "pyannote Segmentation 3.0 — speaker boundary detection (17 MB)".into(),
                kind: "segmenter".into(),
                present: present("models/segmenter/pyannote_segmentation_3.onnx"),
                size_bytes: 17_000_000,
            },
            "https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/main/model.onnx",
            None,
        ),
    ]
}

/// Map a catalog model id to its relative on-disk path.
fn model_dest_path(id: &str) -> Option<&'static str> {
    match id {
        "asr-large-v3-turbo" => Some("models/asr/ggml-large-v3-turbo.bin"),
        "asr-large-v3-turbo-q5" => Some("models/asr/ggml-large-v3-turbo-q5_0.bin"),
        "asr-distil-large-v3" => Some("models/asr/ggml-distil-large-v3.bin"),
        "asr-medium" => Some("models/asr/ggml-medium.bin"),
        "asr-small" => Some("models/asr/ggml-small.bin"),
        "asr-base" => Some("models/asr/ggml-base.bin"),
        "llm-qwen3-14b" => Some("models/llm/Qwen3-14B-Q4_K_M.gguf"),
        "llm-qwen3-8b" => Some("models/llm/Qwen3-8B-Q4_K_M.gguf"),
        "llm-qwen3-4b" => Some("models/llm/Qwen3-4B-Q4_K_M.gguf"),
        "vad-silero" => Some("models/vad/silero_vad.onnx"),
        "embedder-eres2net" => Some("models/embedder/eres2net_en_voxceleb.onnx"),
        "embedder-camplusplus" => Some("models/embedder/campplus_en_voxceleb.onnx"),
        "segmenter-pyannote" => Some("models/segmenter/pyannote_segmentation_3.onnx"),
        _ => None,
    }
}

/// Return the status of all downloadable models.
#[tauri::command]
#[specta::specta]
pub fn get_model_status(state: State<'_, AppState>) -> Vec<ModelInfo> {
    model_catalog(&state.data_root)
        .into_iter()
        .map(|(info, _, _)| info)
        .collect()
}

/// Progress events streamed to the frontend during a download.
#[derive(Debug, Clone, Serialize, specta::Type)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum DownloadEvent {
    /// Periodic progress update.
    #[serde(rename = "progress")]
    Progress { downloaded: u64, total: u64 },
    /// Download finished successfully.
    #[serde(rename = "finished")]
    Finished,
    /// Download failed.
    #[serde(rename = "failed")]
    Failed { error: String },
    /// Download was cancelled by the user.
    #[serde(rename = "cancelled")]
    Cancelled,
}

/// Download a model by id, streaming progress events to the frontend.
///
/// When the catalog entry includes a SHA-256 digest the downloaded file
/// is verified before being promoted from `*.part` to its final path.
/// Concurrent downloads of the same model are rejected.
#[tauri::command]
#[specta::specta]
pub async fn download_model(
    state: State<'_, AppState>,
    model_id: String,
    on_event: Channel<DownloadEvent>,
) -> Result<(), IpcError> {
    // ── Guard: reject concurrent downloads of the same model ─────
    let cancel_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    {
        let mut guard = state
            .in_flight_downloads
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        if guard.contains_key(&model_id) {
            return Err(IpcError::new(
                ErrorCode::SessionConflict,
                format!("model {model_id} is already being downloaded"),
            ));
        }
        guard.insert(model_id.clone(), cancel_flag.clone());
    }
    let downloads_handle = state.in_flight_downloads.clone();
    let mid = model_id.clone();
    // Ensure the guard is removed on all exit paths.
    let _cleanup = scopeguard::guard((), move |()| {
        if let Ok(mut map) = downloads_handle.lock() {
            map.remove(&mid);
        }
    });

    let catalog = model_catalog(&state.data_root);
    let (_, url, expected_sha) = catalog
        .iter()
        .find(|(info, _, _)| info.id == model_id)
        .ok_or_else(|| IpcError::not_found(format!("unknown model: {model_id}")))?;
    let url = url.to_string();
    let expected_sha = expected_sha.map(|s| s.to_string());

    let rel_path = model_dest_path(&model_id)
        .ok_or_else(|| IpcError::not_found(format!("no dest path for model: {model_id}")))?;
    let dest = state.data_root.join(rel_path);

    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| IpcError::storage(format!("failed to create directory: {e}")))?;
    }

    let tmp = dest.with_extension("part");

    let result: Result<(), IpcError> = async {
        let client = reqwest::Client::new();
        let response = client
            .get(&url)
            .send()
            .await
            .map_err(|e| IpcError::network(format!("HTTP request failed: {e}")))?;

        if !response.status().is_success() {
            return Err(IpcError::network(format!("HTTP {}", response.status())));
        }

        let total = response.content_length().unwrap_or(0);
        let mut downloaded: u64 = 0;
        let mut stream = response.bytes_stream();
        let mut file = tokio::fs::File::create(&tmp)
            .await
            .map_err(|e| IpcError::storage(format!("failed to create file: {e}")))?;

        use sha2::{Digest, Sha256};
        use tokio::io::AsyncWriteExt;

        let mut hasher = Sha256::new();
        let mut last_report = std::time::Instant::now();
        while let Some(chunk) = stream.next().await {
            // ── Check cancel flag ────────────────────────────────
            if cancel_flag.load(std::sync::atomic::Ordering::Relaxed) {
                drop(file);
                let _ = tokio::fs::remove_file(&tmp).await;
                let _ = on_event.send(DownloadEvent::Cancelled);
                return Ok(());
            }

            let chunk =
                chunk.map_err(|e| IpcError::network(format!("download stream error: {e}")))?;
            file.write_all(&chunk)
                .await
                .map_err(|e| IpcError::storage(format!("write error: {e}")))?;
            hasher.update(&chunk);
            downloaded += chunk.len() as u64;

            if last_report.elapsed().as_millis() >= 250 || downloaded == total {
                let _ = on_event.send(DownloadEvent::Progress { downloaded, total });
                last_report = std::time::Instant::now();
            }
        }
        file.flush()
            .await
            .map_err(|e| IpcError::storage(format!("flush error: {e}")))?;
        drop(file);

        // ── SHA-256 verification (when hash is known) ────────────
        if let Some(expected) = &expected_sha {
            let hash = hasher.finalize();
            let actual = hash.iter().map(|b| format!("{b:02x}")).collect::<String>();
            if actual != *expected {
                let _ = tokio::fs::remove_file(&tmp).await;
                return Err(IpcError::new(
                    ErrorCode::InvalidInput,
                    format!("SHA-256 mismatch for {model_id}: expected {expected}, got {actual}"),
                ));
            }
        }

        tokio::fs::rename(&tmp, &dest)
            .await
            .map_err(|e| IpcError::storage(format!("rename failed: {e}")))?;

        Ok(())
    }
    .await;

    match result {
        Ok(()) => {
            let _ = on_event.send(DownloadEvent::Finished);
            Ok(())
        }
        Err(e) => {
            let _ = tokio::fs::remove_file(&tmp).await;
            let _ = on_event.send(DownloadEvent::Failed {
                error: e.message.clone(),
            });
            Err(e)
        }
    }
}

/// Unload a lazily-loaded model to free memory. Accepted kinds:
/// `"asr"`, `"llm"`, `"vad"`. Returns `true` if a model was actually
/// unloaded, `false` if it wasn't loaded.
#[tauri::command]
#[specta::specta]
pub async fn unload_model(state: State<'_, AppState>, kind: String) -> Result<bool, IpcError> {
    match kind.as_str() {
        "asr" => Ok(state.transcriber.unload().await),
        "llm" => Ok(state.llm.unload().await),
        "vad" => Ok(state.vad.unload().await),
        _ => Err(IpcError::not_found(format!("unknown model kind: {kind}"))),
    }
}

/// Cancel an in-flight model download.
///
/// Sets the cancel flag for the download loop, which will clean up the
/// `.part` file and send a `Cancelled` event.
#[tauri::command]
#[specta::specta]
pub async fn cancel_download(
    state: State<'_, AppState>,
    model_id: String,
) -> Result<bool, IpcError> {
    let guard = state
        .in_flight_downloads
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    if let Some(flag) = guard.get(&model_id) {
        flag.store(true, std::sync::atomic::Ordering::Relaxed);
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Delete a downloaded model from disk.
///
/// If the model is currently loaded in memory, it is unloaded first.
#[tauri::command]
#[specta::specta]
pub async fn delete_model(state: State<'_, AppState>, model_id: String) -> Result<(), IpcError> {
    let rel_path = model_dest_path(&model_id)
        .ok_or_else(|| IpcError::not_found(format!("unknown model: {model_id}")))?;
    let dest = state.data_root.join(rel_path);

    if !dest.exists() {
        return Err(IpcError::not_found(format!(
            "model file not found: {model_id}"
        )));
    }

    // Unload the model from memory if it's currently loaded.
    let kind = model_id.split('-').next().unwrap_or("");
    match kind {
        "asr" => {
            state.transcriber.unload().await;
        }
        "llm" => {
            state.llm.unload().await;
        }
        "vad" => {
            state.vad.unload().await;
        }
        _ => {}
    }

    tokio::fs::remove_file(&dest)
        .await
        .map_err(|e| IpcError::storage(format!("failed to delete model: {e}")))?;

    Ok(())
}

/// Set the active LLM model. Unloads the currently loaded model (if
/// any) and switches the path so the next `ensure_llm()` call loads
/// the selected model.
///
/// `model_id` must be an `"llm-*"` id from the catalog whose model
/// file is already downloaded.
#[tauri::command]
#[specta::specta]
pub async fn set_active_llm(state: State<'_, AppState>, model_id: String) -> Result<(), IpcError> {
    let rel_path = model_dest_path(&model_id)
        .ok_or_else(|| IpcError::not_found(format!("unknown model: {model_id}")))?;

    if !model_id.starts_with("llm-") {
        return Err(IpcError::new(
            ErrorCode::InvalidInput,
            format!("{model_id} is not an LLM model"),
        ));
    }

    let dest = state.data_root.join(rel_path);
    if !dest.exists() {
        return Err(IpcError::not_found(format!(
            "model not downloaded: {model_id}"
        )));
    }

    // Unload the currently loaded LLM (if any).
    state.llm.unload().await;

    // SAFETY: We hold an exclusive reference via `State` in the Tauri
    // runtime; this is the only writer. The `unsafe` block is required
    // because `llm_model_path` is not behind an async mutex (to keep
    // it cheap to read). The Tauri command pipeline guarantees single-
    // writer semantics per command invocation.
    //
    // We use an interior-mutability pattern via a small helper below
    // to update the path.
    state.set_llm_model_path(dest);

    tracing::info!(model_id = %model_id, "active LLM switched");
    Ok(())
}

/// Return the model id of the currently configured LLM, or `null`
/// when no LLM model file exists on disk.
#[tauri::command]
#[specta::specta]
pub fn get_active_llm(state: State<'_, AppState>) -> Option<String> {
    let path = state.active_llm_path();
    if !path.exists() {
        return None;
    }
    // Reverse-lookup the catalog id from the filename.
    let catalog = model_catalog(&state.data_root);
    for (info, _, _) in &catalog {
        if info.kind == "llm" {
            let rel = model_dest_path(&info.id).unwrap_or("");
            if state.data_root.join(rel) == path {
                return Some(info.id.clone());
            }
        }
    }
    None
}

/// Set the active speaker embedder model. The next diarization session
/// will load the chosen model (ERes2Net or CAM++).
///
/// `model_id` must be an `"embedder-*"` id from the catalog whose
/// model file is already downloaded.
#[tauri::command]
#[specta::specta]
pub async fn set_active_embedder(
    state: State<'_, AppState>,
    model_id: String,
) -> Result<(), IpcError> {
    let rel_path = model_dest_path(&model_id)
        .ok_or_else(|| IpcError::not_found(format!("unknown model: {model_id}")))?;

    if !model_id.starts_with("embedder-") {
        return Err(IpcError::new(
            ErrorCode::InvalidInput,
            format!("{model_id} is not an embedder model"),
        ));
    }

    let dest = state.data_root.join(rel_path);
    if !dest.exists() {
        return Err(IpcError::not_found(format!(
            "model not downloaded: {model_id}"
        )));
    }

    state.set_embed_model_path(dest);
    tracing::info!(model_id = %model_id, "active embedder switched");
    Ok(())
}

/// Return the model id of the currently configured embedder, or `null`
/// when no embedder model file exists on disk.
#[tauri::command]
#[specta::specta]
pub fn get_active_embedder(state: State<'_, AppState>) -> Option<String> {
    let path = state.active_embed_path();
    if !path.exists() {
        return None;
    }
    let catalog = model_catalog(&state.data_root);
    for (info, _, _) in &catalog {
        if info.kind == "embedder" {
            let rel = model_dest_path(&info.id).unwrap_or("");
            if state.data_root.join(rel) == path {
                return Some(info.id.clone());
            }
        }
    }
    None
}

/// Set the active ASR (speech recognition) model. Unloads the currently
/// loaded transcriber (if any) and switches the path so the next
/// `ensure_transcriber()` call loads the selected model.
///
/// `model_id` must be an `"asr-*"` id from the catalog whose model
/// file is already downloaded.
#[tauri::command]
#[specta::specta]
pub async fn set_active_asr(state: State<'_, AppState>, model_id: String) -> Result<(), IpcError> {
    let rel_path = model_dest_path(&model_id)
        .ok_or_else(|| IpcError::not_found(format!("unknown model: {model_id}")))?;

    if !model_id.starts_with("asr-") {
        return Err(IpcError::new(
            ErrorCode::InvalidInput,
            format!("{model_id} is not an ASR model"),
        ));
    }

    let dest = state.data_root.join(rel_path);
    if !dest.exists() {
        return Err(IpcError::not_found(format!(
            "model not downloaded: {model_id}"
        )));
    }

    // Unload the currently loaded transcriber (if any).
    state.transcriber.unload().await;

    state.set_asr_model_path(dest);

    tracing::info!(model_id = %model_id, "active ASR switched");
    Ok(())
}

/// Return the model id of the currently configured ASR model, or `null`
/// when no ASR model file exists on disk.
#[tauri::command]
#[specta::specta]
pub fn get_active_asr(state: State<'_, AppState>) -> Option<String> {
    let path = state.active_asr_path();
    if !path.exists() {
        return None;
    }
    // Reverse-lookup the catalog id from the filename.
    let catalog = model_catalog(&state.data_root);
    for (info, _, _) in &catalog {
        if info.kind == "asr" {
            let rel = model_dest_path(&info.id).unwrap_or("");
            if state.data_root.join(rel) == path {
                return Some(info.id.clone());
            }
        }
    }
    None
}
