//! Model catalog, status, and download commands.

use serde::Serialize;
use tauri::ipc::Channel;

use crate::ipc_error::{ErrorCode, IpcError};

use futures::stream::StreamExt;

use super::workspace_root;

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
fn model_catalog() -> Vec<(ModelInfo, &'static str, Option<&'static str>)> {
    let root = workspace_root();
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
                id: "vad-silero".into(),
                label: "Silero VAD v5 (2 MB)".into(),
                kind: "vad".into(),
                present: present("models/vad/silero_vad.onnx"),
                size_bytes: 2_200_000,
            },
            "https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx",
            None, // TODO: populate sha256
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
        "vad-silero" => Some("models/vad/silero_vad.onnx"),
        _ => None,
    }
}

/// Return the status of all downloadable models.
#[tauri::command]
#[specta::specta]
pub fn get_model_status() -> Vec<ModelInfo> {
    model_catalog()
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
}

/// Download a model by id, streaming progress events to the frontend.
///
/// When the catalog entry includes a SHA-256 digest the downloaded file
/// is verified before being promoted from `*.part` to its final path.
#[tauri::command]
#[specta::specta]
pub async fn download_model(
    model_id: String,
    on_event: Channel<DownloadEvent>,
) -> Result<(), IpcError> {
    let catalog = model_catalog();
    let (_, url, expected_sha) = catalog
        .iter()
        .find(|(info, _, _)| info.id == model_id)
        .ok_or_else(|| IpcError::not_found(format!("unknown model: {model_id}")))?;
    let url = url.to_string();
    let expected_sha = expected_sha.map(|s| s.to_string());

    let rel_path = model_dest_path(&model_id)
        .ok_or_else(|| IpcError::not_found(format!("no dest path for model: {model_id}")))?;
    let dest = workspace_root().join(rel_path);

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
            let actual = format!("{:x}", hasher.finalize());
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
