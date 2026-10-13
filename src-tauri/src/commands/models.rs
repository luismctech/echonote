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
    /// Short, user-friendly name shown as the primary label.
    pub label: String,
    /// One-line description with key traits (language, speed, etc.).
    pub description: String,
    /// Category: `"asr"`, `"llm"`, `"vad"`, `"embedder"`, or `"segmenter"`.
    pub kind: String,
    /// Whether the file exists on disk right now.
    pub present: bool,
    /// Approximate download size in bytes (for progress UI).
    pub size_bytes: u64,
}

/// SHA-256 of the Silero VAD v6.2.3 build the adapter expects
/// (`scripts/inline-silero-vad.py` over upstream commit `5cd79456`).
pub(crate) const SILERO_VAD_V6_2_SHA256: &str =
    "76d1667a4962c0eac51c41ba440b188bb4726293f356ec9bec780f7d1a6f08b8";

/// Byte size of that build; checked before hashing.
pub(crate) const SILERO_VAD_V6_2_BYTES: u64 = 1_557_280;

/// Whether `path` holds exactly `bytes` bytes whose SHA-256 is `sha`.
///
/// The size is compared first, so a file of another size is rejected
/// without being read. Any I/O error counts as a mismatch.
pub(crate) fn file_matches_pin(path: &std::path::Path, bytes: u64, sha: &str) -> bool {
    use sha2::{Digest, Sha256};
    match std::fs::metadata(path) {
        Ok(meta) if meta.is_file() && meta.len() == bytes => {}
        _ => return false,
    }
    let Ok(data) = std::fs::read(path) else {
        return false;
    };
    let digest = Sha256::digest(&data);
    let actual = digest
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    actual == sha
}

/// Whether the Silero VAD file at `path` is the pinned v6.2 build.
///
/// The VAD keeps the same on-disk name across versions, so presence
/// alone cannot tell the v5 file of earlier releases from v6.2. The
/// catalog reports an outdated file as not installed (the Models panel
/// then offers the update) and the shell warns when it has to load one.
pub(crate) fn is_current_silero_vad(path: &std::path::Path) -> bool {
    file_matches_pin(path, SILERO_VAD_V6_2_BYTES, SILERO_VAD_V6_2_SHA256)
}

/// Catalog of models the app can download, with their HF URLs and
/// expected sizes. Only includes models the priority resolvers know.
///
/// Each entry carries an optional SHA-256 hex digest. When present,
/// downloads are verified after completion and rejected on mismatch.
/// Populate hashes by running `shasum -a 256 <model_file>`.
///
/// Model builds EchoNote produces itself (inlined Silero VAD,
/// tract-compatible segmenter) are assets of release v0.6.0; see
/// `docs/releases/v0.6.0-model-assets.md` for provenance and upload steps.
fn model_catalog(root: &std::path::Path) -> Vec<(ModelInfo, &'static str, Option<&'static str>)> {
    let present = |rel: &str| root.join(rel).exists();

    vec![
        (
            ModelInfo {
                id: "asr-large-v3-turbo".into(),
                label: "Whisper Turbo".into(),
                description: "Multilingual · high accuracy · medium speed".into(),
                kind: "asr".into(),
                present: present("models/asr/ggml-large-v3-turbo.bin"),
                size_bytes: 1_600_000_000,
            },
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
            Some("1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69"),
        ),
        (
            ModelInfo {
                id: "asr-large-v3-turbo-q5".into(),
                label: "Whisper Turbo Lite".into(),
                description: "Multilingual · good accuracy · faster".into(),
                kind: "asr".into(),
                present: present("models/asr/ggml-large-v3-turbo-q5_0.bin"),
                size_bytes: 574_000_000,
            },
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin",
            Some("394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2"),
        ),
        (
            ModelInfo {
                id: "asr-distil-large-v3".into(),
                label: "Whisper Fast".into(),
                description: "English only · 5x faster".into(),
                kind: "asr".into(),
                present: present("models/asr/ggml-distil-large-v3.bin"),
                size_bytes: 756_000_000,
            },
            "https://huggingface.co/distil-whisper/distil-large-v3-ggml/resolve/main/ggml-distil-large-v3.bin",
            Some("2883a11b90fb10ed592d826edeaee7d2929bf1ab985109fe9e1e7b4d2b69a298"),
        ),
        (
            ModelInfo {
                id: "asr-medium".into(),
                label: "Whisper Medium".into(),
                description: "Multilingual · medium accuracy".into(),
                kind: "asr".into(),
                present: present("models/asr/ggml-medium.bin"),
                size_bytes: 1_530_000_000,
            },
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
            Some("6c14d5adee5f86394037b4e4e8b59f1673b6cee10e3cf0b11bbdbee79c156208"),
        ),
        (
            ModelInfo {
                id: "asr-small".into(),
                label: "Whisper Small".into(),
                description: "Multilingual · lightweight".into(),
                kind: "asr".into(),
                present: present("models/asr/ggml-small.bin"),
                size_bytes: 488_000_000,
            },
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
            Some("1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b"),
        ),
        (
            ModelInfo {
                id: "asr-base".into(),
                label: "Whisper Base".into(),
                description: "Multilingual · minimal".into(),
                kind: "asr".into(),
                present: present("models/asr/ggml-base.bin"),
                size_bytes: 148_000_000,
            },
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
            Some("60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe"),
        ),
        (
            ModelInfo {
                id: "llm-qwen3-14b".into(),
                label: "Qwen 3 Large".into(),
                description: "Detailed summaries · requires 16 GB RAM".into(),
                kind: "llm".into(),
                present: present("models/llm/Qwen3-14B-Q4_K_M.gguf"),
                size_bytes: 9_200_000_000,
            },
            "https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q4_K_M.gguf",
            Some("500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0"),
        ),
        (
            ModelInfo {
                id: "llm-qwen3-8b".into(),
                label: "Qwen 3 Medium".into(),
                description: "Good quality/speed balance".into(),
                kind: "llm".into(),
                present: present("models/llm/Qwen3-8B-Q4_K_M.gguf"),
                size_bytes: 5_200_000_000,
            },
            "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
            Some("d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"),
        ),
        (
            ModelInfo {
                id: "llm-qwen3-4b".into(),
                label: "Qwen 3 Lite".into(),
                description: "For devices with less than 8 GB RAM".into(),
                kind: "llm".into(),
                present: present("models/llm/Qwen3-4B-Q4_K_M.gguf"),
                size_bytes: 2_600_000_000,
            },
            "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf",
            Some("7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5"),
        ),
        // Gemma 4 (Google, Apache 2.0) QAT q4_0 builds. Offered next to Qwen;
        // never the default, only used once the user selects one. RAM figures
        // are estimates (weights + ~8K-token KV cache), not measurements.
        (
            ModelInfo {
                id: "llm-gemma4-e4b".into(),
                label: "Gemma 4 E4B".into(),
                description: "Compact Google model · about 7 GB RAM in use".into(),
                kind: "llm".into(),
                present: present("models/llm/gemma-4-E4B-it-qat-q4_0.gguf"),
                size_bytes: 5_154_941_280,
            },
            "https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf/resolve/main/gemma-4-E4B_q4_0-it.gguf",
            Some("676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee"),
        ),
        (
            ModelInfo {
                id: "llm-gemma4-12b".into(),
                label: "Gemma 4 12B".into(),
                description: "Large Google model · about 10 GB RAM in use, 24 GB or more recommended"
                    .into(),
                kind: "llm".into(),
                present: present("models/llm/gemma-4-12B-it-qat-q4_0.gguf"),
                size_bytes: 6_975_879_296,
            },
            "https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf/resolve/main/gemma-4-12b-it-qat-q4_0.gguf",
            Some("93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b"),
        ),
        (
            ModelInfo {
                id: "vad-silero".into(),
                label: "Silero VAD".into(),
                description: "Detects when someone is speaking".into(),
                kind: "vad".into(),
                present: is_current_silero_vad(&root.join("models/vad/silero_vad.onnx")),
                size_bytes: SILERO_VAD_V6_2_BYTES,
            },
            // Silero VAD v6.2.3 with its 16 kHz branch inlined: the upstream
            // file dispatches through an ONNX `If` that tract cannot execute.
            // Presence is pinned to this build so v5 files from earlier
            // releases show up as an available update.
            "https://github.com/luismctech/echonote/releases/download/v0.6.0/silero_vad_v6_2.onnx",
            Some(SILERO_VAD_V6_2_SHA256),
        ),
        (
            ModelInfo {
                id: "embedder-eres2net".into(),
                label: "ERes2Net".into(),
                description: "Speaker identification · general purpose".into(),
                kind: "embedder".into(),
                present: present("models/embedder/eres2net_en_voxceleb.onnx"),
                size_bytes: 26_000_000,
            },
            "https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main/3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx",
            Some("c59158379255ad66e161679cca6af8d52d51e389e3224ab7d7a7baae295c2db5"),
        ),
        (
            ModelInfo {
                id: "embedder-camplusplus".into(),
                label: "CAM++".into(),
                description: "Speaker identification · English (VoxCeleb)".into(),
                kind: "embedder".into(),
                present: present("models/embedder/campplus_en_voxceleb.onnx"),
                size_bytes: 28_000_000,
            },
            "https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
            Some("357a834f702b80161e5b981182c038e18553c1f2ca752ed6cec2052365d4129b"),
        ),
        (
            ModelInfo {
                id: "embedder-campplus-multi".into(),
                label: "CAM++ Multilingüe".into(),
                description: "Speaker identification · multilingual (ZH+EN), better on Spanish".into(),
                kind: "embedder".into(),
                present: present("models/embedder/campplus_zh_en_common_advanced.onnx"),
                size_bytes: 28_281_164,
            },
            "https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
            Some("aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2"),
        ),
        (
            ModelInfo {
                id: "segmenter-pyannote".into(),
                label: "pyannote 3.0".into(),
                description: "Detects speaker changes in conversation".into(),
                kind: "segmenter".into(),
                present: present("models/segmenter/pyannote_segmentation_3_tract.onnx"),
                size_bytes: 5_988_123,
            },
            // InstanceNorm-decomposed build of the sherpa-onnx export
            // (`scripts/patch-pyannote-segmenter.py`): the upstream file does
            // not load in tract 0.22, so it never enabled segmentation.
            "https://github.com/luismctech/echonote/releases/download/v0.6.0/pyannote_segmentation_3_tract.onnx",
            Some("f7e21b2898bf772836ac5784b91663d7cb869f8899e5c501e42ad747fb5a7b78"),
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
        "llm-gemma4-e4b" => Some("models/llm/gemma-4-E4B-it-qat-q4_0.gguf"),
        "llm-gemma4-12b" => Some("models/llm/gemma-4-12B-it-qat-q4_0.gguf"),
        "vad-silero" => Some("models/vad/silero_vad.onnx"),
        "embedder-eres2net" => Some("models/embedder/eres2net_en_voxceleb.onnx"),
        "embedder-camplusplus" => Some("models/embedder/campplus_en_voxceleb.onnx"),
        "embedder-campplus-multi" => Some("models/embedder/campplus_zh_en_common_advanced.onnx"),
        "segmenter-pyannote" => Some("models/segmenter/pyannote_segmentation_3_tract.onnx"),
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
/// Supports resuming interrupted downloads via HTTP Range requests. If a
/// `.part` file exists from a previous attempt, the download continues from
/// where it left off (server must support 206 Partial Content; falls back to
/// full restart if it responds 200).
///
/// SHA-256 verification is performed on fresh (non-resumed) downloads only.
/// Resumed downloads skip verification because we cannot reconstruct the hash
/// of bytes already written in a prior session without re-reading the file.
///
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

    // ── Check for resumable partial file ─────────────────────────
    let resume_offset: u64 = if tmp.exists() {
        tokio::fs::metadata(&tmp)
            .await
            .map(|m| m.len())
            .unwrap_or(0)
    } else {
        0
    };

    let client = state.http_client.clone();

    let result: Result<(), IpcError> = async {
        use sha2::{Digest, Sha256};
        use tokio::io::AsyncWriteExt;

        // Build request; add Range header when we have a partial file.
        let mut request = client.get(&url);
        if resume_offset > 0 {
            request = request.header("Range", format!("bytes={resume_offset}-"));
            tracing::info!(%model_id, resume_offset, "attempting to resume download");
        }

        let response = request
            .send()
            .await
            .map_err(|e| IpcError::network(format!("HTTP request failed: {e}")))?;

        let status = response.status().as_u16();
        // 200 = full content (server ignored Range or fresh start)
        // 206 = partial content (server accepted resume)
        if status != 200 && status != 206 {
            return Err(IpcError::network(format!("HTTP {}", response.status())));
        }

        // If server returned 200 despite our Range header, restart from byte 0.
        let actual_resume = if status == 206 { resume_offset } else { 0 };
        if status == 200 && resume_offset > 0 {
            tracing::warn!(%model_id, "server does not support Range; restarting download from scratch");
        }

        let content_length = response.content_length().unwrap_or(0);
        let total = content_length.saturating_add(actual_resume);
        let mut downloaded = actual_resume;

        // Open for append on resume, create (truncate) on fresh start.
        let mut file = if actual_resume > 0 {
            tokio::fs::OpenOptions::new()
                .append(true)
                .open(&tmp)
                .await
                .map_err(|e| IpcError::storage(format!("failed to open partial file: {e}")))?
        } else {
            tokio::fs::File::create(&tmp)
                .await
                .map_err(|e| IpcError::storage(format!("failed to create file: {e}")))?
        };

        // SHA-256 is only computed on full (non-resumed) downloads. For
        // resumed downloads we cannot reconstruct the hash of the already-
        // downloaded portion without re-reading the .part file.
        let verify_hash = expected_sha.is_some() && actual_resume == 0;
        let mut hasher = Sha256::new();

        let mut stream = response.bytes_stream();
        let mut last_report = std::time::Instant::now();

        while let Some(chunk) = stream.next().await {
            // ── Check cancel flag ────────────────────────────────
            if cancel_flag.load(std::sync::atomic::Ordering::Relaxed) {
                drop(file);
                let _ = on_event.send(DownloadEvent::Cancelled);
                return Ok(());
            }

            let chunk =
                chunk.map_err(|e| IpcError::network(format!("download stream error: {e}")))?;
            file.write_all(&chunk)
                .await
                .map_err(|e| IpcError::storage(format!("write error: {e}")))?;
            if verify_hash {
                hasher.update(&chunk);
            }
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

        // ── SHA-256 verification ──────────────────────────────────
        if verify_hash {
            if let Some(expected) = &expected_sha {
                let hash = hasher.finalize();
                let actual = hash.iter().map(|b| format!("{b:02x}")).collect::<String>();
                if actual != *expected {
                    let _ = tokio::fs::remove_file(&tmp).await;
                    return Err(IpcError::new(
                        ErrorCode::InvalidInput,
                        format!(
                            "SHA-256 mismatch for {model_id}: expected {expected}, got {actual}"
                        ),
                    ));
                }
                tracing::info!(%model_id, "SHA-256 verified");
            }
        } else if expected_sha.is_some() && actual_resume > 0 {
            tracing::warn!(
                %model_id,
                "SHA-256 skipped for resumed download — \
                 delete and re-download to verify integrity."
            );
        }

        tokio::fs::rename(&tmp, &dest)
            .await
            .map_err(|e| IpcError::storage(format!("rename failed: {e}")))?;

        Ok(())
    }
    .await;

    match result {
        Ok(()) => {
            // A cached VAD still holds the replaced file; drop it so the next
            // session loads the new build without an app restart.
            if model_id.starts_with("vad-") {
                state.vad.unload().await;
            }
            let _ = on_event.send(DownloadEvent::Finished);
            Ok(())
        }
        Err(e) => {
            // Leave the .part file on disk so the next download attempt can resume.
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
    // Persist so the selection survives app restart.
    super::save_preference(&state.data_root, "llm", rel_path).await?;

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
    super::path_to_catalog_id(&path)
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
    // Persist so the selection survives app restart.
    super::save_preference(&state.data_root, "embedder", rel_path).await?;
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
    super::path_to_catalog_id(&path)
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
    // Persist so the selection survives app restart.
    super::save_preference(&state.data_root, "asr", rel_path).await?;

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
    super::path_to_catalog_id(&path)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::PathBuf;

    /// Fresh empty directory under the system temp dir, removed on drop.
    struct TempRoot(PathBuf);

    impl TempRoot {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "echonote-models-{name}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_nanos())
            ));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }

    impl Drop for TempRoot {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn entry(id: &str) -> (ModelInfo, &'static str, Option<&'static str>) {
        model_catalog(std::path::Path::new("/nonexistent"))
            .into_iter()
            .find(|(info, _, _)| info.id == id)
            .unwrap_or_else(|| panic!("catalog has no {id}"))
    }

    #[test]
    fn gemma4_e4b_points_at_the_pinned_qat_file() {
        let (info, url, sha) = entry("llm-gemma4-e4b");
        assert_eq!(info.kind, "llm");
        assert_eq!(info.size_bytes, 5_154_941_280);
        assert_eq!(
            url,
            "https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf/resolve/main/gemma-4-E4B_q4_0-it.gguf"
        );
        assert_eq!(
            sha,
            Some("676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee")
        );
        assert_eq!(
            model_dest_path("llm-gemma4-e4b"),
            Some("models/llm/gemma-4-E4B-it-qat-q4_0.gguf")
        );
    }

    #[test]
    fn gemma4_12b_points_at_the_pinned_qat_file() {
        let (info, url, sha) = entry("llm-gemma4-12b");
        assert_eq!(info.kind, "llm");
        assert_eq!(info.size_bytes, 6_975_879_296);
        assert_eq!(
            url,
            "https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf/resolve/main/gemma-4-12b-it-qat-q4_0.gguf"
        );
        assert_eq!(
            sha,
            Some("93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b")
        );
        assert_eq!(
            model_dest_path("llm-gemma4-12b"),
            Some("models/llm/gemma-4-12B-it-qat-q4_0.gguf")
        );
    }

    #[test]
    fn every_llm_entry_can_be_selected_with_set_active_llm() {
        // `set_active_llm` needs a destination path and an `llm-` id.
        let llms: Vec<_> = model_catalog(std::path::Path::new("/nonexistent"))
            .into_iter()
            .map(|(info, _, _)| info)
            .filter(|info| info.kind == "llm")
            .collect();
        let ids: Vec<&str> = llms.iter().map(|i| i.id.as_str()).collect();
        for id in [
            "llm-qwen3-14b",
            "llm-qwen3-8b",
            "llm-qwen3-4b",
            "llm-gemma4-e4b",
            "llm-gemma4-12b",
        ] {
            assert!(ids.contains(&id), "{id} missing from {ids:?}");
        }
        for info in &llms {
            assert!(info.id.starts_with("llm-"), "{}", info.id);
            assert!(model_dest_path(&info.id).is_some(), "{}", info.id);
        }
    }

    #[test]
    fn catalog_destinations_agree_with_presence_and_the_reverse_lookup() {
        let root = TempRoot::new("dest");
        let ids: Vec<String> = model_catalog(&root.0)
            .into_iter()
            .map(|(info, _, _)| info.id)
            .collect();
        for id in &ids {
            let rel = model_dest_path(id).unwrap_or_else(|| panic!("no dest for {id}"));
            let path = root.0.join(rel);
            // Not downloaded yet.
            let before = model_catalog(&root.0)
                .into_iter()
                .find(|(info, _, _)| &info.id == id)
                .unwrap();
            assert!(!before.0.present, "{id}");
            // Dropping the file at its destination marks it present...
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"x").unwrap();
            let after = model_catalog(&root.0)
                .into_iter()
                .find(|(info, _, _)| &info.id == id)
                .unwrap();
            // ...except for the Silero VAD, whose presence is pinned to
            // the exact v6.2 build (see `is_current_silero_vad`).
            assert_eq!(after.0.present, id != "vad-silero", "{id}");
            // ...and the active-model getters can name it again.
            assert_eq!(
                super::super::path_to_catalog_id(&path).as_deref(),
                Some(id.as_str())
            );
        }
    }

    #[test]
    fn every_catalog_embedder_dispatches_by_its_id() {
        // A catalog embedder must name its own family: without one it
        // would silently load through the file-name fallback.
        let embedders: Vec<String> = model_catalog(std::path::Path::new("/nonexistent"))
            .into_iter()
            .filter(|(info, _, _)| info.kind == "embedder")
            .map(|(info, _, _)| info.id)
            .collect();
        assert!(!embedders.is_empty());
        for id in &embedders {
            let family = echo_diarize::EmbedderFamily::from_model_id(id)
                .unwrap_or_else(|| panic!("catalog embedder {id} has no family"));
            let rel = model_dest_path(id).unwrap_or_else(|| panic!("no dest for {id}"));
            let path = std::path::Path::new("/data").join(rel);
            assert_eq!(super::super::embedder_family(&path), family, "{id}");
        }
    }

    #[test]
    fn the_en_campplus_is_not_advertised_for_spanish() {
        // T6/T6b DER benchmark: the multilingual CAM++ beats the EN
        // CAM++ on Spanish (and on English).
        let (info, _, _) = entry("embedder-camplusplus");
        assert!(
            !info.description.contains("Spanish"),
            "{}",
            info.description
        );
    }

    #[test]
    fn catalog_ids_are_unique() {
        let ids: Vec<String> = model_catalog(std::path::Path::new("/nonexistent"))
            .into_iter()
            .map(|(info, _, _)| info.id)
            .collect();
        let mut sorted = ids.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), ids.len(), "{ids:?}");
    }

    #[test]
    fn vad_silero_points_at_the_v6_2_asset_of_release_v0_6_0() {
        let (info, url, sha) = entry("vad-silero");
        assert_eq!(info.kind, "vad");
        assert_eq!(info.size_bytes, SILERO_VAD_V6_2_BYTES);
        assert_eq!(info.size_bytes, 1_557_280);
        assert_eq!(
            url,
            "https://github.com/luismctech/echonote/releases/download/v0.6.0/silero_vad_v6_2.onnx"
        );
        assert_eq!(sha, Some(SILERO_VAD_V6_2_SHA256));
        assert_eq!(
            SILERO_VAD_V6_2_SHA256,
            "76d1667a4962c0eac51c41ba440b188bb4726293f356ec9bec780f7d1a6f08b8"
        );
        assert_eq!(
            model_dest_path("vad-silero"),
            Some("models/vad/silero_vad.onnx")
        );
    }

    #[test]
    fn segmenter_points_at_the_tract_compatible_asset_of_release_v0_6_0() {
        // The upstream sherpa export does not load in tract 0.22, so the
        // catalog ships the InstanceNorm-decomposed file the shell prefers.
        let (info, url, sha) = entry("segmenter-pyannote");
        assert_eq!(info.kind, "segmenter");
        assert_eq!(info.size_bytes, 5_988_123);
        assert_eq!(
            url,
            "https://github.com/luismctech/echonote/releases/download/v0.6.0/pyannote_segmentation_3_tract.onnx"
        );
        assert_eq!(
            sha,
            Some("f7e21b2898bf772836ac5784b91663d7cb869f8899e5c501e42ad747fb5a7b78")
        );
        assert_eq!(
            model_dest_path("segmenter-pyannote"),
            Some("models/segmenter/pyannote_segmentation_3_tract.onnx")
        );
    }

    #[test]
    fn an_upstream_only_segmenter_install_is_not_reported_present() {
        // Users who downloaded the upstream export before v0.6.0 never had
        // a working segmenter: the catalog must offer the tract file.
        let root = TempRoot::new("seg-upstream");
        let upstream = root.0.join("models/segmenter/pyannote_segmentation_3.onnx");
        std::fs::create_dir_all(upstream.parent().unwrap()).unwrap();
        std::fs::write(&upstream, b"x").unwrap();
        let (info, _, _) = model_catalog(&root.0)
            .into_iter()
            .find(|(info, _, _)| info.id == "segmenter-pyannote")
            .unwrap();
        assert!(!info.present);
    }

    #[test]
    fn file_matches_pin_needs_the_exact_size_and_digest() {
        let root = TempRoot::new("pin");
        let path = root.0.join("abc.bin");
        std::fs::write(&path, b"abc").unwrap();
        const ABC: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert!(file_matches_pin(&path, 3, ABC));
        assert!(!file_matches_pin(&path, 4, ABC), "size mismatch");
        assert!(
            !file_matches_pin(&path, 3, SILERO_VAD_V6_2_SHA256),
            "digest mismatch"
        );
        assert!(!file_matches_pin(&root.0.join("missing"), 3, ABC));
    }

    #[test]
    fn an_older_silero_build_at_the_vad_path_is_outdated() {
        // The v5 file of release v0.2.1 (1,251,310 B) and any other build
        // with the v6.2 size but different bytes are not the pinned v6.2.
        let root = TempRoot::new("vad-old");
        let path = root.0.join("models/vad/silero_vad.onnx");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        for len in [1_251_310_usize, SILERO_VAD_V6_2_BYTES as usize] {
            std::fs::write(&path, vec![0_u8; len]).unwrap();
            assert!(!is_current_silero_vad(&path), "{len}");
            let (info, _, _) = model_catalog(&root.0)
                .into_iter()
                .find(|(info, _, _)| info.id == "vad-silero")
                .unwrap();
            assert!(!info.present, "{len}");
        }
    }
}
