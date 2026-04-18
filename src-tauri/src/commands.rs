//! Tauri IPC commands exposed to the frontend.
//!
//! Each command here mirrors a typed contract in
//! `src/lib/ipc.ts`. When the surface grows beyond a handful, switch to
//! `tauri-specta` code generation — see ADR note in
//! `docs/adr/0002-rust-plus-react-stack.md`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use tauri::ipc::Channel;
use tauri::State;
use time::{format_description::well_known::Rfc3339, OffsetDateTime};
use tokio::sync::Mutex as AsyncMutex;
use tokio::task::JoinHandle;

use echo_app::{StreamingHandle, StreamingPipeline};
use echo_asr::WhisperCppTranscriber;
use echo_audio::{CpalMicrophoneCapture, RubatoResamplerAdapter};
use echo_domain::{
    AudioCapture, AudioFormat, AudioSource, CaptureSpec, Resampler, StreamingOptions,
    StreamingSessionId, Transcriber, TranscriptEvent,
};

// ---------------------------------------------------------------------------
// Health check (Sprint 0 day 4)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Streaming pipeline (Sprint 0 day 7)
// ---------------------------------------------------------------------------

/// Shared state injected through `tauri::Builder::manage`. Holds the
/// shared adapters so the model is loaded once per app session and the
/// in-flight streaming sessions so they can be stopped from the UI.
pub struct AppState {
    capture: Arc<dyn AudioCapture>,
    resampler: Arc<dyn Resampler>,
    /// Async-locked: the whisper context is heavy and we only build it
    /// on first use. `Option` so the loader can take ownership during
    /// initialization without holding the lock across the disk read.
    transcriber: AsyncMutex<Option<Arc<dyn Transcriber>>>,
    model_path: PathBuf,
    sessions: Mutex<HashMap<StreamingSessionId, SessionEntry>>,
}

struct SessionEntry {
    join: JoinHandle<()>,
    handle: Arc<AsyncMutex<StreamingHandle>>,
}

impl AppState {
    /// Build the shared state. The transcriber is *not* loaded eagerly
    /// — the first `start_streaming` call will pay that cost (~150 ms
    /// for `base.en` on Apple Silicon).
    pub fn new() -> Self {
        let model_path = std::env::var("ECHO_ASR_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("./models/asr/ggml-base.en.bin"));
        Self {
            capture: Arc::new(CpalMicrophoneCapture::new()),
            resampler: Arc::new(RubatoResamplerAdapter),
            transcriber: AsyncMutex::new(None),
            model_path,
            sessions: Mutex::new(HashMap::new()),
        }
    }

    async fn ensure_transcriber(&self) -> Result<Arc<dyn Transcriber>, String> {
        let mut slot = self.transcriber.lock().await;
        if let Some(t) = slot.as_ref() {
            return Ok(t.clone());
        }
        if !self.model_path.exists() {
            return Err(format!(
                "model not found at {}. Set ECHO_ASR_MODEL or run `scripts/download-models.sh base.en`.",
                self.model_path.display()
            ));
        }
        let path = self.model_path.clone();
        let loaded = tokio::task::spawn_blocking(move || WhisperCppTranscriber::load(&path))
            .await
            .map_err(|e| format!("whisper load task panicked: {e}"))?
            .map_err(|e| format!("failed to load whisper model: {e}"))?;
        let arc: Arc<dyn Transcriber> = Arc::new(loaded);
        *slot = Some(arc.clone());
        Ok(arc)
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

/// Options the frontend may pass when starting a streaming session.
/// All fields optional; defaults match `StreamingOptions::default()`.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StartStreamingOptions {
    /// ISO-639-1 language hint. `None` ⇒ auto-detect.
    pub language: Option<String>,
    /// Capture device id. `None` ⇒ system default microphone.
    pub device_id: Option<String>,
    /// Audio chunk size in milliseconds. `None` ⇒ 5000.
    pub chunk_ms: Option<u32>,
    /// RMS threshold below which a chunk is reported as `Skipped`
    /// instead of being sent to the ASR. `None` ⇒ 0.005.
    pub silence_rms_threshold: Option<f32>,
}

/// Start a streaming transcription session. Events are pushed through
/// the supplied `Channel<TranscriptEvent>` until [`stop_streaming`] is
/// invoked or the capture stream ends.
#[tauri::command]
pub async fn start_streaming(
    state: State<'_, AppState>,
    options: Option<StartStreamingOptions>,
    on_event: Channel<TranscriptEvent>,
) -> Result<StreamingSessionId, String> {
    let opts = options.unwrap_or_default();
    let streaming_options = StreamingOptions {
        language: opts.language,
        chunk_ms: opts.chunk_ms.unwrap_or(5_000),
        silence_rms_threshold: opts.silence_rms_threshold.unwrap_or(0.005),
    };
    let spec = CaptureSpec {
        source: AudioSource::Microphone,
        device_id: opts.device_id,
        preferred_format: AudioFormat::WHISPER,
    };

    let transcriber = state.ensure_transcriber().await?;
    let pipeline =
        StreamingPipeline::new(state.capture.clone(), state.resampler.clone(), transcriber);

    let handle = pipeline
        .start_with_spec(spec, streaming_options)
        .await
        .map_err(|e| format!("failed to start streaming: {e}"))?;
    let session_id = handle.session_id();

    // Drain the event receiver in a background task and forward to the
    // IPC channel. The handle moves into an Arc<AsyncMutex<…>> so
    // stop_streaming can take ownership without racing the drain task.
    let handle_arc = Arc::new(AsyncMutex::new(handle));
    let drain_handle = handle_arc.clone();
    let join = tokio::spawn(async move {
        loop {
            let mut guard = drain_handle.lock().await;
            let evt = guard.next_event().await;
            drop(guard);
            match evt {
                Some(evt) => {
                    let terminal = matches!(
                        evt,
                        TranscriptEvent::Stopped { .. } | TranscriptEvent::Failed { .. }
                    );
                    if let Err(e) = on_event.send(evt) {
                        tracing::warn!(error = %e, %session_id, "frontend channel send failed");
                        break;
                    }
                    if terminal {
                        break;
                    }
                }
                None => break,
            }
        }
    });

    state
        .sessions
        .lock()
        .map_err(|e| format!("session map poisoned: {e}"))?
        .insert(
            session_id,
            SessionEntry {
                join,
                handle: handle_arc,
            },
        );

    Ok(session_id)
}

/// Stop a previously-started streaming session. Idempotent: returns
/// `Ok(false)` when the session id is unknown (already stopped or
/// never existed).
#[tauri::command]
pub async fn stop_streaming(
    state: State<'_, AppState>,
    session_id: StreamingSessionId,
) -> Result<bool, String> {
    let entry = state
        .sessions
        .lock()
        .map_err(|e| format!("session map poisoned: {e}"))?
        .remove(&session_id);
    let Some(entry) = entry else {
        return Ok(false);
    };
    {
        let mut guard = entry.handle.lock().await;
        guard
            .stop()
            .await
            .map_err(|e| format!("failed to stop pipeline: {e}"))?;
    }
    let _ = entry.join.await;
    Ok(true)
}
