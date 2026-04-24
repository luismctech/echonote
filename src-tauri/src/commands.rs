//! Tauri IPC commands exposed to the frontend.
//!
//! Each command here mirrors a typed contract in `src/ipc/client.ts`
//! (return shapes are mirrored by the pure TS types under `src/types/`).
//! When the surface grows beyond a handful, switch to `tauri-specta`
//! code generation — see ADR note in
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

use crate::ipc_error::{ErrorCode, IpcError};

use echo_app::{
    AskAboutMeeting, AskAboutMeetingEvent, MeetingRecorder, RenameSpeaker, StreamingHandle,
    StreamingPipeline, SummarizeMeeting,
};
use echo_asr::WhisperCppTranscriber;
use echo_audio::{RoutingAudioCapture, RubatoResamplerAdapter, SileroVad};
use echo_diarize::{Eres2NetEmbedder, OnlineDiarizer};
use echo_domain::{
    AudioCapture, AudioFormat, AudioSource, CaptureSpec, ChatAssistant, ChatMessage, Diarizer,
    LlmModel, Meeting, MeetingId, MeetingSearchHit, MeetingStore, MeetingSummary, Resampler,
    SpeakerId, StreamingOptions, StreamingSessionId, Summary, Transcriber, TranscriptEvent, Vad,
};
use echo_llm::{LlamaCppLlm, LoadOptions as LlamaLoadOptions};
use echo_storage::SqliteMeetingStore;
use futures::stream::StreamExt;

// ---------------------------------------------------------------------------
// Health check (Sprint 0 day 4)
// ---------------------------------------------------------------------------

/// Result returned by [`health_check`]. Mirrors `HealthStatus` on the TS side.
#[derive(Debug, Serialize, specta::Type)]
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
#[specta::specta]
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
// Streaming pipeline (Sprint 0 day 7) + persistence (Sprint 0 day 8)
// ---------------------------------------------------------------------------

/// Shared state injected through `tauri::Builder::manage`. Holds the
/// shared adapters so the model is loaded once per app session, the
/// SQLite-backed meeting store, the per-session meeting recorder and
/// the in-flight streaming sessions so they can be stopped from the UI.
pub struct AppState {
    capture: Arc<dyn AudioCapture>,
    resampler: Arc<dyn Resampler>,
    /// Async-locked: the whisper context is heavy and we only build it
    /// on first use.
    transcriber: AsyncMutex<Option<Arc<dyn Transcriber>>>,
    model_path: PathBuf,
    /// Where to find the speaker embedder ONNX. The diarizer is opt-in
    /// per-session (`StartStreamingOptions::diarize`); this just
    /// records the default location so the UI does not have to know it.
    embed_model_path: PathBuf,
    store: Arc<dyn MeetingStore>,
    recorder: Arc<MeetingRecorder>,
    rename_speaker: Arc<RenameSpeaker>,
    /// In-flight streaming sessions. Wrapped in `Arc` so the drain
    /// task spawned by `start_streaming` can self-cleanup its own
    /// entry on terminal events without going through `tauri::State`.
    sessions: Arc<Mutex<HashMap<StreamingSessionId, SessionEntry>>>,
    /// Path to the local LLM (.gguf) used for `summarize_meeting` +
    /// `ask_about_meeting`. Resolved at startup via `ECHO_LLM_MODEL`
    /// or `models/llm/<preferred>.gguf`. The model itself is loaded
    /// lazily on first use — same pattern as the whisper transcriber
    /// — because the 14B Q4_K_M file pulls ~10 GB into RAM/VRAM and
    /// most app launches do not summarise.
    llm_model_path: PathBuf,
    /// Lazily-loaded LLM, held as the concrete [`LlamaCppLlm`] so we
    /// can derive both adapters (`LlmModel` for summary,
    /// `ChatAssistant` for chat) from the same loaded weights via
    /// [`LlamaCppLlm::chat_handle`]. Wrapped in `AsyncMutex` because
    /// the first caller pays the load cost cooperatively while the
    /// rest wait. Per `docs/SPRINT-1-STATUS.md` §8.3 the model is
    /// shared (~10 GB of weights) but each request still spins up its
    /// own short-lived `LlamaContext`, so chat + summary can run
    /// concurrently without serialization.
    llm: AsyncMutex<Option<Arc<LlamaCppLlm>>>,
    /// Path to the Silero VAD ONNX (`./models/vad/silero_vad.onnx` by
    /// default, overridable via `ECHO_VAD_MODEL`). The model is
    /// loaded lazily on first `start_streaming` call so app startup
    /// stays cheap; once loaded it lives behind an [`Arc`] and every
    /// new session clones it via [`SileroVad::clone_for_new_session`]
    /// (cheap Arc clone of the optimized graph + zeroed LSTM state).
    vad_model_path: PathBuf,
    /// Lazily-loaded Silero VAD template. `None` until the first
    /// session asks for it; `Some(Arc)` afterwards. We hold an
    /// `Arc<SileroVad>` (not `Box<dyn Vad>`) because the per-session
    /// clone helper is on the concrete type.
    ///
    /// `Option` (not just `Arc`) so we can degrade gracefully when
    /// the model file is missing: `start_streaming` proceeds with
    /// the energy-based RMS gate and logs a warning instead of
    /// failing outright. This keeps Sprint-0 setups working without
    /// requiring users to download an extra ~2 MB asset before
    /// recording.
    vad: AsyncMutex<Option<Arc<SileroVad>>>,
}

struct SessionEntry {
    join: JoinHandle<()>,
    handle: Arc<AsyncMutex<StreamingHandle>>,
}

impl AppState {
    /// Cheap clone of the session-map handle, intended for background
    /// tasks that need to remove their own entry on exit.
    fn sessions_handle(&self) -> Arc<Mutex<HashMap<StreamingSessionId, SessionEntry>>> {
        self.sessions.clone()
    }

    /// Ordered shutdown sequence invoked from the Tauri `Exit` hook.
    ///
    /// Drains any in-flight streaming session (best-effort: each
    /// drain task is bounded by a short timeout so we never hang the
    /// quit) and then closes the SQLite pool so WAL frames are
    /// checkpointed before the process exits. The shell follows this
    /// with an `_exit(0)` to sidestep ggml's atexit destructor, which
    /// would otherwise SIGABRT on macOS — but every Rust resource we
    /// can flush in-process is flushed first.
    pub async fn shutdown(&self) {
        // Drain sessions first, before closing the pool, since the
        // recorder writes the final `Stopped`/`Failed` rows from the
        // drain task.
        let entries: Vec<SessionEntry> = match self.sessions.lock() {
            Ok(mut map) => map.drain().map(|(_, e)| e).collect(),
            Err(poisoned) => {
                tracing::warn!("shutdown: session map poisoned; recovering");
                poisoned.into_inner().drain().map(|(_, e)| e).collect()
            }
        };
        let pending = entries.len();
        if pending > 0 {
            tracing::info!(pending, "shutdown: stopping in-flight sessions");
        }
        for entry in entries {
            // 1. Politely ask the streaming pipeline to stop. The
            //    handle's `stop` returns once the pipeline has flushed
            //    its tail audio + emitted the terminal event.
            let mut handle = entry.handle.lock().await;
            if let Err(e) = handle.stop().await {
                tracing::warn!(error = %e, "shutdown: stop_streaming failed");
            }
            drop(handle);
            // 2. Wait for the drain task to observe the terminal event,
            //    persist it, and exit. Bounded so a wedged task can't
            //    block app close indefinitely.
            match tokio::time::timeout(std::time::Duration::from_secs(5), entry.join).await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => tracing::warn!(error = %e, "shutdown: drain task panicked"),
                Err(_) => {
                    tracing::warn!("shutdown: drain task did not finish within 5s; abandoning")
                }
            }
        }
        // 3. Close the storage pool now that no producer can still be
        //    writing. Default impl is a no-op, the SQLite adapter
        //    flushes WAL frames here.
        self.store.close().await;
        tracing::info!("shutdown: complete");
    }

    /// Build the shared state. Async because opening the SQLite database
    /// runs migrations, which is I/O. The transcriber is *not* loaded
    /// eagerly — the first `start_streaming` call pays that cost.
    pub async fn initialize(app_data_dir: Option<PathBuf>) -> Result<Self, IpcError> {
        // Prefer the multilingual `ggml-base.bin` when it's installed
        // (the user may have downloaded it for Spanish / pt / fr / …),
        // and only fall back to the `.en`-only base if that's all that
        // exists. This keeps Sprint 0 setups working without forcing a
        // re-download, while making non-English audio actually
        // transcribe out of the box for users who ran
        // `scripts/download-models.sh base`.
        let model_path =
            resolve_asset_path(std::env::var("ECHO_ASR_MODEL").ok(), preferred_asr_model());
        let embed_model_path = resolve_asset_path(
            std::env::var("ECHO_EMBED_MODEL").ok(),
            // Matches what `scripts/download-models.sh embed` writes.
            "models/embedder/eres2net_en_voxceleb.onnx",
        );
        let llm_model_path =
            resolve_asset_path(std::env::var("ECHO_LLM_MODEL").ok(), preferred_llm_model());
        let vad_model_path = resolve_asset_path(
            std::env::var("ECHO_VAD_MODEL").ok(),
            // Matches what `scripts/download-models.sh vad` writes.
            "models/vad/silero_vad.onnx",
        );

        let db_path = resolve_db_path(app_data_dir);
        tracing::info!(
            asr_model = %model_path.display(),
            embed_model = %embed_model_path.display(),
            llm_model = %llm_model_path.display(),
            vad_model = %vad_model_path.display(),
            db_path = %db_path.display(),
            "echo-shell paths resolved"
        );
        let store = SqliteMeetingStore::open(&db_path).await.map_err(|e| {
            IpcError::storage(format!("open meeting store at {}: {e}", db_path.display()))
        })?;
        tracing::info!(db_path = %db_path.display(), "meeting store ready");

        let store: Arc<dyn MeetingStore> = Arc::new(store);
        let recorder = Arc::new(MeetingRecorder::with_default_title(store.clone()));
        let rename_speaker = Arc::new(RenameSpeaker::new(store.clone()));

        Ok(Self {
            capture: Arc::new(RoutingAudioCapture::with_default_adapters()),
            resampler: Arc::new(RubatoResamplerAdapter),
            transcriber: AsyncMutex::new(None),
            model_path,
            embed_model_path,
            store,
            recorder,
            rename_speaker,
            sessions: Arc::new(Mutex::new(HashMap::new())),
            llm_model_path,
            llm: AsyncMutex::new(None),
            vad_model_path,
            vad: AsyncMutex::new(None),
        })
    }

    /// Lazily load the local LLM. Cached for the lifetime of the
    /// process — model load is the expensive operation (mmap + CUDA/
    /// Metal init), per-prompt generation runs against the cached
    /// instance.
    ///
    /// Returns the concrete [`Arc<LlamaCppLlm>`] (rather than
    /// `Arc<dyn LlmModel>`) so callers that need other adapters from
    /// the same loaded model — currently [`Self::ensure_chat`] — can
    /// derive them via [`LlamaCppLlm::chat_handle`]. Trait-object
    /// callers ([`Self::ensure_llm`]) cast on the way out.
    async fn ensure_llm_concrete(&self) -> Result<Arc<LlamaCppLlm>, IpcError> {
        let mut slot = self.llm.lock().await;
        if let Some(m) = slot.as_ref() {
            return Ok(m.clone());
        }
        // Re-resolve: a model may have been downloaded at runtime via
        // the model manager after the initial path was set at startup.
        let path = if self.llm_model_path.exists() {
            self.llm_model_path.clone()
        } else {
            let fresh =
                resolve_asset_path(std::env::var("ECHO_LLM_MODEL").ok(), preferred_llm_model());
            if !fresh.exists() {
                return Err(IpcError::model_not_ready(format!(
                    "LLM model not found at {}. Download one from the Models panel \
                     or run `scripts/download-models.sh llm`.",
                    self.llm_model_path.display()
                )));
            }
            tracing::info!(path = %fresh.display(), "re-resolved LLM path after runtime download");
            fresh
        };
        let load_opts = LlamaLoadOptions::default();
        // `LlamaCppLlm::load_with` is synchronous (mmap + ggml init);
        // off-load to the blocking pool so we never stall the Tauri
        // command executor while a 10 GB model loads.
        let loaded = tokio::task::spawn_blocking({
            let path = path.clone();
            move || LlamaCppLlm::load_with(&path, load_opts)
        })
        .await
        .map_err(|e| IpcError::internal(format!("LLM load task panicked: {e}")))?
        .map_err(|e| {
            IpcError::new(
                ErrorCode::Llm,
                format!("failed to load LLM at {}: {e}", path.display()),
            )
        })?;
        let arc = Arc::new(loaded);
        *slot = Some(arc.clone());
        Ok(arc)
    }

    /// Lazy `LlmModel` accessor used by `summarize_meeting`. Same
    /// load cost (paid once per process) as [`Self::ensure_chat`];
    /// both share the cached `Arc<LlamaCppLlm>`.
    async fn ensure_llm(&self) -> Result<Arc<dyn LlmModel>, IpcError> {
        Ok(self.ensure_llm_concrete().await? as Arc<dyn LlmModel>)
    }

    /// Lazy `ChatAssistant` accessor used by `ask_about_meeting`.
    /// Calls [`LlamaCppLlm::chat_handle`] on the cached instance, so
    /// no extra weights are loaded — only the trait-object wrapper is
    /// allocated. Subsequent calls reuse the same loaded model.
    async fn ensure_chat(&self) -> Result<Arc<dyn ChatAssistant>, IpcError> {
        let llm = self.ensure_llm_concrete().await?;
        Ok(Arc::new(llm.chat_handle()) as Arc<dyn ChatAssistant>)
    }

    /// Build a fresh diarizer for the upcoming session. Each session
    /// gets its own `OnlineDiarizer` because the trait is stateful
    /// (the cluster centroids belong to a single recording). The
    /// embedder is reloaded from disk per session — the ONNX load is
    /// O(100ms), well under the user-perceptible threshold for a
    /// "Start" click — but cached `tract` graphs could be wired in if
    /// this ever shows up in profiling.
    fn build_diarizer(
        &self,
        override_path: Option<PathBuf>,
    ) -> Result<Box<dyn Diarizer>, IpcError> {
        let path = override_path.unwrap_or_else(|| self.embed_model_path.clone());
        if !path.exists() {
            return Err(IpcError::model_not_ready(format!(
                "speaker embedder not found at {}. Run `scripts/download-models.sh embed` \
                 or set ECHO_EMBED_MODEL.",
                path.display()
            )));
        }
        let started = std::time::Instant::now();
        let embedder = Eres2NetEmbedder::new(&path).map_err(|e| {
            IpcError::new(
                ErrorCode::Diarization,
                format!("load speaker embedder at {}: {e}", path.display()),
            )
        })?;
        tracing::info!(
            model = %path.display(),
            load_ms = started.elapsed().as_millis() as u64,
            "speaker embedder ready"
        );
        Ok(Box::new(OnlineDiarizer::with_defaults(Box::new(embedder))))
    }

    /// Lazily load the Silero VAD ONNX once per process.
    ///
    /// Returns `Ok(None)` (instead of an error) when the model file
    /// is missing — callers fall back to the pure-RMS silence gate
    /// and the user sees a one-time warning in the logs. This keeps
    /// EchoNote usable without forcing every install to run
    /// `scripts/download-models.sh vad`.
    async fn ensure_vad(&self) -> Result<Option<Arc<SileroVad>>, IpcError> {
        let mut slot = self.vad.lock().await;
        if let Some(v) = slot.as_ref() {
            return Ok(Some(v.clone()));
        }
        if !self.vad_model_path.exists() {
            tracing::warn!(
                vad_model = %self.vad_model_path.display(),
                "Silero VAD model not found; falling back to RMS gate. Run \
                 `scripts/download-models.sh vad` for sharper voice/non-voice \
                 discrimination (recommended)."
            );
            return Ok(None);
        }
        let path = self.vad_model_path.clone();
        let started = std::time::Instant::now();
        // Loading + tract optimization is sync CPU work; offload so
        // the Tauri command executor can keep handling IPC while the
        // first session pays the cost.
        let loaded = tokio::task::spawn_blocking(move || SileroVad::for_meetings(&path))
            .await
            .map_err(|e| IpcError::internal(format!("Silero VAD load task panicked: {e}")))?
            .map_err(|e| {
                IpcError::new(
                    ErrorCode::Vad,
                    format!(
                        "failed to load Silero VAD at {}: {e}",
                        self.vad_model_path.display()
                    ),
                )
            })?;
        tracing::info!(
            vad_model = %self.vad_model_path.display(),
            load_ms = started.elapsed().as_millis() as u64,
            "Silero VAD ready"
        );
        let arc = Arc::new(loaded);
        *slot = Some(arc.clone());
        Ok(Some(arc))
    }

    async fn ensure_transcriber(&self) -> Result<Arc<dyn Transcriber>, IpcError> {
        let mut slot = self.transcriber.lock().await;
        if let Some(t) = slot.as_ref() {
            return Ok(t.clone());
        }
        let path = if self.model_path.exists() {
            self.model_path.clone()
        } else {
            let fresh =
                resolve_asset_path(std::env::var("ECHO_ASR_MODEL").ok(), preferred_asr_model());
            if !fresh.exists() {
                return Err(IpcError::model_not_ready(format!(
                    "ASR model not found at {}. Download one from the Models panel \
                     or run `scripts/download-models.sh`.",
                    self.model_path.display()
                )));
            }
            tracing::info!(path = %fresh.display(), "re-resolved ASR path after runtime download");
            fresh
        };
        let loaded = tokio::task::spawn_blocking(move || WhisperCppTranscriber::load(&path))
            .await
            .map_err(|e| IpcError::internal(format!("whisper load task panicked: {e}")))?
            .map_err(|e| {
                IpcError::new(ErrorCode::Asr, format!("failed to load whisper model: {e}"))
            })?;
        let arc: Arc<dyn Transcriber> = Arc::new(loaded);
        *slot = Some(arc.clone());
        Ok(arc)
    }
}

/// Resolve the SQLite database path. Honours `ECHO_DB_PATH` for tests
/// and falls back to `./echonote.db` (next to the binary). A real
/// Resolve the database file path.
///
/// Priority:
/// 1. `ECHO_DB_PATH` env var (absolute or workspace-relative) — for
///    developer overrides and CI.
/// 2. `app_data_dir` from Tauri (OS-appropriate persistent storage).
/// 3. `workspace_root()/echonote.db` — last resort for `cargo run`.
fn resolve_db_path(app_data_dir: Option<PathBuf>) -> PathBuf {
    if let Some(raw) = std::env::var("ECHO_DB_PATH")
        .ok()
        .filter(|s| !s.trim().is_empty())
    {
        let p = PathBuf::from(&raw);
        if p.is_absolute() {
            return p;
        }
        let rooted = workspace_root().join(&p);
        if rooted.exists() {
            return rooted;
        }
        return p;
    }
    if let Some(dir) = app_data_dir {
        return dir.join("echonote.db");
    }
    workspace_root().join("echonote.db")
}

/// Pick the LLM model to load by default, in priority order:
/// Qwen 3 first (better Spanish coverage and more recent training),
/// largest dense first then MoE then legacy Qwen 2.5 fallbacks for
/// back-compat with Sprint 1 day 9 setups. Resolution against the
/// workspace happens later in [`resolve_asset_path`]; here we only
/// pick a *relative* path that the resolver checks for existence.
fn preferred_llm_model() -> &'static str {
    const CANDIDATES: &[&str] = &[
        // Qwen 3 — current default family (Spanish-first). Filenames
        // mirror the official Qwen team's HF naming convention
        // (`Qwen3-<size>-Q4_K_M.gguf`, no `-Instruct-` infix).
        "models/llm/Qwen3-30B-A3B-Q4_K_M.gguf",
        "models/llm/Qwen3-14B-Q4_K_M.gguf",
        "models/llm/Qwen3-8B-Q4_K_M.gguf",
        // Qwen 2.5 — legacy fallback (kept for back-compat).
        "models/llm/qwen2.5-7b-instruct-q4_k_m.gguf",
        "models/llm/qwen2.5-3b-instruct-q4_k_m.gguf",
    ];
    let root = workspace_root();
    for rel in CANDIDATES {
        if root.join(rel).exists() {
            return rel;
        }
    }
    // Nothing installed yet — default to the canonical 14B Qwen 3 path
    // so the error message points the user at the right download
    // command (`scripts/download-models.sh llm`).
    "models/llm/Qwen3-14B-Q4_K_M.gguf"
}

/// Pick the ASR model to load by default, in priority order: the
/// Spanish fine-tune first (lowest WER on Spanish meetings), then
/// the largest installed multilingual ggml model so non-English
/// users get a working transcript without env overrides, then
/// English-only fallbacks for backwards compatibility with Sprint 0
/// setups. Resolution against the workspace happens later in
/// [`resolve_asset_path`]; here we only pick a *relative* path that
/// the resolver checks for existence.
fn preferred_asr_model() -> &'static str {
    const CANDIDATES: &[&str] = &[
        // Spanish-first multilingual — best for our target audience.
        "models/asr/ggml-large-v3-turbo-es.bin",
        "models/asr/ggml-large-v3-turbo.bin",
        // Quantized turbo — same model, ~60% smaller, minimal WER loss.
        "models/asr/ggml-large-v3-turbo-q5_0.bin",
        "models/asr/ggml-large-v3.bin",
        "models/asr/ggml-medium.bin",
        "models/asr/ggml-small.bin",
        "models/asr/ggml-base.bin",
        "models/asr/ggml-tiny.bin",
        // English-only fast option — distil-whisper is 5x faster with
        // minimal WER loss but only supports English.
        "models/asr/ggml-distil-large-v3.bin",
        // English-only fallbacks (Sprint 0 / dev / benchmark setups).
        "models/asr/ggml-base.en.bin",
        "models/asr/ggml-small.en.bin",
        "models/asr/ggml-tiny.en.bin",
    ];
    let root = workspace_root();
    for rel in CANDIDATES {
        if root.join(rel).exists() {
            return rel;
        }
    }
    // Nothing installed yet — default to multilingual turbo so the
    // error message points the user at the right download command
    // (`scripts/download-models.sh`).
    "models/asr/ggml-large-v3-turbo.bin"
}

/// Resolve an asset path with sensible dev-vs-prod fallbacks.
///
/// Order of resolution:
/// 1. The explicit override (env var) — used as-is when absolute, otherwise
///    treated relative to the workspace root.
/// 2. The default relative path resolved against the workspace root, derived
///    from `CARGO_MANIFEST_DIR` (which points at `src-tauri/`) by walking up
///    one level. This avoids the "model not found at ./models/..." footgun
///    when `tauri dev` launches the binary with cwd = `src-tauri/`.
/// 3. Finally, fall back to the path as-is, so `cargo run -p echo-shell`
///    from the workspace root still works.
fn resolve_asset_path(override_value: Option<String>, default_relative: &str) -> PathBuf {
    let workspace_root = workspace_root();

    if let Some(raw) = override_value.filter(|s| !s.trim().is_empty()) {
        let raw_path = PathBuf::from(&raw);
        if raw_path.is_absolute() {
            return raw_path;
        }
        // Try workspace-root-relative first; fall back to cwd-relative.
        let rooted = workspace_root.join(&raw_path);
        if rooted.exists() {
            return rooted;
        }
        return raw_path;
    }

    let rooted = workspace_root.join(default_relative);
    if rooted.exists() {
        return rooted;
    }
    // Either missing (so the caller will surface a useful error message
    // pointing at the workspace path) or running from a context where the
    // cwd happens to be correct.
    rooted
}

/// Best-effort workspace root: the directory above `src-tauri/` (where
/// `CARGO_MANIFEST_DIR` points). Falls back to the current working
/// directory if the parent is somehow missing.
fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
}

/// IPC mirror of [`echo_domain::AudioSource`] with camelCase naming
/// so the frontend stays stylistically consistent (the domain enum
/// uses snake_case for storage / CLI compatibility).
#[derive(Debug, Clone, Copy, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub enum IpcAudioSource {
    /// Default microphone via cpal.
    Microphone,
    /// System audio loopback (macOS 13+ via ScreenCaptureKit).
    SystemOutput,
}

impl From<IpcAudioSource> for AudioSource {
    fn from(value: IpcAudioSource) -> Self {
        match value {
            IpcAudioSource::Microphone => AudioSource::Microphone,
            IpcAudioSource::SystemOutput => AudioSource::SystemOutput,
        }
    }
}

/// Options the frontend may pass when starting a streaming session.
/// All fields optional; defaults match `StreamingOptions::default()`.
#[derive(Debug, Default, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct StartStreamingOptions {
    /// Capture source. `None` ⇒ microphone, preserving Sprint 0
    /// behavior. `systemOutput` requires macOS 13+ with Screen
    /// Recording permission (Sprint 1 day 2/3).
    pub source: Option<IpcAudioSource>,
    /// ISO-639-1 language hint. `None` ⇒ auto-detect.
    pub language: Option<String>,
    /// Capture device id. `None` ⇒ system default microphone. Ignored
    /// when `source = systemOutput`.
    pub device_id: Option<String>,
    /// Audio chunk size in milliseconds. `None` ⇒ 5000.
    pub chunk_ms: Option<u32>,
    /// RMS threshold below which a chunk is reported as `Skipped`
    /// instead of being sent to the ASR. `None` ⇒ 0.02.
    pub silence_rms_threshold: Option<f32>,
    /// Enable speaker diarization for this session. `None`/`false` ⇒
    /// disabled (keeps Sprint 0 behaviour). When enabled the backend
    /// loads the speaker embedder named by `ECHO_EMBED_MODEL` (or the
    /// default path) and attaches an `OnlineDiarizer` to the pipeline,
    /// so every emitted `Chunk` event carries `speakerId` +
    /// `speakerSlot` and the meeting persists its speakers.
    pub diarize: Option<bool>,
    /// Override path to the speaker-embedder ONNX. `None` ⇒ use the
    /// path resolved at app start. Mostly useful for tests / power
    /// users; the UI does not surface it.
    pub embed_model_path: Option<PathBuf>,
    /// Disable the neural (Silero) VAD for this session and fall back
    /// to the energy-based RMS gate. `None`/`false` ⇒ neural VAD is
    /// used when the ONNX model is installed (default and recommended:
    /// fewer Whisper hallucinations on silent / noisy chunks). Set to
    /// `true` only as an escape hatch — e.g. for very soft speakers
    /// the model misclassifies as silence, or to reproduce pre-Silero
    /// behaviour for benchmarking.
    pub disable_neural_vad: Option<bool>,
}

/// Start a streaming transcription session. Events are pushed through
/// the supplied `Channel<TranscriptEvent>` until [`stop_streaming`] is
/// invoked or the capture stream ends. Persists to SQLite incrementally
/// via the [`MeetingRecorder`].
#[tauri::command]
#[specta::specta]
pub async fn start_streaming(
    state: State<'_, AppState>,
    options: Option<StartStreamingOptions>,
    on_event: Channel<TranscriptEvent>,
) -> Result<StreamingSessionId, IpcError> {
    let opts = options.unwrap_or_default();
    let streaming_options = StreamingOptions {
        language: opts.language,
        chunk_ms: opts.chunk_ms.unwrap_or(5_000),
        silence_rms_threshold: opts.silence_rms_threshold.unwrap_or(0.02),
    };
    let source: AudioSource = opts
        .source
        .map(AudioSource::from)
        .unwrap_or(AudioSource::Microphone);
    // System-output loopback identifies the target by display, not by
    // a named device; surface a warning if the frontend forgets.
    let device_id = match source {
        AudioSource::Microphone => opts.device_id,
        AudioSource::SystemOutput => {
            if opts.device_id.is_some() {
                tracing::warn!(
                    "deviceId ignored when source = systemOutput \
                     (ScreenCaptureKit selects the primary display)"
                );
            }
            None
        }
    };
    let spec = CaptureSpec {
        source,
        device_id,
        preferred_format: AudioFormat::WHISPER,
    };

    let transcriber = state.ensure_transcriber().await?;
    let mut pipeline =
        StreamingPipeline::new(state.capture.clone(), state.resampler.clone(), transcriber);
    if opts.diarize.unwrap_or(false) {
        let diarizer = state.build_diarizer(opts.embed_model_path)?;
        pipeline = pipeline.with_diarizer(diarizer);
    }
    if !opts.disable_neural_vad.unwrap_or(false) {
        // ensure_vad returns Ok(None) when the model file is missing
        // — log once at warn level and continue with the RMS gate.
        if let Some(template) = state.ensure_vad().await? {
            let session_vad: Box<dyn Vad> = Box::new(template.clone_for_new_session());
            pipeline = pipeline.with_vad(session_vad);
        }
    } else {
        tracing::info!("neural VAD disabled by request; using RMS gate");
    }

    let handle = pipeline
        .start_with_spec(spec, streaming_options)
        .await
        .map_err(|e| IpcError::new(ErrorCode::Audio, format!("failed to start streaming: {e}")))?;
    let session_id = handle.session_id();

    // Drain the event receiver in a background task. Each event is
    // first persisted (recorder.record) and then forwarded to the IPC
    // channel; if persistence fails, we log and keep the UI responsive
    // — losing a row is preferable to crashing the live transcript.
    //
    // When the task exits — whether via a Stopped/Failed terminal
    // event, an upstream channel close, or a frontend disconnect — we
    // remove our entry from the session registry so the HashMap doesn't
    // accumulate orphans. If the user then calls `stop_streaming` for
    // the same id, it correctly resolves to `Ok(false)` ("already
    // stopped") instead of trying to drive a half-dead pipeline.
    let handle_arc = Arc::new(AsyncMutex::new(handle));
    let drain_handle = handle_arc.clone();
    let recorder = state.recorder.clone();
    let sessions_for_drain = state.sessions_handle();
    let join = tokio::spawn(async move {
        loop {
            let mut guard = drain_handle.lock().await;
            let evt = guard.next_event().await;
            drop(guard);
            match evt {
                Some(evt) => {
                    if let Err(e) = recorder.record(&evt).await {
                        tracing::warn!(error = %e, %session_id, "recorder.record failed");
                    }
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
        // Self-cleanup: the session is over (terminal event, channel
        // close, or frontend gone). The `stop_streaming` path already
        // removes its own entry — this branch covers every other exit.
        if let Ok(mut map) = sessions_for_drain.lock() {
            map.remove(&session_id);
        }
    });

    state
        .sessions
        .lock()
        .map_err(|e| {
            IpcError::new(
                ErrorCode::SessionConflict,
                format!("session map poisoned: {e}"),
            )
        })?
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
#[specta::specta]
pub async fn stop_streaming(
    state: State<'_, AppState>,
    session_id: StreamingSessionId,
) -> Result<bool, IpcError> {
    let entry = state
        .sessions
        .lock()
        .map_err(|e| {
            IpcError::new(
                ErrorCode::SessionConflict,
                format!("session map poisoned: {e}"),
            )
        })?
        .remove(&session_id);
    let Some(entry) = entry else {
        return Ok(false);
    };
    {
        let mut guard = entry.handle.lock().await;
        guard.stop().await.map_err(|e| {
            IpcError::new(ErrorCode::Audio, format!("failed to stop pipeline: {e}"))
        })?;
    }
    let _ = entry.join.await;
    Ok(true)
}

// ---------------------------------------------------------------------------
// Meetings (Sprint 0 day 8)
// ---------------------------------------------------------------------------

/// List meetings, newest first.
#[tauri::command]
#[specta::specta]
pub async fn list_meetings(
    state: State<'_, AppState>,
    limit: Option<u32>,
) -> Result<Vec<MeetingSummary>, IpcError> {
    state
        .store
        .list(limit.unwrap_or(0))
        .await
        .map_err(|e| IpcError::storage(format!("list meetings: {e}")))
}

/// Fetch a single meeting (header + segments). Returns `null` when
/// the id does not exist.
#[tauri::command]
#[specta::specta]
pub async fn get_meeting(
    state: State<'_, AppState>,
    id: MeetingId,
) -> Result<Option<Meeting>, IpcError> {
    state
        .store
        .get(id)
        .await
        .map_err(|e| IpcError::storage(format!("get meeting: {e}")))
}

/// Delete a meeting and its segments. Returns `true` when the row
/// existed and was removed.
#[tauri::command]
#[specta::specta]
pub async fn delete_meeting(state: State<'_, AppState>, id: MeetingId) -> Result<bool, IpcError> {
    state
        .store
        .delete(id)
        .await
        .map_err(|e| IpcError::storage(format!("delete meeting: {e}")))
}

/// Full-text search over segment text. Returns one hit per meeting,
/// ordered by FTS5 BM25 rank (best match first). Empty / whitespace-
/// only queries return an empty vec — the frontend wires this to
/// `onChange` debounced, so the initial empty state never errors.
///
/// `limit` defaults to 20 (a comfortable sidebar page); `0` means no
/// cap. The query is sanitised inside the storage layer, so the
/// frontend can pass raw user input without worrying about FTS5
/// syntax characters.
#[tauri::command]
#[specta::specta]
pub async fn search_meetings(
    state: State<'_, AppState>,
    query: String,
    limit: Option<u32>,
) -> Result<Vec<MeetingSearchHit>, IpcError> {
    state
        .store
        .search(&query, limit.unwrap_or(20))
        .await
        .map_err(|e| IpcError::storage(format!("search meetings: {e}")))
}

/// Rename a diarized speaker (or clear the label back to anonymous
/// by passing `null`/empty). Returns the freshly-loaded meeting so
/// the frontend can re-render speakers + segment chips from a single
/// source of truth without an extra `get_meeting` round-trip.
#[tauri::command]
#[specta::specta]
pub async fn rename_speaker(
    state: State<'_, AppState>,
    meeting_id: MeetingId,
    speaker_id: SpeakerId,
    label: Option<String>,
) -> Result<Meeting, IpcError> {
    state
        .rename_speaker
        .execute(meeting_id, speaker_id, label)
        .await
        .map_err(IpcError::from)?;
    // Re-fetch so the UI sees the canonical post-rename state
    // (including speakers with refreshed labels and segments).
    state
        .store
        .get(meeting_id)
        .await
        .map_err(|e| IpcError::storage(format!("reload meeting: {e}")))?
        .ok_or_else(|| {
            IpcError::not_found(format!("meeting {meeting_id} disappeared after rename"))
        })
}

// ---------------------------------------------------------------------------
// LLM summaries (Sprint 1 day 9)
// ---------------------------------------------------------------------------

/// Generate (or regenerate) the local-LLM summary for a meeting.
///
/// `template` selects the prompt template: `"general"` (default),
/// `"oneOnOne"`, `"sprintReview"`, `"interview"`, `"salesCall"`, or
/// `"lecture"`. Passing `null` or omitting the field defaults to
/// `"general"`.
#[tauri::command]
#[specta::specta]
pub async fn summarize_meeting(
    state: State<'_, AppState>,
    meeting_id: MeetingId,
    template: Option<String>,
) -> Result<Summary, IpcError> {
    let llm = state.ensure_llm().await?;
    let use_case = SummarizeMeeting::new(llm, state.store.clone());
    let tmpl = template.as_deref().unwrap_or("general");
    use_case
        .execute(meeting_id, tmpl)
        .await
        .map_err(IpcError::from)
}

/// Fetch the most recent summary for a meeting, or `null` when none
/// has been generated yet. The frontend uses this on `MeetingDetail`
/// mount so the panel can render either the existing summary or the
/// "Generate" affordance without a redundant generate call.
#[tauri::command]
#[specta::specta]
pub async fn get_summary(
    state: State<'_, AppState>,
    meeting_id: MeetingId,
) -> Result<Option<Summary>, IpcError> {
    state
        .store
        .get_summary(meeting_id)
        .await
        .map_err(|e| IpcError::storage(format!("get summary: {e}")))
}

// ---------------------------------------------------------------------------
// Chat with transcript (Sprint 1 day 10 — CU-05)
// ---------------------------------------------------------------------------

/// Run one chat turn against a meeting's transcript.
///
/// Each invocation streams the assistant's reply token-by-token through
/// `on_event` and resolves the IPC promise once the stream terminates
/// (with [`AskAboutMeetingEvent::Finished`] on success or
/// [`AskAboutMeetingEvent::Failed`] on a mid-decode error).
///
/// ## Lifecycle
///
/// 1. Pre-stream errors (meeting not found, empty question, model
///    not loaded) come back as `Err(String)` so the UI can show a
///    toast without parsing event variants.
/// 2. Once the stream is open, **every** terminal condition travels
///    as a final event on the channel: success → `Finished`,
///    mid-decode failure → `Failed`. The promise then resolves
///    `Ok(())`.
/// 3. Cancellation is automatic: when the React component unmounts
///    (or the user closes the chat panel), Tauri drops the
///    `Channel`. The next `on_event.send` returns `Err(_)`, this
///    function exits its drain loop, the `BoxStream` is dropped, the
///    underlying `mpsc::Receiver` closes and the blocking decoder
///    thread inside `LlamaCppChat` exits on the next
///    `tx.blocking_send` failure. No explicit `cancel_chat` command
///    is needed — symmetric with `start_streaming` was deliberately
///    not chosen because chat is bounded (one Q&A turn ≤ ~10s on
///    Qwen 14B, vs streaming sessions that can run hours).
///
/// ## Why we don't spawn a background task
///
/// Unlike `start_streaming`, which returns a session id and runs
/// indefinitely until `stop_streaming`, a chat turn is a single
/// bounded interaction. Draining the stream inline keeps the IPC
/// surface minimal (one command, one channel, one promise) and means
/// any error gets observed by the calling `await` instead of escaping
/// into a detached `tokio::spawn`.
#[tauri::command]
#[specta::specta]
pub async fn ask_about_meeting(
    state: State<'_, AppState>,
    meeting_id: MeetingId,
    history: Option<Vec<ChatMessage>>,
    question: String,
    on_event: Channel<AskAboutMeetingEvent>,
) -> Result<(), IpcError> {
    let chat = state.ensure_chat().await?;
    let use_case = AskAboutMeeting::new(chat, state.store.clone());

    let mut stream = use_case
        .execute(meeting_id, history.unwrap_or_default(), question)
        .await
        .map_err(IpcError::from)?;

    while let Some(event) = stream.next().await {
        if let Err(e) = on_event.send(event) {
            tracing::warn!(
                error = %e,
                %meeting_id,
                "ask_about_meeting channel send failed; aborting drain",
            );
            break;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Export meeting (CU-08)
// ---------------------------------------------------------------------------

/// Supported export formats.
#[derive(Debug, Clone, Copy, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub enum ExportFormat {
    Markdown,
    Txt,
}

/// Export a meeting (with optional summary) to a file at `dest_path`.
///
/// The frontend is responsible for showing the save-file dialog (via
/// `@tauri-apps/plugin-dialog`) and passing the chosen path here. This
/// command generates the formatted content and writes it atomically.
///
/// **Security:** `dest_path` is validated to be inside the user's home
/// directory and must not contain path-traversal components (`..`).
#[tauri::command]
#[specta::specta]
pub async fn export_meeting(
    state: State<'_, AppState>,
    meeting_id: MeetingId,
    format: ExportFormat,
    dest_path: String,
) -> Result<(), IpcError> {
    // ── Path validation ──────────────────────────────────────────
    let dest = PathBuf::from(&dest_path);
    if !dest.is_absolute() {
        return Err(IpcError::new(
            ErrorCode::InvalidInput,
            "export path must be absolute",
        ));
    }
    // Reject explicit traversal components before canonicalizing.
    if dest
        .components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return Err(IpcError::new(
            ErrorCode::InvalidInput,
            "export path must not contain '..' components",
        ));
    }
    let home =
        dirs::home_dir().ok_or_else(|| IpcError::internal("cannot determine home directory"))?;
    // Canonicalize the parent (the file itself may not exist yet).
    let parent = dest.parent().ok_or_else(|| {
        IpcError::new(
            ErrorCode::InvalidInput,
            "export path has no parent directory",
        )
    })?;
    let canonical_parent = parent
        .canonicalize()
        .map_err(|e| IpcError::storage(format!("invalid export directory: {e}")))?;
    if !canonical_parent.starts_with(&home) {
        return Err(IpcError::new(
            ErrorCode::InvalidInput,
            "export path must be within the home directory",
        ));
    }
    let safe_dest = canonical_parent
        .join(dest.file_name().ok_or_else(|| {
            IpcError::new(ErrorCode::InvalidInput, "export path has no filename")
        })?);

    // ── Generate + write ─────────────────────────────────────────
    let meeting = state
        .store
        .get(meeting_id)
        .await
        .map_err(|e| IpcError::storage(format!("get meeting: {e}")))?
        .ok_or_else(|| IpcError::not_found(format!("meeting {meeting_id} not found")))?;

    let summary = state
        .store
        .get_summary(meeting_id)
        .await
        .map_err(|e| IpcError::storage(format!("get summary: {e}")))?;

    let content = match format {
        ExportFormat::Markdown => render_markdown(&meeting, summary.as_ref()),
        ExportFormat::Txt => render_plain_text(&meeting, summary.as_ref()),
    };

    tokio::fs::write(&safe_dest, content.as_bytes())
        .await
        .map_err(|e| IpcError::storage(format!("write file: {e}")))?;

    Ok(())
}

fn format_ts(ms: u32) -> String {
    let total_s = ms / 1000;
    let m = total_s / 60;
    let s = total_s % 60;
    format!("{m:02}:{s:02}")
}

fn speaker_label(meeting: &Meeting, speaker_id: Option<&echo_domain::SpeakerId>) -> Option<String> {
    let sid = speaker_id?;
    meeting.speakers.iter().find(|sp| &sp.id == sid).map(|sp| {
        sp.label
            .clone()
            .unwrap_or_else(|| format!("Speaker {}", sp.slot + 1))
    })
}

fn render_summary_body_md(summary: &Summary) -> String {
    use echo_domain::SummaryContent;
    let mut out = String::new();

    match &summary.content {
        SummaryContent::General {
            summary: s,
            key_points,
            decisions,
            action_items,
            open_questions,
        } => {
            out.push_str(s);
            out.push_str("\n\n");
            if !key_points.is_empty() {
                out.push_str("### Key points\n\n");
                for p in key_points {
                    out.push_str(&format!("- {p}\n"));
                }
                out.push('\n');
            }
            if !decisions.is_empty() {
                out.push_str("### Decisions\n\n");
                for d in decisions {
                    out.push_str(&format!("- {d}\n"));
                }
                out.push('\n');
            }
            if !action_items.is_empty() {
                out.push_str("### Action items\n\n");
                for ai in action_items {
                    let mut line = format!("- [ ] {}", ai.task);
                    if let Some(o) = &ai.owner {
                        line.push_str(&format!(" — *{o}*"));
                    }
                    if let Some(d) = &ai.due {
                        line.push_str(&format!(" (due: {d})"));
                    }
                    out.push_str(&line);
                    out.push('\n');
                }
                out.push('\n');
            }
            if !open_questions.is_empty() {
                out.push_str("### Open questions\n\n");
                for q in open_questions {
                    out.push_str(&format!("- {q}\n"));
                }
                out.push('\n');
            }
        }
        SummaryContent::OneOnOne {
            summary: s,
            wins,
            blockers,
            growth_feedback,
            next_steps,
            follow_up_topics,
        } => {
            out.push_str(s);
            out.push_str("\n\n");
            md_list(&mut out, "### Wins", wins);
            md_list(&mut out, "### Blockers", blockers);
            md_list(&mut out, "### Growth feedback", growth_feedback);
            if !next_steps.is_empty() {
                out.push_str("### Next steps\n\n");
                for ai in next_steps {
                    let mut line = format!("- [ ] {}", ai.task);
                    if let Some(o) = &ai.owner {
                        line.push_str(&format!(" — *{o}*"));
                    }
                    out.push_str(&line);
                    out.push('\n');
                }
                out.push('\n');
            }
            md_list(&mut out, "### Follow-up topics", follow_up_topics);
        }
        SummaryContent::SprintReview {
            summary: s,
            completed_items,
            carry_over,
            risks,
            next_sprint_priorities,
        } => {
            out.push_str(s);
            out.push_str("\n\n");
            md_list(&mut out, "### Completed", completed_items);
            md_list(&mut out, "### Carry-over", carry_over);
            md_list(&mut out, "### Risks", risks);
            md_list(
                &mut out,
                "### Next sprint priorities",
                next_sprint_priorities,
            );
        }
        SummaryContent::Interview {
            summary: s,
            quotes,
            themes,
            pain_points,
            opportunities,
        } => {
            out.push_str(s);
            out.push_str("\n\n");
            if !quotes.is_empty() {
                out.push_str("### Quotes\n\n");
                for q in quotes {
                    out.push_str(&format!(
                        "> \"{}\"\n> — {}{}\n\n",
                        q.quote,
                        q.speaker,
                        q.context
                            .as_deref()
                            .map(|c| format!(" ({c})"))
                            .unwrap_or_default()
                    ));
                }
            }
            md_list(&mut out, "### Themes", themes);
            md_list(&mut out, "### Pain points", pain_points);
            md_list(&mut out, "### Opportunities", opportunities);
        }
        SummaryContent::SalesCall {
            summary: s,
            customer_context,
            pain_points,
            interest_signals,
            objections,
            next_steps,
            deal_stage_indicator,
        } => {
            out.push_str(s);
            out.push_str("\n\n");
            if let Some(ctx) = customer_context {
                out.push_str(&format!("**Customer context:** {ctx}\n\n"));
            }
            md_list(&mut out, "### Pain points", pain_points);
            md_list(&mut out, "### Interest signals", interest_signals);
            md_list(&mut out, "### Objections", objections);
            if !next_steps.is_empty() {
                out.push_str("### Next steps\n\n");
                for ai in next_steps {
                    out.push_str(&format!("- [ ] {}\n", ai.task));
                }
                out.push('\n');
            }
            if let Some(stage) = deal_stage_indicator {
                out.push_str(&format!("**Deal stage:** {stage}\n\n"));
            }
        }
        SummaryContent::Lecture {
            summary: s,
            concepts_covered,
            definitions,
            examples,
            homework_or_next,
        } => {
            out.push_str(s);
            out.push_str("\n\n");
            md_list(&mut out, "### Concepts covered", concepts_covered);
            if !definitions.is_empty() {
                out.push_str("### Definitions\n\n");
                for d in definitions {
                    out.push_str(&format!("- **{}**: {}\n", d.term, d.definition));
                }
                out.push('\n');
            }
            md_list(&mut out, "### Examples", examples);
            md_list(&mut out, "### Homework / next", homework_or_next);
        }
        SummaryContent::FreeText { text } => {
            out.push_str(text);
            out.push_str("\n\n");
        }
        _ => {}
    }
    out
}

fn md_list(out: &mut String, heading: &str, items: &[String]) {
    if items.is_empty() {
        return;
    }
    out.push_str(heading);
    out.push_str("\n\n");
    for item in items {
        out.push_str(&format!("- {item}\n"));
    }
    out.push('\n');
}

fn render_markdown(meeting: &Meeting, summary: Option<&Summary>) -> String {
    let m = &meeting.summary;
    let date = &m.started_at;
    let dur_s = m.duration_ms / 1000;
    let dur = if dur_s < 60 {
        format!("{dur_s}s")
    } else {
        format!("{}m {:02}s", dur_s / 60, dur_s % 60)
    };

    let mut out = format!("# {}\n\n", m.title);
    out.push_str(&format!(
        "**Date:** {}  \n**Duration:** {}  \n**Language:** {}  \n**Segments:** {}\n\n",
        date,
        dur,
        m.language.as_deref().unwrap_or("?"),
        m.segment_count,
    ));

    if !meeting.speakers.is_empty() {
        out.push_str("**Participants:** ");
        let names: Vec<String> = meeting
            .speakers
            .iter()
            .map(|sp| {
                sp.label
                    .clone()
                    .unwrap_or_else(|| format!("Speaker {}", sp.slot + 1))
            })
            .collect();
        out.push_str(&names.join(", "));
        out.push_str("\n\n");
    }

    out.push_str("---\n\n");

    if let Some(s) = summary {
        out.push_str("## Summary\n\n");
        out.push_str(&render_summary_body_md(s));
        out.push_str("---\n\n");
    }

    out.push_str("## Transcript\n\n");
    for seg in &meeting.segments {
        let ts = format_ts(seg.start_ms);
        let label = speaker_label(meeting, seg.speaker_id.as_ref());
        let text = seg.text.trim();
        if text.is_empty() {
            continue;
        }
        match label {
            Some(name) => out.push_str(&format!("**[{ts}] {name}:** {text}\n\n")),
            None => out.push_str(&format!("**[{ts}]** {text}\n\n")),
        }
    }

    out
}

fn render_plain_text(meeting: &Meeting, summary: Option<&Summary>) -> String {
    let m = &meeting.summary;
    let dur_s = m.duration_ms / 1000;
    let dur = if dur_s < 60 {
        format!("{dur_s}s")
    } else {
        format!("{}m {:02}s", dur_s / 60, dur_s % 60)
    };

    let mut out = format!("{}\n{}\n\n", m.title, "=".repeat(m.title.len()));
    out.push_str(&format!(
        "Date:     {}\nDuration: {}\nLanguage: {}\nSegments: {}\n",
        m.started_at,
        dur,
        m.language.as_deref().unwrap_or("?"),
        m.segment_count,
    ));

    if !meeting.speakers.is_empty() {
        let names: Vec<String> = meeting
            .speakers
            .iter()
            .map(|sp| {
                sp.label
                    .clone()
                    .unwrap_or_else(|| format!("Speaker {}", sp.slot + 1))
            })
            .collect();
        out.push_str(&format!("Participants: {}\n", names.join(", ")));
    }

    if let Some(s) = summary {
        out.push_str("\n\nSUMMARY\n-------\n\n");
        out.push_str(&render_summary_body_txt(s));
    }

    out.push_str("\n\nTRANSCRIPT\n----------\n\n");
    for seg in &meeting.segments {
        let ts = format_ts(seg.start_ms);
        let label = speaker_label(meeting, seg.speaker_id.as_ref());
        let text = seg.text.trim();
        if text.is_empty() {
            continue;
        }
        match label {
            Some(name) => out.push_str(&format!("[{ts}] {name}: {text}\n")),
            None => out.push_str(&format!("[{ts}] {text}\n")),
        }
    }

    out
}

fn render_summary_body_txt(summary: &Summary) -> String {
    use echo_domain::SummaryContent;
    let mut out = String::new();

    match &summary.content {
        SummaryContent::General {
            summary: s,
            key_points,
            decisions,
            action_items,
            open_questions,
        } => {
            out.push_str(s);
            out.push_str("\n\n");
            txt_list(&mut out, "KEY POINTS", key_points);
            txt_list(&mut out, "DECISIONS", decisions);
            if !action_items.is_empty() {
                out.push_str("ACTION ITEMS\n");
                for ai in action_items {
                    let mut line = format!("  - {}", ai.task);
                    if let Some(o) = &ai.owner {
                        line.push_str(&format!(" ({o})"));
                    }
                    if let Some(d) = &ai.due {
                        line.push_str(&format!(" [due: {d}]"));
                    }
                    out.push_str(&line);
                    out.push('\n');
                }
                out.push('\n');
            }
            txt_list(&mut out, "OPEN QUESTIONS", open_questions);
        }
        _ => {
            let md = render_summary_body_md(summary);
            for line in md.lines() {
                let stripped = line
                    .trim_start_matches('#')
                    .trim_start_matches("**")
                    .trim_end_matches("**")
                    .trim_start_matches("- [ ] ")
                    .trim_start_matches("- ")
                    .trim_start_matches("> ");
                out.push_str(stripped.trim());
                out.push('\n');
            }
        }
    }
    out
}

fn txt_list(out: &mut String, heading: &str, items: &[String]) {
    if items.is_empty() {
        return;
    }
    out.push_str(heading);
    out.push('\n');
    for item in items {
        out.push_str(&format!("  - {item}\n"));
    }
    out.push('\n');
}

// ---------------------------------------------------------------------------
// Model management
// ---------------------------------------------------------------------------

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
