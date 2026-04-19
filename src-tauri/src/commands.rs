//! Tauri IPC commands exposed to the frontend.
//!
//! Each command here mirrors a typed contract in `src/lib/ipc.ts`. When
//! the surface grows beyond a handful, switch to `tauri-specta` code
//! generation — see ADR note in `docs/adr/0002-rust-plus-react-stack.md`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use tauri::ipc::Channel;
use tauri::State;
use time::{format_description::well_known::Rfc3339, OffsetDateTime};
use tokio::sync::Mutex as AsyncMutex;
use tokio::task::JoinHandle;

use echo_app::{
    MeetingRecorder, RenameSpeaker, RenameSpeakerError, StreamingHandle, StreamingPipeline,
};
use echo_asr::WhisperCppTranscriber;
use echo_audio::{RoutingAudioCapture, RubatoResamplerAdapter};
use echo_diarize::{Eres2NetEmbedder, OnlineDiarizer};
use echo_domain::{
    AudioCapture, AudioFormat, AudioSource, CaptureSpec, Diarizer, Meeting, MeetingId,
    MeetingSearchHit, MeetingStore, MeetingSummary, Resampler, SpeakerId, StreamingOptions,
    StreamingSessionId, Transcriber, TranscriptEvent,
};
use echo_storage::SqliteMeetingStore;

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
    sessions: Mutex<HashMap<StreamingSessionId, SessionEntry>>,
}

struct SessionEntry {
    join: JoinHandle<()>,
    handle: Arc<AsyncMutex<StreamingHandle>>,
}

impl AppState {
    /// Build the shared state. Async because opening the SQLite database
    /// runs migrations, which is I/O. The transcriber is *not* loaded
    /// eagerly — the first `start_streaming` call pays that cost.
    pub async fn initialize() -> Result<Self, String> {
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

        let db_path = resolve_db_path();
        tracing::info!(
            asr_model = %model_path.display(),
            embed_model = %embed_model_path.display(),
            db_path = %db_path.display(),
            "echo-shell paths resolved"
        );
        let store = SqliteMeetingStore::open(&db_path)
            .await
            .map_err(|e| format!("open meeting store at {}: {e}", db_path.display()))?;
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
            sessions: Mutex::new(HashMap::new()),
        })
    }

    /// Build a fresh diarizer for the upcoming session. Each session
    /// gets its own `OnlineDiarizer` because the trait is stateful
    /// (the cluster centroids belong to a single recording). The
    /// embedder is reloaded from disk per session — the ONNX load is
    /// O(100ms), well under the user-perceptible threshold for a
    /// "Start" click — but cached `tract` graphs could be wired in if
    /// this ever shows up in profiling.
    fn build_diarizer(&self, override_path: Option<PathBuf>) -> Result<Box<dyn Diarizer>, String> {
        let path = override_path.unwrap_or_else(|| self.embed_model_path.clone());
        if !path.exists() {
            return Err(format!(
                "speaker embedder not found at {}. Run `scripts/download-models.sh embed` \
                 or set ECHO_EMBED_MODEL.",
                path.display()
            ));
        }
        let started = std::time::Instant::now();
        let embedder = Eres2NetEmbedder::new(&path)
            .map_err(|e| format!("load speaker embedder at {}: {e}", path.display()))?;
        tracing::info!(
            model = %path.display(),
            load_ms = started.elapsed().as_millis() as u64,
            "speaker embedder ready"
        );
        Ok(Box::new(OnlineDiarizer::with_defaults(Box::new(embedder))))
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

/// Resolve the SQLite database path. Honours `ECHO_DB_PATH` for tests
/// and falls back to `./echonote.db` (next to the binary). A real
/// installer would point this at the OS-appropriate app-data dir;
/// that's deferred until Sprint 1 when the installer lands.
fn resolve_db_path() -> PathBuf {
    resolve_asset_path(std::env::var("ECHO_DB_PATH").ok(), "echonote.db")
}

/// Pick the ASR model to load by default, in priority order:
/// the largest installed multilingual ggml model first (so non-English
/// users get a working transcript without env overrides), then
/// English-only fallbacks for backwards compatibility with Sprint 0
/// setups. Resolution against the workspace happens later in
/// [`resolve_asset_path`]; here we only pick a *relative* path that
/// the resolver checks for existence.
fn preferred_asr_model() -> &'static str {
    const CANDIDATES: &[&str] = &[
        "models/asr/ggml-large-v3.bin",
        "models/asr/ggml-medium.bin",
        "models/asr/ggml-small.bin",
        "models/asr/ggml-base.bin",
        "models/asr/ggml-tiny.bin",
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
    // Nothing installed yet — default to multilingual base so the
    // error message points the user at the right download command.
    "models/asr/ggml-base.bin"
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
#[derive(Debug, Clone, Copy, Deserialize)]
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
#[derive(Debug, Default, Deserialize)]
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
    /// instead of being sent to the ASR. `None` ⇒ 0.005.
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
}

/// Start a streaming transcription session. Events are pushed through
/// the supplied `Channel<TranscriptEvent>` until [`stop_streaming`] is
/// invoked or the capture stream ends. Persists to SQLite incrementally
/// via the [`MeetingRecorder`].
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

    let handle = pipeline
        .start_with_spec(spec, streaming_options)
        .await
        .map_err(|e| format!("failed to start streaming: {e}"))?;
    let session_id = handle.session_id();

    // Drain the event receiver in a background task. Each event is
    // first persisted (recorder.record) and then forwarded to the IPC
    // channel; if persistence fails, we log and keep the UI responsive
    // — losing a row is preferable to crashing the live transcript.
    let handle_arc = Arc::new(AsyncMutex::new(handle));
    let drain_handle = handle_arc.clone();
    let recorder = state.recorder.clone();
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

// ---------------------------------------------------------------------------
// Meetings (Sprint 0 day 8)
// ---------------------------------------------------------------------------

/// List meetings, newest first.
#[tauri::command]
pub async fn list_meetings(
    state: State<'_, AppState>,
    limit: Option<u32>,
) -> Result<Vec<MeetingSummary>, String> {
    state
        .store
        .list(limit.unwrap_or(0))
        .await
        .map_err(|e| format!("list meetings: {e}"))
}

/// Fetch a single meeting (header + segments). Returns `null` when
/// the id does not exist.
#[tauri::command]
pub async fn get_meeting(
    state: State<'_, AppState>,
    id: MeetingId,
) -> Result<Option<Meeting>, String> {
    state
        .store
        .get(id)
        .await
        .map_err(|e| format!("get meeting: {e}"))
}

/// Delete a meeting and its segments. Returns `true` when the row
/// existed and was removed.
#[tauri::command]
pub async fn delete_meeting(state: State<'_, AppState>, id: MeetingId) -> Result<bool, String> {
    state
        .store
        .delete(id)
        .await
        .map_err(|e| format!("delete meeting: {e}"))
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
pub async fn search_meetings(
    state: State<'_, AppState>,
    query: String,
    limit: Option<u32>,
) -> Result<Vec<MeetingSearchHit>, String> {
    state
        .store
        .search(&query, limit.unwrap_or(20))
        .await
        .map_err(|e| format!("search meetings: {e}"))
}

/// Rename a diarized speaker (or clear the label back to anonymous
/// by passing `null`/empty). Returns the freshly-loaded meeting so
/// the frontend can re-render speakers + segment chips from a single
/// source of truth without an extra `get_meeting` round-trip.
#[tauri::command]
pub async fn rename_speaker(
    state: State<'_, AppState>,
    meeting_id: MeetingId,
    speaker_id: SpeakerId,
    label: Option<String>,
) -> Result<Meeting, String> {
    state
        .rename_speaker
        .execute(meeting_id, speaker_id, label)
        .await
        .map_err(|e| match e {
            RenameSpeakerError::NotFound { .. } => format!("not found: {e}"),
            RenameSpeakerError::Invalid(msg) => format!("invalid: {msg}"),
            RenameSpeakerError::Storage(err) => format!("storage: {err}"),
        })?;
    // Re-fetch so the UI sees the canonical post-rename state
    // (including speakers with refreshed labels and segments).
    state
        .store
        .get(meeting_id)
        .await
        .map_err(|e| format!("reload meeting: {e}"))?
        .ok_or_else(|| format!("meeting {meeting_id} disappeared after rename"))
}
