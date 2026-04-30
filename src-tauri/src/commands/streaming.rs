//! Streaming transcription commands (`start_streaming`, `stop_streaming`).

use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use serde::Deserialize;
use tauri::ipc::Channel;
use tauri::State;
use tokio::sync::Mutex as AsyncMutex;

use crate::ipc_error::{ErrorCode, IpcError};

use echo_app::StreamingPipeline;
use echo_domain::{
    AudioFormat, AudioSource, CaptureSpec, StreamingOptions, StreamingSessionId, TranscriptEvent,
    Vad,
};

use super::{AppState, SessionEntry};

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
    let paused_flag = handle.paused_flag();

    let handle_arc = Arc::new(AsyncMutex::new(handle));
    let drain_handle = handle_arc.clone();
    let recorder = state.recorder.clone();
    let sessions_for_drain = state.sessions_handle();
    let join = tokio::spawn(async move {
        let mut saw_terminal = false;
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
                    if terminal {
                        saw_terminal = true;
                    }
                    if let Err(e) = on_event.send(evt) {
                        tracing::warn!(error = %e, %session_id, "frontend channel send failed");
                        break;
                    }
                    if saw_terminal {
                        break;
                    }
                }
                None => break,
            }
        }
        // If the drain loop exited without a terminal event (e.g.
        // frontend disconnected, channel returned None), synthesise a
        // Failed event so the MeetingRecorder cleans up its session
        // stats and finalises the meeting row.
        if !saw_terminal {
            tracing::warn!(%session_id, "drain loop exited without terminal event; synthesising Failed");
            let synth = TranscriptEvent::Failed {
                session_id,
                message: "drain exited without terminal event".into(),
            };
            if let Err(e) = recorder.record(&synth).await {
                tracing::warn!(error = %e, %session_id, "recorder.record(synth Failed) failed");
            }
        }
        if let Ok(mut map) = sessions_for_drain.lock() {
            map.remove(&session_id);
        }
    });

    let insert_result = state.sessions.lock().map_err(|e| {
        // Abort the spawned drain task so it doesn't run detached
        // with a live microphone and no way to stop it.
        join.abort();
        IpcError::new(
            ErrorCode::SessionConflict,
            format!("session map poisoned: {e}"),
        )
    });
    insert_result?.insert(
        session_id,
        SessionEntry {
            join,
            handle: handle_arc,
            paused: paused_flag,
            _keep_awake: keepawake::Builder::default()
                .idle(true)
                .reason("Recording audio session")
                .app_name("EchoNote")
                .app_reverse_domain("com.echonote.app")
                .create()
                .map_err(|e| tracing::warn!("keep-awake unavailable: {e}"))
                .ok(),
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
    // If paused the drain loop is blocked on `next_event` while
    // holding the handle lock — unpause first so the loop spins,
    // releases the lock, and sees the stop signal.
    entry.paused.store(false, Ordering::Release);
    {
        let mut guard = entry.handle.lock().await;
        guard.stop().await.map_err(|e| {
            IpcError::new(ErrorCode::Audio, format!("failed to stop pipeline: {e}"))
        })?;
    }
    let _ = entry.join.await;
    Ok(true)
}

/// Pause a running streaming session. Audio capture keeps running but
/// frames are discarded. Returns `Ok(false)` when the session id is
/// unknown or was already paused.
#[tauri::command]
#[specta::specta]
pub async fn pause_streaming(
    state: State<'_, AppState>,
    session_id: StreamingSessionId,
) -> Result<bool, IpcError> {
    let flag = {
        let map = state.sessions.lock().map_err(|e| {
            IpcError::new(
                ErrorCode::SessionConflict,
                format!("session map poisoned: {e}"),
            )
        })?;
        let Some(entry) = map.get(&session_id) else {
            return Ok(false);
        };
        Arc::clone(&entry.paused)
    };
    // compare_exchange avoids a redundant Paused event when the
    // pipeline was already paused.
    let swapped = flag
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_ok();
    Ok(swapped)
}

/// Resume a paused streaming session. Returns `Ok(false)` when the
/// session id is unknown or was not paused.
#[tauri::command]
#[specta::specta]
pub async fn resume_streaming(
    state: State<'_, AppState>,
    session_id: StreamingSessionId,
) -> Result<bool, IpcError> {
    let flag = {
        let map = state.sessions.lock().map_err(|e| {
            IpcError::new(
                ErrorCode::SessionConflict,
                format!("session map poisoned: {e}"),
            )
        })?;
        let Some(entry) = map.get(&session_id) else {
            return Ok(false);
        };
        Arc::clone(&entry.paused)
    };
    let swapped = flag
        .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
        .is_ok();
    Ok(swapped)
}

/// Look up the database `MeetingId` associated with a streaming session.
/// The streaming session id and the meeting id are distinct UUIDs — the
/// recorder creates a fresh meeting row on `Started` and keeps the mapping
/// internally. This command exposes that mapping so the frontend can
/// address the meeting (e.g. to add notes) while the session is running.
#[tauri::command]
#[specta::specta]
pub async fn get_meeting_id(
    state: State<'_, AppState>,
    session_id: StreamingSessionId,
) -> Result<Option<echo_domain::MeetingId>, IpcError> {
    Ok(state.recorder.meeting_id_for_session(session_id).await)
}
