//! `StreamingPipeline` — real-time transcription orchestrator.
//!
//! Wires three ports together:
//!
//! ```text
//!   AudioCapture  →  Resampler  →  Transcriber
//!        │               │              │
//!        ▼               ▼              ▼
//!     frames        16 kHz mono     TranscriptEvent
//!                                   (mpsc channel)
//! ```
//!
//! The pipeline buffers raw frames at the device sample rate, flushes a
//! "chunk" every [`StreamingOptions::chunk_ms`] of audio, runs the
//! buffered chunk through the resampler and the transcriber, then emits
//! a [`TranscriptEvent::Chunk`] (or [`TranscriptEvent::Skipped`] when
//! the chunk is below the silence floor).
//!
//! Backpressure: a single ASR call is in flight at a time. If the
//! transcriber is slower than real time the next chunk waits — the UI
//! sees this as growing latency rather than dropped audio. A drop
//! policy can be added in Sprint 1 once metrics are wired.

mod sample_pool;

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

use echo_domain::{
    AudioCapture, AudioFormat, CaptureSpec, Diarizer, DomainError, Resampler, Sample, SpeakerId,
    StreamingOptions, StreamingSessionId, TranscribeOptions, Transcriber, TranscriptEvent, Vad,
    VoiceState,
};

/// Capacity of the event channel. ~10 minutes of headroom for a
/// 5-second chunk cadence; if consumers are slower than that we have a
/// bigger problem.
const EVENT_CHANNEL_CAPACITY: usize = 128;

/// Errors specific to the streaming pipeline.
#[derive(Debug, thiserror::Error)]
pub enum StreamingError {
    /// Domain port returned an error during setup or runtime.
    #[error(transparent)]
    Domain(#[from] DomainError),
    /// Tried to operate on a handle whose background task is gone.
    #[error("streaming task is no longer running")]
    TaskGone,
}

/// Bundle of port implementations required to build a pipeline.
///
/// `Clone` is intentionally omitted: the optional [`Diarizer`] is a
/// `Box<dyn Diarizer>` that owns mutable state (cluster centroids, the
/// embedder's LSTM, …) and cannot be shared across multiple sessions
/// safely. Build one pipeline per session — the inner port handles
/// (`Arc<dyn …>`) can still be reused across constructions.
pub struct StreamingPipeline {
    capture: Arc<dyn AudioCapture>,
    resampler: Arc<dyn Resampler>,
    transcriber: Arc<dyn Transcriber>,
    /// Optional diarizer. When set, every transcribed chunk is also
    /// fed through `Diarizer::assign` and the resulting `SpeakerId`
    /// (plus its arrival-order slot) is attached to the emitted
    /// [`TranscriptEvent::Chunk`]. Skipped chunks bypass diarization.
    diarizer: Option<Box<dyn Diarizer>>,
    /// Optional neural Voice Activity Detector. When set, the pipeline
    /// pushes every resampled chunk through it **in chronological
    /// order** (the LSTM state is sequence-sensitive) and only sends
    /// chunks classified as `Voiced` to the transcriber. Chunks the
    /// VAD calls `Silence` are emitted as `TranscriptEvent::Skipped`,
    /// same as the energy-based RMS gate.
    ///
    /// When this is set the cheap RMS gate is bypassed — Silero is
    /// strictly more discriminating (rejects pure tones, fans, music
    /// and keyboard noise that the RMS gate lets through) and the
    /// pipeline must feed it every chunk for its temporal model to
    /// stay coherent.
    vad: Option<Box<dyn Vad>>,
}

impl StreamingPipeline {
    /// Wire the pipeline with concrete adapters.
    pub fn new(
        capture: Arc<dyn AudioCapture>,
        resampler: Arc<dyn Resampler>,
        transcriber: Arc<dyn Transcriber>,
    ) -> Self {
        Self {
            capture,
            resampler,
            transcriber,
            diarizer: None,
            vad: None,
        }
    }

    /// Attach a diarizer. Call once before `start*`. The diarizer must
    /// agree with the resampler on sample rate (16 kHz mono); the
    /// pipeline asserts this at runtime to fail loudly on misconfig.
    #[must_use]
    pub fn with_diarizer(mut self, diarizer: Box<dyn Diarizer>) -> Self {
        self.diarizer = Some(diarizer);
        self
    }

    /// Attach a neural VAD (typically `SileroVad`) to gate Whisper
    /// behind voice activity detection. See the field doc on
    /// [`StreamingPipeline::vad`] for the full semantics.
    ///
    /// Call once before `start*`. The VAD must run at 16 kHz mono
    /// (the pipeline's canonical Whisper format); otherwise the
    /// session will fail to start with a clear error message.
    #[must_use]
    pub fn with_vad(mut self, vad: Box<dyn Vad>) -> Self {
        self.vad = Some(vad);
        self
    }

    /// Start capture + transcription for the default microphone.
    pub async fn start(self, options: StreamingOptions) -> Result<StreamingHandle, StreamingError> {
        self.start_with_spec(CaptureSpec::default_microphone(), options)
            .await
    }

    /// Start capture + transcription for a custom [`CaptureSpec`].
    pub async fn start_with_spec(
        self,
        spec: CaptureSpec,
        options: StreamingOptions,
    ) -> Result<StreamingHandle, StreamingError> {
        let session_id = StreamingSessionId::new();
        let (event_tx, event_rx) = mpsc::channel::<TranscriptEvent>(EVENT_CHANNEL_CAPACITY);
        let (stop_tx, stop_rx) = oneshot::channel::<()>();

        if let Some(d) = &self.diarizer {
            // The pipeline always feeds the diarizer the resampled
            // 16 kHz mono buffer, so any embedder expecting another
            // rate is misconfigured. Surface it now instead of
            // returning silently-bad embeddings later.
            if d.sample_rate_hz() != echo_audio_whisper_rate() {
                return Err(StreamingError::Domain(DomainError::Invariant(format!(
                    "diarizer sample rate {} does not match the pipeline's 16 kHz mono",
                    d.sample_rate_hz()
                ))));
            }
        }
        if let Some(v) = &self.vad {
            // Same invariant for the VAD: it sees the resampled buffer.
            if v.sample_rate_hz() != echo_audio_whisper_rate() {
                return Err(StreamingError::Domain(DomainError::Invariant(format!(
                    "VAD sample rate {} does not match the pipeline's 16 kHz mono",
                    v.sample_rate_hz()
                ))));
            }
        }

        let join = tokio::spawn(run_pipeline(
            self.capture,
            self.resampler,
            self.transcriber,
            self.diarizer,
            self.vad,
            session_id,
            spec,
            options,
            event_tx,
            stop_rx,
        ));

        Ok(StreamingHandle {
            session_id,
            events: event_rx,
            stop_tx: Some(stop_tx),
            join: Some(join),
        })
    }
}

/// Whisper's canonical sample rate. Inlined here to avoid pulling
/// `echo-audio` into the application layer just for the constant.
const fn echo_audio_whisper_rate() -> u32 {
    16_000
}

/// Handle returned by [`StreamingPipeline::start`]. Drops the underlying
/// task on `Drop` (best-effort cancel via the stop channel).
pub struct StreamingHandle {
    session_id: StreamingSessionId,
    events: mpsc::Receiver<TranscriptEvent>,
    stop_tx: Option<oneshot::Sender<()>>,
    join: Option<JoinHandle<()>>,
}

impl StreamingHandle {
    /// Session id assigned when the pipeline started.
    #[must_use]
    pub fn session_id(&self) -> StreamingSessionId {
        self.session_id
    }

    /// Receive the next event, or `None` when the task has finished.
    pub async fn next_event(&mut self) -> Option<TranscriptEvent> {
        self.events.recv().await
    }

    /// Signal the background task to stop and wait for it to drain.
    /// Idempotent.
    pub async fn stop(&mut self) -> Result<(), StreamingError> {
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }
        if let Some(join) = self.join.take() {
            join.await.map_err(|e| {
                error!(error = %e, "streaming task panicked");
                StreamingError::TaskGone
            })?;
        }
        Ok(())
    }
}

impl Drop for StreamingHandle {
    fn drop(&mut self) {
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_pipeline(
    capture: Arc<dyn AudioCapture>,
    resampler: Arc<dyn Resampler>,
    transcriber: Arc<dyn Transcriber>,
    mut diarizer: Option<Box<dyn Diarizer>>,
    mut vad: Option<Box<dyn Vad>>,
    session_id: StreamingSessionId,
    spec: CaptureSpec,
    options: StreamingOptions,
    events: mpsc::Sender<TranscriptEvent>,
    mut stop_rx: oneshot::Receiver<()>,
) {
    info!(%session_id, vad_enabled = vad.is_some(), "streaming pipeline starting");

    // Reset VAD state at session start so the LSTM/hysteresis don't
    // carry over from a previous session reusing the same template.
    if let Some(v) = vad.as_deref_mut() {
        v.reset();
    }

    let mut stream = match capture.start(spec).await {
        Ok(s) => s,
        Err(e) => {
            error!(error = %e, "capture.start failed");
            let _ = events
                .send(TranscriptEvent::Failed {
                    session_id,
                    message: format!("capture failed to start: {e}"),
                })
                .await;
            return;
        }
    };

    let format = stream.format();
    let _ = events
        .send(TranscriptEvent::Started {
            session_id,
            input_format: format,
        })
        .await;

    let chunk_samples = chunk_size_in_samples(format, options.chunk_ms);
    let transcribe_options = TranscribeOptions {
        language: options.language.clone(),
        ..Default::default()
    };

    let mut pool = sample_pool::SamplePool::new(chunk_samples);
    let mut buffer: Vec<Sample> = Vec::with_capacity(chunk_samples);
    let mut chunk_index: u32 = 0;
    let mut total_audio_ms: u32 = 0;
    let mut total_segments: u32 = 0;
    let pipeline_started = Instant::now();

    loop {
        tokio::select! {
            biased;
            _ = &mut stop_rx => {
                debug!(%session_id, "stop signal received");
                break;
            }
            frame = stream.next_frame() => {
                match frame {
                    Some(f) => buffer.extend_from_slice(&f.samples),
                    None => {
                        debug!(%session_id, "capture stream ended");
                        break;
                    }
                }
            }
        }

        while buffer.len() >= chunk_samples {
            let mut chunk_buf = pool.checkout();
            chunk_buf.extend_from_slice(&buffer[..chunk_samples]);
            buffer.drain(..chunk_samples);
            let offset_ms = total_audio_ms;
            total_audio_ms = total_audio_ms.saturating_add(options.chunk_ms);

            let segs = process_chunk(
                &resampler,
                &transcriber,
                &mut diarizer,
                &mut vad,
                session_id,
                chunk_index,
                offset_ms,
                options.chunk_ms,
                options.silence_rms_threshold,
                format,
                &chunk_buf,
                &transcribe_options,
                &events,
            )
            .await;
            pool.checkin(chunk_buf);
            total_segments = total_segments.saturating_add(segs);
            chunk_index = chunk_index.saturating_add(1);
        }
    }

    if let Err(e) = stream.stop().await {
        warn!(error = %e, "stream.stop returned an error");
    }

    // Flush whatever is still buffered as a final, possibly-shorter chunk.
    if !buffer.is_empty() {
        let chunk_ms = samples_to_ms(buffer.len(), format);
        let offset_ms = total_audio_ms;
        total_audio_ms = total_audio_ms.saturating_add(chunk_ms);
        let segs = process_chunk(
            &resampler,
            &transcriber,
            &mut diarizer,
            &mut vad,
            session_id,
            chunk_index,
            offset_ms,
            chunk_ms,
            options.silence_rms_threshold,
            format,
            &buffer,
            &transcribe_options,
            &events,
        )
        .await;
        total_segments = total_segments.saturating_add(segs);
    }

    info!(
        %session_id,
        total_audio_ms,
        total_segments,
        elapsed_ms = pipeline_started.elapsed().as_millis() as u64,
        "streaming pipeline stopped"
    );

    let _ = events
        .send(TranscriptEvent::Stopped {
            session_id,
            total_segments,
            total_audio_ms,
        })
        .await;
}

#[allow(clippy::too_many_arguments)]
async fn process_chunk(
    resampler: &Arc<dyn Resampler>,
    transcriber: &Arc<dyn Transcriber>,
    diarizer: &mut Option<Box<dyn Diarizer>>,
    vad: &mut Option<Box<dyn Vad>>,
    session_id: StreamingSessionId,
    chunk_index: u32,
    offset_ms: u32,
    duration_ms: u32,
    silence_threshold: f32,
    format: AudioFormat,
    raw_samples: &[Sample],
    transcribe_options: &TranscribeOptions,
    events: &mpsc::Sender<TranscriptEvent>,
) -> u32 {
    // RMS is computed up-front for telemetry (Skipped events carry it
    // so the UI can show a meaningful "below the gate" indicator),
    // and used as the cheap fast-path silence gate when no neural VAD
    // is configured. When VAD *is* configured the RMS gate is bypassed
    // — Silero distinguishes voice from non-voice noise (fans, music,
    // pure tones) that exceed any reasonable RMS threshold, and the
    // VAD's LSTM needs every chunk in chronological order to keep its
    // temporal model coherent.
    let chunk_rms = rms(raw_samples);
    if vad.is_none() && silence_threshold > 0.0 && chunk_rms < silence_threshold {
        debug!(%session_id, chunk_index, chunk_rms, "skipping silent chunk (RMS gate)");
        let _ = events
            .send(TranscriptEvent::Skipped {
                session_id,
                chunk_index,
                offset_ms,
                duration_ms,
                rms: chunk_rms,
            })
            .await;
        return 0;
    }

    let resampled = match resampler.to_whisper(raw_samples, format) {
        Ok(s) => s,
        Err(e) => {
            warn!(error = %e, chunk_index, "resampler failed; dropping chunk");
            return 0;
        }
    };

    // Neural VAD gate runs on the resampled 16 kHz mono buffer (the
    // sample rate Silero was trained for). We always push the chunk
    // — even loud ones — so the LSTM hidden state stays in sync with
    // the audio stream; the decision to skip transcription is based
    // purely on the returned `VoiceState`.
    if let Some(v) = vad.as_deref_mut() {
        match v.push(&resampled).await {
            Ok(VoiceState::Silence) => {
                debug!(%session_id, chunk_index, chunk_rms, "skipping silent chunk (neural VAD)");
                let _ = events
                    .send(TranscriptEvent::Skipped {
                        session_id,
                        chunk_index,
                        offset_ms,
                        duration_ms,
                        rms: chunk_rms,
                    })
                    .await;
                return 0;
            }
            Ok(VoiceState::Voiced) => {
                // Fall through to diarize + transcribe.
            }
            Err(e) => {
                // VAD is best-effort: if it fails we still want to
                // transcribe rather than silently drop the chunk.
                warn!(error = %e, chunk_index, "VAD failed; transcribing chunk anyway");
            }
        }
    }

    // Diarize the same 16 kHz mono buffer we're about to transcribe.
    // Done before the ASR call so the call site stays linear; the
    // ASR call below is by far the dominant cost (RTF ≪ 1 → tens of
    // ms vs Whisper's hundreds), so the lack of parallelism here is
    // deliberate: keeping the borrow on `diarizer` exclusive is
    // simpler than spawning a second task and joining.
    let speaker = match diarizer.as_deref_mut() {
        Some(d) => assign_speaker(d, &resampled, chunk_index, session_id).await,
        None => None,
    };

    let asr_started = Instant::now();
    let result = transcriber.transcribe(&resampled, transcribe_options).await;
    let asr_elapsed = asr_started.elapsed();

    match result {
        Ok(mut transcript) => {
            // Make segment timestamps absolute relative to the session start.
            for seg in &mut transcript.segments {
                seg.start_ms = seg.start_ms.saturating_add(offset_ms);
                seg.end_ms = seg.end_ms.saturating_add(offset_ms);
                // Stamp the speaker on every segment in this chunk so
                // the storage layer can persist it without needing the
                // chunk-level event.
                if let Some((id, _)) = speaker {
                    seg.speaker_id = Some(id);
                }
            }
            let segs = transcript.segments.len() as u32;
            let rtf = if duration_ms == 0 {
                0.0
            } else {
                (asr_elapsed.as_secs_f32())
                    / (Duration::from_millis(u64::from(duration_ms)).as_secs_f32())
            };
            let _ = events
                .send(TranscriptEvent::Chunk {
                    session_id,
                    chunk_index,
                    offset_ms,
                    segments: transcript.segments,
                    language: transcript.language,
                    rtf,
                    speaker_id: speaker.map(|(id, _)| id),
                    speaker_slot: speaker.map(|(_, slot)| slot),
                })
                .await;
            segs
        }
        Err(e) => {
            warn!(error = %e, chunk_index, "transcribe failed; dropping chunk");
            0
        }
    }
}

/// Run the diarizer over `samples` and look up the assigned speaker's
/// arrival-order slot. Returns `None` when the diarizer chose not to
/// embed (chunk too short, low confidence, …) or errored — diarization
/// is *strictly* best-effort, so we never abort the pipeline because
/// of it.
async fn assign_speaker(
    diarizer: &mut dyn Diarizer,
    samples: &[Sample],
    chunk_index: u32,
    session_id: StreamingSessionId,
) -> Option<(SpeakerId, u32)> {
    match diarizer.assign(samples).await {
        Ok(Some(id)) => {
            let slot = diarizer
                .speakers()
                .into_iter()
                .find(|s| s.id == id)
                .map(|s| s.slot)
                .unwrap_or_else(|| {
                    // Shouldn't happen — assign() just told us this id
                    // exists. Log and fall back to slot 0 so the UI
                    // still gets *some* colour.
                    warn!(
                        %session_id, chunk_index, %id,
                        "diarizer returned id not present in speakers() snapshot"
                    );
                    0
                });
            Some((id, slot))
        }
        Ok(None) => None,
        Err(e) => {
            warn!(
                %session_id, chunk_index, error = %e,
                "diarizer failed; chunk will be unlabelled"
            );
            None
        }
    }
}

fn chunk_size_in_samples(format: AudioFormat, chunk_ms: u32) -> usize {
    let frames = (format.sample_rate_hz as u64 * u64::from(chunk_ms)) / 1_000;
    (frames as usize) * (format.channels as usize)
}

fn samples_to_ms(samples: usize, format: AudioFormat) -> u32 {
    if format.channels == 0 || format.sample_rate_hz == 0 {
        return 0;
    }
    let frames = samples / (format.channels as usize);
    let ms = (frames as u64 * 1_000) / u64::from(format.sample_rate_hz);
    ms as u32
}

fn rms(samples: &[Sample]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

#[cfg(test)]
mod tests;
