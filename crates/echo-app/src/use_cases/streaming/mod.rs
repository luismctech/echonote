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

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

use echo_domain::{
    AudioCapture, AudioFormat, CaptureSpec, DomainError, Resampler, Sample, StreamingOptions,
    StreamingSessionId, TranscribeOptions, Transcriber, TranscriptEvent,
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
#[derive(Clone)]
pub struct StreamingPipeline {
    capture: Arc<dyn AudioCapture>,
    resampler: Arc<dyn Resampler>,
    transcriber: Arc<dyn Transcriber>,
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
        }
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

        let join = tokio::spawn(run_pipeline(
            self.capture.clone(),
            self.resampler.clone(),
            self.transcriber.clone(),
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
    session_id: StreamingSessionId,
    spec: CaptureSpec,
    options: StreamingOptions,
    events: mpsc::Sender<TranscriptEvent>,
    mut stop_rx: oneshot::Receiver<()>,
) {
    info!(%session_id, "streaming pipeline starting");

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
            let chunk_samples_vec: Vec<Sample> = buffer.drain(..chunk_samples).collect();
            let offset_ms = total_audio_ms;
            total_audio_ms = total_audio_ms.saturating_add(options.chunk_ms);

            let segs = process_chunk(
                &resampler,
                &transcriber,
                session_id,
                chunk_index,
                offset_ms,
                options.chunk_ms,
                options.silence_rms_threshold,
                format,
                chunk_samples_vec,
                &transcribe_options,
                &events,
            )
            .await;
            total_segments = total_segments.saturating_add(segs);
            chunk_index = chunk_index.saturating_add(1);
        }
    }

    if let Err(e) = stream.stop().await {
        warn!(error = %e, "stream.stop returned an error");
    }

    // Flush whatever is still buffered as a final, possibly-shorter chunk.
    if !buffer.is_empty() {
        let chunk_samples_vec = std::mem::take(&mut buffer);
        let chunk_ms = samples_to_ms(chunk_samples_vec.len(), format);
        let offset_ms = total_audio_ms;
        total_audio_ms = total_audio_ms.saturating_add(chunk_ms);
        let segs = process_chunk(
            &resampler,
            &transcriber,
            session_id,
            chunk_index,
            offset_ms,
            chunk_ms,
            options.silence_rms_threshold,
            format,
            chunk_samples_vec,
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
    session_id: StreamingSessionId,
    chunk_index: u32,
    offset_ms: u32,
    duration_ms: u32,
    silence_threshold: f32,
    format: AudioFormat,
    raw_samples: Vec<Sample>,
    transcribe_options: &TranscribeOptions,
    events: &mpsc::Sender<TranscriptEvent>,
) -> u32 {
    // Silence gate runs on the raw multi-channel buffer; that's fine —
    // RMS is invariant to channel layout for our purposes.
    let chunk_rms = rms(&raw_samples);
    if silence_threshold > 0.0 && chunk_rms < silence_threshold {
        debug!(%session_id, chunk_index, chunk_rms, "skipping silent chunk");
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

    let resampled = match resampler.to_whisper(&raw_samples, format) {
        Ok(s) => s,
        Err(e) => {
            warn!(error = %e, chunk_index, "resampler failed; dropping chunk");
            return 0;
        }
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
