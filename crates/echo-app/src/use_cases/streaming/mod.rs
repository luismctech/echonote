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
//! When [`StreamingOptions::auto_gain`] is set (the default), every
//! captured frame first goes through an automatic gain stage (see
//! `gain`) that raises quiet capture to a normal speech level, so the
//! VAD, the RMS gate, the transcriber, the diarizer and the saved audio
//! all see normalised audio.
//!
//! The pipeline buffers raw frames at the device sample rate and cuts
//! them into chunks, runs each chunk through the resampler and the
//! transcriber, then emits a [`TranscriptEvent::Chunk`] (or
//! [`TranscriptEvent::Skipped`] when the chunk holds no speech).
//!
//! Chunking policy:
//!
//! - **Speech-aligned** (a VAD is attached and
//!   [`StreamingOptions::speech_aligned_chunks`] is set, the default):
//!   the VAD runs incrementally over the buffer and chunks end at
//!   speech pauses between [`StreamingOptions::min_chunk_ms`] and
//!   [`StreamingOptions::max_chunk_ms`]; silence between utterances is
//!   reported as skipped and never transcribed, and so are chunks with
//!   too little confident speech (see `chunker`).
//! - **Fixed**: a chunk every [`StreamingOptions::chunk_ms`], gated by
//!   the VAD per chunk when one is attached, by the RMS floor otherwise.
//!
//! Draft text (speech-aligned mode, [`StreamingOptions::draft_interval_ms`]):
//! a speech chunk only closes at a pause, up to `max_chunk_ms` after
//! its first word, so while it is open the pipeline transcribes what it
//! holds so far every `draft_interval_ms` of new audio and emits it as
//! [`TranscriptEvent::Draft`]. Drafts skip the diarizer, the session
//! clock and the audio sink, are shed while transcription runs behind
//! capture, and never change how final chunks are cut or transcribed.
//!
//! Chunk offsets are derived from the device frames actually emitted,
//! so segment timestamps stay exact with variable-length chunks.
//! Pausing flushes the buffered audio first; paused audio is not part
//! of session time.
//!
//! Backpressure: capture is read by its own task (see `read_capture`)
//! into an unbounded queue, so the device is always drained promptly.
//! A single ASR call is in flight at a time; if the transcriber is
//! slower than real time the queue grows instead of the device
//! dropping frames, and the lag is reported after every chunk as
//! [`TranscriptEvent::Progress`]. On stop, the queued audio is still
//! processed before `Stopped`.

mod chunker;
mod gain;
mod refine;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
// Tokio's clock, so tests driving time see the same instants.
use tokio::time::Instant;
use tracing::{debug, error, info, warn};

use chunker::{draft_due, ChunkBounds, Cut, SpeechTimeline, TimedFrame};
use echo_domain::{
    AudioCapture, AudioFormat, AudioFrame, AudioStream, CaptureSpec, ChunkContext, Diarizer,
    DomainError, Punctuator, RefineOutcome, Resampler, Sample, SpeakerId, SpeakerMerge, SpeakerRef,
    SpeakerSpan, StreamingOptions, StreamingSessionId, TranscribeOptions, Transcriber,
    TranscriptEvent, Vad, VoiceState,
};
use gain::AutoGain;
pub use refine::DEFAULT_REFINE_TIMEOUT;
use refine::{refine_at_stop, AttributionLedger};

use crate::FrameSink;

/// Capacity of the event channel. Large because stop drains the
/// capture backlog (two events per chunk) while a caller may await
/// [`StreamingHandle::stop`] without reading events; tokio allocates
/// the slots lazily, so the headroom costs nothing until used.
const EVENT_CHANNEL_CAPACITY: usize = 8_192;

/// Lag above which the pipeline logs (once per episode) that
/// transcription runs behind capture.
const LAG_INFO_MS: u32 = 5_000;

/// Lag above which the pipeline warns, at most once per
/// [`LAG_WARN_INTERVAL`].
const LAG_WARN_MS: u32 = 30_000;

/// Minimum spacing between two lag warnings.
const LAG_WARN_INTERVAL: Duration = Duration::from_secs(60);

/// Capture backlog above which a due draft is skipped: the queued
/// audio must reach a final chunk first.
const DRAFT_MAX_BACKLOG_MS: u32 = 2_000;

/// How far the buffered audio may trail capture before a due draft is
/// skipped. Measured to the end of the buffer, not of the emitted
/// audio (the `Progress` lag), because the open chunk itself is
/// buffered and would count as lag.
const DRAFT_MAX_BEHIND_MS: u32 = 3_000;

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
    /// pushes all audio through it **in chronological order** (the
    /// LSTM state is sequence-sensitive). With speech-aligned chunking
    /// (the default) it decides where chunks end and which stretches
    /// are silence; with fixed chunks it gates each chunk. Either way
    /// silence is emitted as `TranscriptEvent::Skipped`, same as the
    /// energy-based RMS gate, and never reaches the transcriber.
    ///
    /// When this is set the cheap RMS gate is bypassed — Silero is
    /// strictly more discriminating (rejects pure tones, fans, music
    /// and keyboard noise that the RMS gate lets through) and the
    /// pipeline must feed it every chunk for its temporal model to
    /// stay coherent.
    vad: Option<Box<dyn Vad>>,
    /// Optional punctuation restorer applied to every segment text
    /// after transcription. When `None` segments are emitted as-is
    /// (Whisper large-v3-turbo produces reasonable punctuation on its
    /// own; the restorer is most valuable for small/fast models).
    punctuator: Option<Arc<dyn Punctuator>>,
    /// Hard limit on the end-of-session diarization refine.
    refine_timeout: Duration,
    /// Optional sink for the session audio (16 kHz mono).
    audio_sink: Option<Box<dyn FrameSink>>,
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
            punctuator: None,
            refine_timeout: DEFAULT_REFINE_TIMEOUT,
            audio_sink: None,
        }
    }

    /// Override the end-of-session refine deadline
    /// ([`DEFAULT_REFINE_TIMEOUT`] by default). When the refine fails or
    /// runs out of time, the online speaker attribution stands and
    /// `Stopped` is still emitted.
    #[must_use]
    pub fn with_refine_timeout(mut self, timeout: Duration) -> Self {
        self.refine_timeout = timeout;
        self
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

    /// Attach a punctuation restorer. Applied to every segment text
    /// after transcription succeeds. Useful with small/fast models
    /// whose raw output lacks consistent punctuation.
    #[must_use]
    pub fn with_punctuator(mut self, punctuator: Arc<dyn Punctuator>) -> Self {
        self.punctuator = Some(punctuator);
        self
    }

    /// Save the session audio to `sink`: every emitted chunk, speech and
    /// skipped silence alike, resampled to 16 kHz mono, so time in the
    /// saved audio equals session time (paused audio is excluded from
    /// both). The sink is finished at stop. A write error drops the sink
    /// with a warning; transcription carries on.
    #[must_use]
    pub fn with_audio_sink(mut self, sink: Box<dyn FrameSink>) -> Self {
        self.audio_sink = Some(sink);
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
        self.start_inner(PipelineSource::Spec(spec), options).await
    }

    /// Start transcription with an already-started audio stream. Use this
    /// when the caller needs to start the capture itself (e.g. to retain
    /// [`MixControls`] for a `Mixed` session) and then hand the stream off
    /// to the pipeline.
    pub async fn start_with_stream(
        self,
        stream: Box<dyn echo_domain::AudioStream>,
        options: StreamingOptions,
    ) -> Result<StreamingHandle, StreamingError> {
        self.start_inner(PipelineSource::Prestarted(stream), options)
            .await
    }

    async fn start_inner(
        self,
        source: PipelineSource,
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

        if self.vad.is_some()
            && options.speech_aligned_chunks
            && (options.min_chunk_ms == 0 || options.min_chunk_ms > options.max_chunk_ms)
        {
            return Err(StreamingError::Domain(DomainError::Invariant(format!(
                "speech-aligned chunk bounds must satisfy 0 < min_chunk_ms ({}) <= max_chunk_ms ({})",
                options.min_chunk_ms, options.max_chunk_ms
            ))));
        }

        let paused = Arc::new(AtomicBool::new(false));

        let join = tokio::spawn(run_pipeline(
            self.capture,
            self.resampler,
            self.transcriber,
            self.diarizer,
            self.vad,
            self.punctuator,
            session_id,
            source,
            options,
            event_tx,
            stop_rx,
            Arc::clone(&paused),
            self.refine_timeout,
            self.audio_sink,
        ));

        Ok(StreamingHandle {
            session_id,
            events: event_rx,
            stop_tx: Some(stop_tx),
            join: Some(join),
            paused,
        })
    }
}

/// Describes how a pipeline session obtains its audio stream.
enum PipelineSource {
    /// Derive the stream by calling `capture.start(spec)` inside the pipeline task.
    Spec(CaptureSpec),
    /// Use a stream that was already started by the caller.
    Prestarted(Box<dyn echo_domain::AudioStream>),
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
    paused: Arc<AtomicBool>,
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

    /// Pause the pipeline. Audio frames continue to arrive but are
    /// discarded until [`resume`](Self::resume) is called.
    pub fn pause(&self) {
        self.paused.store(true, Ordering::Release);
    }

    /// Resume the pipeline after a pause.
    pub fn resume(&self) {
        self.paused.store(false, Ordering::Release);
    }

    /// Returns `true` if the pipeline is currently paused.
    #[must_use]
    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Acquire)
    }

    /// Shared reference to the pause flag.
    ///
    /// Callers that only need to flip or check the flag can clone this
    /// `Arc` and avoid locking the full handle — which matters when
    /// the drain loop holds the lock waiting for the next event.
    #[must_use]
    pub fn paused_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.paused)
    }

    /// Signal the background task to stop without waiting for it.
    /// Audio already captured is still processed, so events keep
    /// arriving until `Stopped`; callers that consume events elsewhere
    /// use this instead of [`stop`](Self::stop). Idempotent.
    pub fn request_stop(&mut self) {
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }
    }

    /// Signal the background task to stop and wait for it to drain.
    /// Idempotent.
    pub async fn stop(&mut self) -> Result<(), StreamingError> {
        self.request_stop();
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

/// A captured frame, tagged with the pause state when it arrived.
///
/// The tag is taken at capture time because the pipeline may dequeue
/// the frame much later: reading the flag then would drop audio
/// captured before a pause, or keep audio captured during it.
struct CapturedFrame {
    samples: Vec<Sample>,
    paused: bool,
}

/// Capture-side counters shared by [`read_capture`] and the pipeline.
#[derive(Default)]
struct CaptureMeter {
    /// Wall-clock time covered by non-paused frames, in microseconds.
    active_us: AtomicU64,
    /// Samples queued by the reader and not yet taken by the pipeline.
    backlog_samples: AtomicU64,
}

/// Drain `stream` into `frames` until stop is signalled (or the stop
/// sender is dropped) or the stream ends, then stop the stream.
///
/// Runs in its own task so a slow transcriber never keeps the device
/// queue from being emptied: capture adapters drop frames when their
/// bounded queue is full. Every frame accounts its arrival gap as
/// active time unless it was captured while paused.
async fn read_capture(
    mut stream: Box<dyn AudioStream>,
    mut stop_rx: oneshot::Receiver<()>,
    paused: Arc<AtomicBool>,
    frames: mpsc::UnboundedSender<CapturedFrame>,
    meter: Arc<CaptureMeter>,
    session_id: StreamingSessionId,
) {
    let mut last_arrival = Instant::now();
    loop {
        let frame = tokio::select! {
            biased;
            _ = &mut stop_rx => {
                debug!(%session_id, "stop signal received");
                break;
            }
            frame = stream.next_frame() => match frame {
                Some(f) => f,
                None => {
                    debug!(%session_id, "capture stream ended");
                    break;
                }
            },
        };
        let now = Instant::now();
        let paused = paused.load(Ordering::Acquire);
        if !paused {
            let gap = now.saturating_duration_since(last_arrival);
            let gap_us = u64::try_from(gap.as_micros()).unwrap_or(u64::MAX);
            meter.active_us.fetch_add(gap_us, Ordering::Relaxed);
        }
        last_arrival = now;
        let len = frame.samples.len() as u64;
        meter.backlog_samples.fetch_add(len, Ordering::Relaxed);
        let captured = CapturedFrame {
            samples: frame.samples,
            paused,
        };
        if frames.send(captured).is_err() {
            // The pipeline is gone; nothing will consume more audio.
            break;
        }
    }
    if let Err(e) = stream.stop().await {
        warn!(error = %e, "stream.stop returned an error");
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_pipeline(
    capture: Arc<dyn AudioCapture>,
    resampler: Arc<dyn Resampler>,
    transcriber: Arc<dyn Transcriber>,
    diarizer: Option<Box<dyn Diarizer>>,
    mut vad: Option<Box<dyn Vad>>,
    punctuator: Option<Arc<dyn Punctuator>>,
    session_id: StreamingSessionId,
    source: PipelineSource,
    options: StreamingOptions,
    events: mpsc::Sender<TranscriptEvent>,
    stop_rx: oneshot::Receiver<()>,
    paused: Arc<AtomicBool>,
    refine_timeout: Duration,
    audio_sink: Option<Box<dyn FrameSink>>,
) {
    info!(%session_id, vad_enabled = vad.is_some(), "streaming pipeline starting");

    // Reset VAD state at session start so the LSTM/hysteresis don't
    // carry over from a previous session reusing the same template.
    if let Some(v) = vad.as_deref_mut() {
        v.reset();
    }

    let stream = match source {
        PipelineSource::Spec(spec) => match capture.start(spec).await {
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
        },
        PipelineSource::Prestarted(s) => s,
    };

    let format = stream.format();
    let _ = events
        .send(TranscriptEvent::Started {
            session_id,
            input_format: format,
        })
        .await;

    let mut chunker = Chunker::new(vad, format, &options);
    let mut emitter = ChunkEmitter {
        resampler,
        transcriber,
        diarizer,
        punctuator,
        session_id,
        format,
        transcribe_options: TranscribeOptions {
            language: options.language.clone(),
            ..Default::default()
        },
        silence_threshold: options.silence_rms_threshold,
        events: events.clone(),
        chunk_index: 0,
        emitted_frames: 0,
        intake_frames: 0,
        total_segments: 0,
        epoch: 0,
        ledger: AttributionLedger::default(),
        audio_sink: audio_sink.map(std::sync::Mutex::new),
        meter: Arc::new(CaptureMeter::default()),
        lag_alarm: LagAlarm::default(),
    };
    info!(%session_id, speech_aligned = chunker.is_speech_aligned(), "chunking policy");

    // Normalises quiet capture before the VAD, ASR, diarizer and saved
    // audio see it (a -54 dBFS mic made the VAD drop real speech).
    let mut auto_gain = options.auto_gain.then(|| AutoGain::new(format));
    info!(%session_id, auto_gain = auto_gain.is_some(), "capture level normalisation");

    let (frame_tx, mut frame_rx) = mpsc::unbounded_channel();
    let reader = tokio::spawn(read_capture(
        stream,
        stop_rx,
        paused,
        frame_tx,
        Arc::clone(&emitter.meter),
        session_id,
    ));

    let mut buffer: Vec<Sample> = Vec::new();
    let pipeline_started = Instant::now();
    let mut was_paused = false;

    // Ends once the reader has stopped (stop signal or stream end) and
    // every frame it queued has been processed, so no audio captured
    // before stop is lost.
    while let Some(frame) = frame_rx.recv().await {
        emitter
            .meter
            .backlog_samples
            .fetch_sub(frame.samples.len() as u64, Ordering::Relaxed);
        let is_paused = frame.paused;

        // Emit state-change events exactly once per transition.
        if is_paused && !was_paused {
            debug!(%session_id, "pipeline paused");
            // Transcribe what was captured before the pause instead of
            // dropping it, then report the pause.
            chunker.flush(&mut buffer, &mut emitter).await;
            let _ = events.send(TranscriptEvent::Paused { session_id }).await;
            was_paused = true;
        } else if !is_paused && was_paused {
            debug!(%session_id, "pipeline resumed");
            // Reset VAD hidden state — the LSTM would be stale after
            // the gap, producing unreliable voice/silence decisions.
            chunker.reset_vad();
            // The level history spans the gap too; start it afresh.
            if let Some(gain) = auto_gain.as_mut() {
                gain.reset();
            }
            // Audio before and after the pause is not contiguous: the
            // diarizer's log must not treat it as one run.
            emitter.epoch = emitter.epoch.saturating_add(1);
            let _ = events.send(TranscriptEvent::Resumed { session_id }).await;
            was_paused = false;
        }

        // Audio captured while paused is discarded: it is not part of
        // session time.
        if is_paused {
            continue;
        }

        // After the pause check, so paused audio is never measured.
        let mut samples = frame.samples;
        if let Some(gain) = auto_gain.as_mut() {
            gain.process(&mut samples);
        }
        emitter.intake_frames += samples.len() as u64 / u64::from(emitter.format.channels.max(1));
        buffer.extend_from_slice(&samples);
        chunker.drain_ready(&mut buffer, &mut emitter).await;
    }

    if let Err(e) = reader.await {
        error!(%session_id, error = %e, "capture reader task failed");
    }
    if let Some(gain) = &auto_gain {
        // Diagnostics: how much the capture level had to be raised.
        info!(%session_id, final_gain = gain.gain(), "auto gain at stop");
    }

    // Flush whatever is still buffered as final, possibly-shorter chunks.
    chunker.flush(&mut buffer, &mut emitter).await;
    emitter.finish_audio();

    // Offline diarization refinement (ADR 0001, D3). ASR runs inline per
    // chunk, so every segment was emitted before this point and the
    // plan covers all of them. The refine runs on the blocking pool
    // under a deadline; on any failure the online result stands.
    if let Some(d) = emitter.diarizer.take() {
        match refine_at_stop(d, refine_timeout, session_id).await {
            Some((RefineOutcome::Merges(merges), _)) => {
                let payload: Vec<SpeakerMerge> = merges
                    .into_iter()
                    .map(|(absorbed, surviving)| SpeakerMerge {
                        absorbed,
                        surviving,
                    })
                    .collect();
                info!(
                    %session_id,
                    merges = payload.len(),
                    "offline diarization refinement collapsed over-split speakers"
                );
                let _ = events
                    .send(TranscriptEvent::SpeakersMerged {
                        session_id,
                        merges: payload,
                    })
                    .await;
            }
            Some((RefineOutcome::Relabel(plan), speakers)) => {
                if let Some(r) = emitter.ledger.reassign(&plan, &speakers) {
                    let plan_id = uuid::Uuid::now_v7();
                    info!(
                        %session_id,
                        %plan_id,
                        segments = r.segments.len(),
                        created = r.created.len(),
                        retired = r.retired.len(),
                        "offline diarization re-segmentation moved segments"
                    );
                    let _ = events
                        .send(TranscriptEvent::SpeakersReassigned {
                            session_id,
                            plan_id,
                            created: r.created,
                            retired: r.retired,
                            segments: r.segments,
                            chunks: r.chunks,
                        })
                        .await;
                }
            }
            Some((RefineOutcome::Unchanged, _)) | None => {}
        }
    }

    let total_audio_ms = emitter.offset_ms();
    let total_segments = emitter.total_segments;
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

/// How the buffered audio is split into chunks.
enum Chunker {
    /// Every `chunk_samples`, gated per chunk by the VAD when one is
    /// attached, by the RMS floor otherwise.
    Fixed {
        chunk_samples: usize,
        vad: Option<Box<dyn Vad>>,
    },
    /// At speech pauses, see [`SpeechChunker`].
    Speech(SpeechChunker),
}

impl Chunker {
    fn new(vad: Option<Box<dyn Vad>>, format: AudioFormat, options: &StreamingOptions) -> Self {
        match vad {
            Some(vad) if options.speech_aligned_chunks => {
                Self::Speech(SpeechChunker::new(vad, format, options))
            }
            vad => Self::Fixed {
                chunk_samples: chunk_size_in_samples(format, options.chunk_ms),
                vad,
            },
        }
    }

    fn is_speech_aligned(&self) -> bool {
        matches!(self, Self::Speech(_))
    }

    fn reset_vad(&mut self) {
        match self {
            Self::Fixed { vad, .. } => {
                if let Some(v) = vad.as_deref_mut() {
                    v.reset();
                }
            }
            Self::Speech(s) => s.vad.reset(),
        }
    }

    /// Emit every chunk the buffered audio already allows.
    async fn drain_ready(&mut self, buffer: &mut Vec<Sample>, emitter: &mut ChunkEmitter) {
        match self {
            Self::Fixed { chunk_samples, vad } => {
                while buffer.len() >= *chunk_samples {
                    emitter
                        .process(Gate::of(vad), &buffer[..*chunk_samples])
                        .await;
                    buffer.drain(..*chunk_samples);
                }
            }
            Self::Speech(s) => s.drain_ready(buffer, emitter).await,
        }
    }

    /// Emit the whole buffer (stop, pause) and leave it empty.
    async fn flush(&mut self, buffer: &mut Vec<Sample>, emitter: &mut ChunkEmitter) {
        match self {
            Self::Fixed { vad, .. } => {
                if !buffer.is_empty() {
                    emitter.process(Gate::of(vad), buffer).await;
                    buffer.clear();
                }
            }
            Self::Speech(s) => s.flush(buffer, emitter).await,
        }
    }
}

/// Audio pushed to the VAD per analysis step. Bounds the cut-decision
/// latency and keeps resampling calls reasonably large.
const ANALYSIS_BLOCK_MS: u32 = 200;

/// Window before `max_chunk_ms` searched for the quietest forced cut.
const FORCED_CUT_LOOKBACK_MS: u32 = 1_500;

/// Silence kept before a speech onset so a late VAD decision does not
/// clip the first phoneme.
const SPEECH_PRE_ROLL_MS: u32 = 300;

/// Least confident speech a speech-aligned chunk needs to reach the
/// transcriber. Chunks below it are noise blips the VAD let through
/// (one window over the start threshold plus ~0.5 s of hysteresis),
/// on which Whisper hallucinates "Gracias.". In the 2026-10 chunking
/// bench every such hallucination held at most 160 ms of confident
/// speech and every real chunk at least 288 ms.
const MIN_CONFIDENT_SPEECH_MS: u32 = 250;

/// Cuts chunks at speech pauses. Runs the VAD incrementally over the
/// buffer (in [`ANALYSIS_BLOCK_MS`] steps, in chronological order) and
/// lets [`SpeechTimeline`] decide the cut points. Chunks it emits are
/// already classified, so they bypass every per-chunk gate.
struct SpeechChunker {
    vad: Box<dyn Vad>,
    bounds: ChunkBounds,
    /// Minimum unanalysed audio before the next VAD push.
    block: usize,
    /// Buffer samples already pushed to the VAD.
    analyzed: usize,
    timeline: SpeechTimeline,
    /// New audio between two drafts of the open chunk; 0 disables them.
    draft_interval: usize,
    /// Buffer position the open chunk was last drafted up to; reset by
    /// every cut, since a cut closes the drafted region.
    last_draft_end: Option<usize>,
}

impl SpeechChunker {
    fn new(vad: Box<dyn Vad>, format: AudioFormat, options: &StreamingOptions) -> Self {
        let samples = |ms| chunk_size_in_samples(format, ms);
        Self {
            vad,
            bounds: ChunkBounds {
                min: samples(options.min_chunk_ms),
                max: samples(options.max_chunk_ms),
                lookback: samples(FORCED_CUT_LOOKBACK_MS),
                pre_roll: samples(SPEECH_PRE_ROLL_MS),
                min_speech: samples(MIN_CONFIDENT_SPEECH_MS),
            },
            block: samples(ANALYSIS_BLOCK_MS).max(1),
            analyzed: 0,
            timeline: SpeechTimeline::default(),
            draft_interval: samples(options.draft_interval_ms),
            last_draft_end: None,
        }
    }

    async fn drain_ready(&mut self, buffer: &mut Vec<Sample>, emitter: &mut ChunkEmitter) {
        if buffer.len() - self.analyzed >= self.block {
            self.analyze(buffer, emitter).await;
        }
        while let Some(cut) = self.timeline.next_cut(&self.bounds) {
            self.emit(cut, buffer, emitter).await;
        }
        self.draft_open_region(buffer, emitter).await;
    }

    /// Emit a draft of the speech chunk still open after every ready
    /// cut, when one is due (see [`draft_due`]) and transcription keeps
    /// up with capture. Finals always win: a due draft is skipped, not
    /// postponed, when the pipeline runs behind.
    async fn draft_open_region(&mut self, buffer: &[Sample], emitter: &ChunkEmitter) {
        let Some(region) = self.timeline.open_region(&self.bounds) else {
            return;
        };
        if !draft_due(&region, self.last_draft_end, self.draft_interval) {
            return;
        }
        self.last_draft_end = Some(region.end);
        let end = region.end.min(buffer.len());
        if region.start >= end {
            return;
        }
        if let Some((backlog_ms, behind_ms)) = emitter.draft_shed(buffer.len()) {
            debug!(session_id = %emitter.session_id, backlog_ms, behind_ms, "skipping draft: transcription is behind capture");
            return;
        }
        emitter
            .draft(&buffer[region.start..end], region.start)
            .await;
    }

    async fn flush(&mut self, buffer: &mut Vec<Sample>, emitter: &mut ChunkEmitter) {
        if buffer.len() > self.analyzed {
            self.analyze(buffer, emitter).await;
        }
        while let Some(cut) = self.timeline.final_cut(&self.bounds, buffer.len()) {
            self.emit(cut, buffer, emitter).await;
        }
        buffer.clear();
        self.timeline.clear();
        self.analyzed = 0;
        self.last_draft_end = None;
    }

    /// Push the unanalysed tail of `buffer` through the VAD and record
    /// its frames in buffer coordinates. The VAD sees 16 kHz audio, so
    /// frame ends are mapped back proportionally to device frames. A
    /// failing resampler or VAD marks the block voiced: transcribing
    /// noise is better than silently dropping speech.
    async fn analyze(&mut self, buffer: &[Sample], emitter: &ChunkEmitter) {
        let start = self.analyzed;
        let block = &buffer[start..];
        self.analyzed = buffer.len();
        let channels = usize::from(emitter.format.channels.max(1));
        let block_frames = block.len() / channels;

        let fail_open = |timeline: &mut SpeechTimeline| {
            timeline.push(TimedFrame {
                end: start + block_frames * channels,
                probability: 1.0,
                state: VoiceState::Voiced,
            });
        };

        let resampled = match emitter.resampler.to_whisper(block, emitter.format) {
            Ok(r) if !r.is_empty() => r,
            Ok(_) => return,
            Err(e) => {
                warn!(error = %e, "resampler failed during VAD analysis; treating block as speech");
                fail_open(&mut self.timeline);
                return;
            }
        };
        match self.vad.push_frames(&resampled).await {
            Ok(frames) => {
                for f in frames {
                    let frame_end =
                        (f.end_sample * block_frames / resampled.len()).min(block_frames);
                    self.timeline.push(TimedFrame {
                        end: start + frame_end * channels,
                        probability: f.probability,
                        state: f.state,
                    });
                }
            }
            Err(e) => {
                warn!(error = %e, "VAD failed; treating block as speech");
                fail_open(&mut self.timeline);
            }
        }
    }

    async fn emit(&mut self, cut: Cut, buffer: &mut Vec<Sample>, emitter: &mut ChunkEmitter) {
        let n = match cut {
            Cut::Silence(n) | Cut::Speech(n) => n.min(buffer.len()),
        };
        if n == 0 {
            // Cannot happen (frames never outrun the buffer), but an
            // empty cut would spin the caller's loop forever.
            self.timeline.clear();
            return;
        }
        match cut {
            Cut::Silence(_) => emitter.skip(&buffer[..n]).await,
            Cut::Speech(_) => emitter.process(Gate::Open, &buffer[..n]).await,
        }
        buffer.drain(..n);
        self.analyzed = self.analyzed.saturating_sub(n);
        self.timeline.consume(n);
        self.last_draft_end = None;
    }
}

/// Turns chunks into events and keeps the session clock. Offsets come
/// from the device frames actually emitted (`emitted_frames`), so
/// variable-length chunks keep segment timestamps exact.
struct ChunkEmitter {
    resampler: Arc<dyn Resampler>,
    transcriber: Arc<dyn Transcriber>,
    diarizer: Option<Box<dyn Diarizer>>,
    punctuator: Option<Arc<dyn Punctuator>>,
    session_id: StreamingSessionId,
    format: AudioFormat,
    transcribe_options: TranscribeOptions,
    silence_threshold: f32,
    events: mpsc::Sender<TranscriptEvent>,
    chunk_index: u32,
    /// Device frames (samples per channel) emitted so far.
    emitted_frames: u64,
    /// Non-paused device frames taken from the capture queue into the
    /// buffer so far: emitted plus the open chunk. Lag is measured
    /// against this, so waiting for a pause does not count as lag.
    intake_frames: u64,
    total_segments: u32,
    /// Capture run: 0 at start, +1 after every resume.
    epoch: u16,
    /// Vector-free record of every attribution, for the stop refine.
    ledger: AttributionLedger,
    /// Session audio sink; dropped after its first error. The mutex
    /// only makes the emitter `Sync` (it is borrowed across awaits); it
    /// is never contended.
    audio_sink: Option<std::sync::Mutex<Box<dyn FrameSink>>>,
    /// Capture-side counters, for the lag reported after every chunk.
    meter: Arc<CaptureMeter>,
    /// Rate limit for the lag log lines.
    lag_alarm: LagAlarm,
}

/// Rate limit for the "transcription is behind" log lines.
#[derive(Default)]
struct LagAlarm {
    /// Set while the lag stays above [`LAG_INFO_MS`]; re-armed below it.
    behind: bool,
    /// Last warning, to space them [`LAG_WARN_INTERVAL`] apart.
    last_warn: Option<Instant>,
}

impl LagAlarm {
    fn observe(&mut self, session_id: StreamingSessionId, lag_ms: u32, backlog_ms: u32) {
        if lag_ms <= LAG_INFO_MS {
            self.behind = false;
            return;
        }
        if !self.behind {
            self.behind = true;
            info!(%session_id, lag_ms, backlog_ms, "transcription is running behind capture");
        }
        let due = self
            .last_warn
            .is_none_or(|t| t.elapsed() >= LAG_WARN_INTERVAL);
        if lag_ms > LAG_WARN_MS && due {
            self.last_warn = Some(Instant::now());
            warn!(%session_id, lag_ms, backlog_ms, "transcription is far behind capture");
        }
    }
}

impl ChunkEmitter {
    /// Session time at the end of the audio emitted so far.
    fn offset_ms(&self) -> u32 {
        frames_to_ms(self.emitted_frames, self.format)
    }

    /// Advance the session clock and chunk index past `raw`.
    fn advance(&mut self, raw: &[Sample]) {
        self.save_audio(raw);
        let channels = u64::from(self.format.channels.max(1));
        self.emitted_frames += raw.len() as u64 / channels;
        self.chunk_index = self.chunk_index.saturating_add(1);
    }

    /// Active (non-paused) capture time so far, in milliseconds.
    fn active_ms(&self) -> u64 {
        self.meter.active_us.load(Ordering::Relaxed) / 1_000
    }

    /// Captured audio queued but not yet buffered, in milliseconds.
    fn backlog_ms(&self) -> u32 {
        let backlog = self.meter.backlog_samples.load(Ordering::Relaxed);
        samples_to_ms(usize::try_from(backlog).unwrap_or(usize::MAX), self.format)
    }

    /// `Some((backlog_ms, behind_ms))` when a draft must be skipped
    /// because transcription runs behind capture: more than
    /// [`DRAFT_MAX_BACKLOG_MS`] queued, or the end of the `buffered`
    /// samples more than [`DRAFT_MAX_BEHIND_MS`] behind capture time.
    fn draft_shed(&self, buffered: usize) -> Option<(u32, u32)> {
        let backlog_ms = self.backlog_ms();
        let buffered_end =
            u64::from(self.offset_ms()) + u64::from(samples_to_ms(buffered, self.format));
        let behind_ms =
            u32::try_from(self.active_ms().saturating_sub(buffered_end)).unwrap_or(u32::MAX);
        (backlog_ms > DRAFT_MAX_BACKLOG_MS || behind_ms > DRAFT_MAX_BEHIND_MS)
            .then_some((backlog_ms, behind_ms))
    }

    /// Transcribe and punctuate `raw`, the open speech chunk starting
    /// `start` samples into the buffer, and emit it as a draft. Only the
    /// transcriber and the punctuator run: no diarizer, attribution,
    /// session clock, chunk index or audio sink. A failure or empty
    /// text emits nothing; the final chunk follows anyway.
    async fn draft(&self, raw: &[Sample], start: usize) {
        let session_id = self.session_id;
        let resampled = match self.resampler.to_whisper(raw, self.format) {
            Ok(s) if !s.is_empty() => s,
            Ok(_) => return,
            Err(e) => {
                debug!(%session_id, error = %e, "resampler failed; no draft");
                return;
            }
        };
        let transcript = match self
            .transcriber
            .transcribe(&resampled, &self.transcribe_options)
            .await
        {
            Ok(t) => t,
            Err(e) => {
                debug!(%session_id, error = %e, "draft transcription failed");
                return;
            }
        };
        let language = transcript.language.as_deref();
        let text = transcript
            .segments
            .iter()
            .map(|seg| match self.punctuator.as_deref() {
                Some(p) => p.punctuate(&seg.text, language),
                None => seg.text.clone(),
            })
            .map(|t| t.trim().to_owned())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        if text.is_empty() {
            return;
        }
        let channels = u64::from(self.format.channels.max(1));
        let offset_ms = frames_to_ms(self.emitted_frames + start as u64 / channels, self.format);
        let _ = self
            .events
            .send(TranscriptEvent::Draft {
                session_id,
                chunk_index: self.chunk_index,
                offset_ms,
                text,
            })
            .await;
    }

    /// Report how far the pipeline's intake trails active capture time.
    async fn report_progress(&mut self) {
        let intake_ms = u64::from(frames_to_ms(self.intake_frames, self.format));
        let lag_ms = u32::try_from(self.active_ms().saturating_sub(intake_ms)).unwrap_or(u32::MAX);
        let backlog_ms = self.backlog_ms();
        self.lag_alarm.observe(self.session_id, lag_ms, backlog_ms);
        let _ = self
            .events
            .send(TranscriptEvent::Progress {
                session_id: self.session_id,
                lag_ms,
                backlog_ms,
            })
            .await;
    }

    /// Write `raw` to the audio sink as 16 kHz mono. A failed resample
    /// writes silence of the same duration so later audio stays aligned
    /// with session time.
    fn save_audio(&mut self, raw: &[Sample]) {
        let Some(sink) = self.audio_sink.as_mut() else {
            return;
        };
        let samples = self
            .resampler
            .to_whisper(raw, self.format)
            .unwrap_or_else(|e| {
                warn!(error = %e, "resampler failed; saving silence for this chunk");
                let channels = u64::from(self.format.channels.max(1));
                let frames = raw.len() as u64 / channels;
                let rate = u64::from(self.format.sample_rate_hz.max(1));
                let len = frames * u64::from(AudioFormat::WHISPER.sample_rate_hz) / rate;
                vec![0.0; usize::try_from(len).unwrap_or(0)]
            });
        let frame = AudioFrame {
            samples,
            format: AudioFormat::WHISPER,
            captured_at_ns: 0,
        };
        let sink = sink
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Err(e) = sink.accept(&frame) {
            warn!(session_id = %self.session_id, error = %e, "saving session audio failed; audio is no longer saved");
            self.audio_sink = None;
        }
    }

    /// Finish the audio sink, if any (stop).
    fn finish_audio(&mut self) {
        if let Some(sink) = self.audio_sink.take() {
            let mut sink = sink
                .into_inner()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Err(e) = sink.finish() {
                warn!(session_id = %self.session_id, error = %e, "finishing the session audio failed");
            }
        }
    }

    /// Report `raw` as skipped silence without transcribing it.
    async fn skip(&mut self, raw: &[Sample]) {
        let chunk_rms = rms(raw);
        debug!(session_id = %self.session_id, chunk_index = self.chunk_index, chunk_rms, "skipping silent chunk (speech-aligned VAD)");
        self.send_skipped(raw, chunk_rms).await;
        self.advance(raw);
        self.report_progress().await;
    }

    async fn send_skipped(&self, raw: &[Sample], chunk_rms: f32) {
        let _ = self
            .events
            .send(TranscriptEvent::Skipped {
                session_id: self.session_id,
                chunk_index: self.chunk_index,
                offset_ms: self.offset_ms(),
                duration_ms: samples_to_ms(raw.len(), self.format),
                rms: chunk_rms,
            })
            .await;
    }

    /// Gate, diarize and transcribe one chunk, then advance the clock.
    async fn process(&mut self, gate: Gate<'_>, raw: &[Sample]) {
        let segs = self.process_inner(gate, raw).await;
        self.total_segments = self.total_segments.saturating_add(segs);
        self.advance(raw);
        self.report_progress().await;
    }

    /// Returns the number of segments emitted (0 when skipped/failed).
    async fn process_inner(&mut self, gate: Gate<'_>, raw_samples: &[Sample]) -> u32 {
        let session_id = self.session_id;
        let chunk_index = self.chunk_index;
        let offset_ms = self.offset_ms();
        let duration_ms = samples_to_ms(raw_samples.len(), self.format);

        // RMS is computed up-front for telemetry (Skipped events carry it
        // so the UI can show a meaningful "below the gate" indicator),
        // and used as the cheap fast-path silence gate when no neural VAD
        // is configured. When VAD *is* configured the RMS gate is bypassed
        // — Silero distinguishes voice from non-voice noise (fans, music,
        // pure tones) that exceed any reasonable RMS threshold, and the
        // VAD's LSTM needs every chunk in chronological order to keep its
        // temporal model coherent.
        let chunk_rms = rms(raw_samples);
        if matches!(gate, Gate::Rms)
            && self.silence_threshold > 0.0
            && chunk_rms < self.silence_threshold
        {
            debug!(%session_id, chunk_index, chunk_rms, "skipping silent chunk (RMS gate)");
            self.send_skipped(raw_samples, chunk_rms).await;
            return 0;
        }

        let resampled = match self.resampler.to_whisper(raw_samples, self.format) {
            Ok(s) => s,
            Err(e) => {
                warn!(error = %e, chunk_index, "resampler failed; dropping chunk");
                return 0;
            }
        };

        // Per-chunk neural VAD gate (fixed policy) runs on the resampled
        // 16 kHz mono buffer (the sample rate Silero was trained for). We
        // always push the chunk — even loud ones — so the LSTM hidden
        // state stays in sync with the audio stream; the decision to skip
        // transcription is based purely on the returned `VoiceState`.
        if let Gate::Vad(v) = gate {
            match v.push(&resampled).await {
                Ok(VoiceState::Silence) => {
                    debug!(%session_id, chunk_index, chunk_rms, "skipping silent chunk (neural VAD)");
                    self.send_skipped(raw_samples, chunk_rms).await;
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
        //
        // `spans` may contain overlapping ranges when two speakers talk
        // simultaneously — the segmenter inside the diarizer emits one
        // span per active speaker per frame. We preserve those overlaps
        // and let the per-segment attribution below pick the dominant
        // speaker for each Whisper segment by sample-window overlap.
        // Spans are relative to this chunk, whatever its length.
        let ctx = ChunkContext {
            chunk_index,
            offset_ms,
            epoch: self.epoch,
        };
        let spans = match self.diarizer.as_deref_mut() {
            Some(d) => assign_speaker_spans(d, &resampled, ctx, session_id).await,
            None => Vec::new(),
        };

        let asr_started = Instant::now();
        let result = self
            .transcriber
            .transcribe(&resampled, &self.transcribe_options)
            .await;
        let asr_elapsed = asr_started.elapsed();

        match result {
            Ok(mut transcript) => {
                // Apply punctuation restoration before emitting so consumers
                // receive clean, consistently-punctuated text regardless of
                // which ASR model was active.
                if let Some(p) = self.punctuator.as_deref() {
                    let lang = transcript.language.as_deref();
                    for seg in &mut transcript.segments {
                        seg.text = p.punctuate(&seg.text, lang);
                    }
                }
                // Attribute each segment to the speaker whose span overlaps
                // it the most, then shift its timestamps to session time.
                // Order matters: speaker attribution uses chunk-local time
                // (before the offset_ms shift) so the sample-window math
                // lines up with the diarizer's chunk-relative spans.
                for seg in &mut transcript.segments {
                    if !spans.is_empty() {
                        let sample_start = ms_to_chunk_samples(seg.start_ms);
                        let sample_end = ms_to_chunk_samples(seg.end_ms);
                        if let Some(id) = dominant_span(&spans, sample_start, sample_end) {
                            seg.speaker_id = Some(id);
                        }
                    }
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
                // Chunk-level speaker for the UI palette: dominant span
                // across the entire chunk by sample-weighted majority.
                let chunk_dominant = chunk_dominant_speaker(&spans);
                self.ledger.record(
                    chunk_index,
                    &spans,
                    transcript.segments.iter().map(|seg| {
                        (
                            seg.id,
                            ms_to_chunk_samples(seg.start_ms.saturating_sub(offset_ms)),
                            ms_to_chunk_samples(seg.end_ms.saturating_sub(offset_ms)),
                            seg.speaker_id,
                        )
                    }),
                    chunk_dominant.map(|(id, _)| id),
                );
                let speakers = referenced_speakers(&spans, chunk_dominant);
                let _ = self
                    .events
                    .send(TranscriptEvent::Chunk {
                        session_id,
                        chunk_index,
                        offset_ms,
                        segments: transcript.segments,
                        language: transcript.language,
                        rtf,
                        speaker_id: chunk_dominant.map(|(id, _)| id),
                        speaker_slot: chunk_dominant.map(|(_, slot)| slot),
                        speakers,
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
}

/// Per-chunk gate applied by [`ChunkEmitter::process`].
enum Gate<'a> {
    /// Fixed policy with a VAD: push the chunk, skip it on `Silence`.
    Vad(&'a mut dyn Vad),
    /// Fixed policy without a VAD: skip chunks below the RMS floor.
    Rms,
    /// Speech-aligned chunk, already classified as speech.
    Open,
}

impl<'a> Gate<'a> {
    fn of(vad: &'a mut Option<Box<dyn Vad>>) -> Self {
        match vad.as_deref_mut() {
            Some(v) => Self::Vad(v),
            None => Self::Rms,
        }
    }
}

/// Convert chunk-local milliseconds to a chunk-local sample index at
/// the pipeline's canonical 16 kHz rate. Used to align Whisper
/// segments (which report time) with diarizer spans (which report
/// sample indices) inside the same chunk.
fn ms_to_chunk_samples(ms: u32) -> usize {
    (u64::from(ms) * u64::from(echo_audio_whisper_rate()) / 1_000) as usize
}

/// Distinct speakers named by `spans`, in first-seen order, always
/// including the chunk-dominant pair. Segments take their speaker from
/// `dominant_span`, so this is a superset of every `Segment::speaker_id`
/// in the chunk and lets the recorder persist them before the segments.
fn referenced_speakers(
    spans: &[SpeakerSpan],
    chunk_dominant: Option<(SpeakerId, u32)>,
) -> Vec<SpeakerRef> {
    let mut out: Vec<SpeakerRef> = Vec::new();
    let pairs = chunk_dominant
        .into_iter()
        .chain(spans.iter().map(|s| (s.speaker_id, s.slot)));
    for (id, slot) in pairs {
        if !out.iter().any(|r| r.id == id) {
            out.push(SpeakerRef { id, slot });
        }
    }
    out
}

/// Sample-weighted majority across the chunk: returns the
/// `(SpeakerId, slot)` whose total active duration is largest. `None`
/// when the diarizer emitted no spans.
fn chunk_dominant_speaker(spans: &[SpeakerSpan]) -> Option<(SpeakerId, u32)> {
    if spans.is_empty() {
        return None;
    }
    let mut totals: Vec<(SpeakerId, u32, usize)> = Vec::with_capacity(spans.len());
    for span in spans {
        let len = span.len_samples();
        if let Some((_, _, acc)) = totals.iter_mut().find(|(id, _, _)| *id == span.speaker_id) {
            *acc += len;
        } else {
            totals.push((span.speaker_id, span.slot, len));
        }
    }
    totals
        .into_iter()
        .max_by_key(|(_, _, n)| *n)
        .map(|(id, slot, _)| (id, slot))
}

/// Find the speaker whose span overlaps the `[start, end)` window the
/// most. `None` when none of the spans intersect the window.
fn dominant_span(spans: &[SpeakerSpan], start: usize, end: usize) -> Option<SpeakerId> {
    let mut best: Option<(SpeakerId, usize)> = None;
    for span in spans {
        let ov = span.overlap_samples(start, end);
        if ov == 0 {
            continue;
        }
        best = Some(match best {
            Some((_, prev)) if prev >= ov => best.unwrap(),
            _ => (span.speaker_id, ov),
        });
    }
    best.map(|(id, _)| id)
}

/// Run the diarizer over `samples` and return every speaker-attributed
/// span in the chunk. Returns an empty vec when the diarizer chose not
/// to embed (chunk too short, low confidence, …) or errored —
/// diarization is *strictly* best-effort, so we never abort the
/// pipeline because of it.
///
/// The returned spans may overlap when two speakers talk
/// simultaneously; callers are expected to handle overlap by mapping
/// individual Whisper segments to the dominant overlapping span,
/// rather than collapsing the chunk to a single speaker.
async fn assign_speaker_spans(
    diarizer: &mut dyn Diarizer,
    samples: &[Sample],
    ctx: ChunkContext,
    session_id: StreamingSessionId,
) -> Vec<SpeakerSpan> {
    match diarizer.assign_spans_at(samples, ctx).await {
        Ok(spans) => spans,
        Err(e) => {
            warn!(
                %session_id, chunk_index = ctx.chunk_index, error = %e,
                "diarizer failed; chunk will be unlabelled"
            );
            Vec::new()
        }
    }
}

fn chunk_size_in_samples(format: AudioFormat, chunk_ms: u32) -> usize {
    let frames = (format.sample_rate_hz as u64 * u64::from(chunk_ms)) / 1_000;
    (frames as usize) * (format.channels as usize)
}

fn frames_to_ms(frames: u64, format: AudioFormat) -> u32 {
    if format.sample_rate_hz == 0 {
        return 0;
    }
    u32::try_from(frames * 1_000 / u64::from(format.sample_rate_hz)).unwrap_or(u32::MAX)
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
