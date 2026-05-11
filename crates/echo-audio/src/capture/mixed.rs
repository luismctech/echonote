//! Mixed-source audio capture: mic + system audio merged into one stream.
//!
//! `MixedStream` reads concurrently from a microphone stream and a system
//! audio (loopback) stream, accumulates samples in two independent ring
//! buffers, and emits mixed frames whenever either buffer reaches the
//! threshold. Mixing is a per-sample average with a simple zero-pad when
//! one source is behind the other.
//!
//! Each source can be independently muted at any time by storing `false`
//! in the corresponding `Arc<AtomicBool>` returned alongside the stream.
//! When a source is muted its samples are zeroed in the mix; the underlying
//! capture keeps running so no audio is dropped on unmute.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tracing::{debug, warn};

use echo_domain::{AudioFormat, AudioFrame, AudioStream, DomainError, Sample};

/// Interleaved-sample threshold before a mixed frame is emitted.
/// ~10 ms at 48 kHz stereo (1024 interleaved samples = 512 stereo frames).
const MIX_THRESHOLD: usize = 1024;

/// Maximum per-source buffer depth before old samples are dropped.
/// ~340 ms at 48 kHz stereo — protects against pathological clock drift.
const MAX_DRIFT_SAMPLES: usize = MIX_THRESHOLD * 16;

/// Channel depth in mixed frames. ~5 s of headroom at 10 ms/frame.
const CHANNEL_CAPACITY: usize = 512;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Atomic flags that independently gate each source's contribution to the mix.
///
/// Both flags start as `true` (both sources contribute). Setting a flag to
/// `false` zeros out that source's samples; setting it back to `true`
/// restores contribution immediately on the next emitted frame.
#[derive(Clone)]
pub struct MixControls {
    /// When `true` the microphone samples are included in the mix.
    pub mic_active: Arc<AtomicBool>,
    /// When `true` the system-audio samples are included in the mix.
    pub sys_active: Arc<AtomicBool>,
}

impl Default for MixControls {
    fn default() -> Self {
        Self {
            mic_active: Arc::new(AtomicBool::new(true)),
            sys_active: Arc::new(AtomicBool::new(true)),
        }
    }
}

/// An [`AudioStream`] that merges a microphone stream and a system-audio
/// stream into a single output. Construct via [`MixedStream::new`].
pub struct MixedStream {
    rx: mpsc::Receiver<AudioFrame>,
    stop_tx: Option<oneshot::Sender<()>>,
    join: Option<JoinHandle<()>>,
    format: AudioFormat,
}

impl MixedStream {
    /// Spawn the mixer task and return the stream + its control handles.
    ///
    /// Both `mic_stream` and `sys_stream` **must** be live capture streams
    /// (calling `next_frame()` should not immediately return `None`).
    /// `format` is the declared output format — both sources are treated as
    /// if they deliver samples at this rate/channel count, so callers should
    /// ensure both were started with a matching `preferred_format`.
    pub fn new(
        mic_stream: Box<dyn AudioStream>,
        sys_stream: Box<dyn AudioStream>,
        format: AudioFormat,
    ) -> (Self, MixControls) {
        let controls = MixControls::default();
        let (tx, rx) = mpsc::channel::<AudioFrame>(CHANNEL_CAPACITY);
        let (stop_tx, stop_rx) = oneshot::channel::<()>();

        let mic_active = controls.mic_active.clone();
        let sys_active = controls.sys_active.clone();

        let join = tokio::spawn(run_mixer(
            mic_stream, sys_stream, format, tx, stop_rx, mic_active, sys_active,
        ));

        let stream = Self {
            rx,
            stop_tx: Some(stop_tx),
            join: Some(join),
            format,
        };

        (stream, controls)
    }
}

#[async_trait]
impl AudioStream for MixedStream {
    fn format(&self) -> AudioFormat {
        self.format
    }

    async fn next_frame(&mut self) -> Option<AudioFrame> {
        self.rx.recv().await
    }

    async fn stop(&mut self) -> Result<(), DomainError> {
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }
        if let Some(join) = self.join.take() {
            join.abort();
        }
        self.rx.close();
        Ok(())
    }
}

impl Drop for MixedStream {
    fn drop(&mut self) {
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }
        if let Some(join) = self.join.take() {
            join.abort();
        }
    }
}

// ---------------------------------------------------------------------------
// Mixer task
// ---------------------------------------------------------------------------

async fn run_mixer(
    mut mic: Box<dyn AudioStream>,
    mut sys: Box<dyn AudioStream>,
    format: AudioFormat,
    tx: mpsc::Sender<AudioFrame>,
    mut stop_rx: oneshot::Receiver<()>,
    mic_active: Arc<AtomicBool>,
    sys_active: Arc<AtomicBool>,
) {
    debug!("mixed-stream mixer started");

    let mut mic_buf: Vec<Sample> = Vec::with_capacity(MIX_THRESHOLD * 4);
    let mut sys_buf: Vec<Sample> = Vec::with_capacity(MIX_THRESHOLD * 4);
    let mut elapsed_ns: u64 = 0;

    // Nanoseconds per interleaved sample at the declared output rate.
    let ns_per_sample = if format.sample_rate_hz > 0 && format.channels > 0 {
        1_000_000_000u64 / (format.sample_rate_hz as u64 * format.channels as u64)
    } else {
        0
    };

    loop {
        // Drain both buffers into mixed frames before waiting for more input.
        while mic_buf.len() >= MIX_THRESHOLD || sys_buf.len() >= MIX_THRESHOLD {
            let n = MIX_THRESHOLD;
            let use_mic = mic_active.load(Ordering::Relaxed);
            let use_sys = sys_active.load(Ordering::Relaxed);

            let mut mixed = Vec::with_capacity(n);
            for i in 0..n {
                let mic_s = if use_mic && i < mic_buf.len() {
                    mic_buf[i]
                } else {
                    0.0_f32
                };
                let sys_s = if use_sys && i < sys_buf.len() {
                    sys_buf[i]
                } else {
                    0.0_f32
                };
                let sample = match (use_mic && i < mic_buf.len(), use_sys && i < sys_buf.len()) {
                    (true, true) => (mic_s + sys_s) * 0.5,
                    (true, false) => mic_s,
                    (false, true) => sys_s,
                    (false, false) => 0.0_f32,
                };
                mixed.push(sample.clamp(-1.0, 1.0));
            }

            let consumed_mic = n.min(mic_buf.len());
            let consumed_sys = n.min(sys_buf.len());
            mic_buf.drain(..consumed_mic);
            sys_buf.drain(..consumed_sys);

            let frame = AudioFrame {
                samples: mixed,
                format,
                captured_at_ns: elapsed_ns,
            };
            elapsed_ns = elapsed_ns.saturating_add(ns_per_sample * n as u64);

            if tx.send(frame).await.is_err() {
                debug!("mixed-stream: consumer dropped, stopping mixer");
                return;
            }
        }

        // Wait for new samples from either source.
        tokio::select! {
            _ = &mut stop_rx => {
                debug!("mixed-stream: stop signal received");
                break;
            }
            frame = mic.next_frame() => {
                match frame {
                    Some(f) => mic_buf.extend_from_slice(&f.samples),
                    None => {
                        debug!("mixed-stream: mic stream ended");
                        break;
                    }
                }
            }
            frame = sys.next_frame() => {
                match frame {
                    Some(f) => sys_buf.extend_from_slice(&f.samples),
                    None => {
                        debug!("mixed-stream: system stream ended");
                        break;
                    }
                }
            }
        }

        // Trim buffers that have drifted too far ahead.
        if mic_buf.len() > MAX_DRIFT_SAMPLES {
            let excess = mic_buf.len() - MAX_DRIFT_SAMPLES;
            warn!(
                excess,
                "mixed-stream: mic buffer drifted; dropping old samples"
            );
            mic_buf.drain(..excess);
        }
        if sys_buf.len() > MAX_DRIFT_SAMPLES {
            let excess = sys_buf.len() - MAX_DRIFT_SAMPLES;
            warn!(
                excess,
                "mixed-stream: sys buffer drifted; dropping old samples"
            );
            sys_buf.drain(..excess);
        }
    }

    debug!("mixed-stream mixer stopped");
}
