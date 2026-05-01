//! Cross-platform microphone capture backed by [`cpal`].
//!
//! cpal owns its own real-time audio thread per stream. We bridge from
//! that thread to async land via an unbounded `tokio::sync::mpsc`
//! channel: the cpal callback only forwards a `Vec<f32>` and never
//! blocks, allocates beyond `Vec::with_capacity` or holds a mutex.
//!
//! ## Lifetime / Send concerns
//!
//! `cpal::Stream` is `!Send` on every host because it is tied to the OS
//! audio thread that owns the buffer. To keep the [`AudioStream`] handle
//! `Send` we move ownership of the cpal `Stream` onto a dedicated
//! `std::thread`, parking that thread until a oneshot tells it to drop
//! the stream.

use std::thread;
use std::time::Duration;

use async_trait::async_trait;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, StreamConfig, SupportedStreamConfig};
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, error, info, warn};

use echo_domain::{
    AudioCapture, AudioFormat, AudioFrame, AudioSource, AudioStream, CaptureSpec, DeviceInfo,
    DomainError, Sample,
};

/// Number of frames buffered between the audio thread and consumers
/// before warnings fire. Sized for 5 s of 48 kHz stereo (~2 MB).
const CHANNEL_CAPACITY_HINT: usize = 512;

/// Buffer period hint passed to `build_input_stream`. On macOS
/// CoreAudio this has no visible effect (the default is already ~5 ms).
/// On Windows WASAPI shared-mode it overrides the device period that can
/// otherwise be 20–100 ms, significantly reducing callback latency.
const BUFFER_PERIOD: Duration = Duration::from_millis(10);

/// Default cpal-based microphone adapter.
///
/// Cheap to construct; does not touch the host until [`Self::start`] or
/// [`Self::list_devices`] is called.
#[derive(Debug, Default, Clone)]
pub struct CpalMicrophoneCapture;

impl CpalMicrophoneCapture {
    /// Constructs an adapter bound to the system default cpal host.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl AudioCapture for CpalMicrophoneCapture {
    async fn list_devices(&self, source: AudioSource) -> Result<Vec<DeviceInfo>, DomainError> {
        if source != AudioSource::Microphone {
            return Err(DomainError::AudioDeviceUnavailable(format!(
                "{source:?} capture is not implemented on this build (Phase 1 work)"
            )));
        }

        // cpal device enumeration is synchronous and may block on host
        // queries (CoreAudio in particular). Run it on a blocking pool
        // so we do not stall the async runtime.
        tokio::task::spawn_blocking(enumerate_input_devices)
            .await
            .map_err(|e| DomainError::AudioCaptureFailed(format!("device enumeration join: {e}")))?
    }

    async fn start(&self, spec: CaptureSpec) -> Result<Box<dyn AudioStream>, DomainError> {
        if spec.source != AudioSource::Microphone {
            return Err(DomainError::AudioDeviceUnavailable(format!(
                "{:?} capture is not implemented on this build (Phase 1 work)",
                spec.source
            )));
        }

        // Blocking the runtime here is OK because cpal's start path is
        // milliseconds-fast and we want errors propagated synchronously.
        let started = tokio::task::spawn_blocking(move || start_capture(&spec))
            .await
            .map_err(|e| DomainError::AudioCaptureFailed(format!("start join: {e}")))??;

        Ok(Box::new(started))
    }
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

fn enumerate_input_devices() -> Result<Vec<DeviceInfo>, DomainError> {
    let host = cpal::default_host();
    let default_id = host
        .default_input_device()
        .and_then(|d| d.description().ok().map(|desc| desc.name().to_string()))
        .unwrap_or_default();

    let devices = host
        .input_devices()
        .map_err(|e| DomainError::AudioDeviceUnavailable(format!("input_devices: {e}")))?;

    let mut out = Vec::new();
    for dev in devices {
        let name = dev
            .description()
            .map(|d| d.name().to_string())
            .unwrap_or_else(|_| "<unnamed>".to_string());
        out.push(DeviceInfo {
            id: name.clone(),
            is_default: !default_id.is_empty() && name == default_id,
            name,
        });
    }
    Ok(out)
}

/// Sender half pushed from the cpal callback. Kept as a type alias so
/// the callback signature stays readable.
type FrameSender = mpsc::Sender<AudioFrame>;

struct StartedStream {
    rx: mpsc::Receiver<AudioFrame>,
    stop: Option<oneshot::Sender<()>>,
    format: AudioFormat,
    join: Option<thread::JoinHandle<()>>,
}

#[async_trait]
impl AudioStream for StartedStream {
    fn format(&self) -> AudioFormat {
        self.format
    }

    async fn next_frame(&mut self) -> Option<AudioFrame> {
        self.rx.recv().await
    }

    async fn stop(&mut self) -> Result<(), DomainError> {
        if let Some(tx) = self.stop.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.join.take() {
            // Joining the cpal thread can block briefly while the device
            // tears down; offload to the blocking pool.
            tokio::task::spawn_blocking(move || handle.join())
                .await
                .map_err(|e| DomainError::AudioCaptureFailed(format!("join task: {e}")))?
                .map_err(|e| {
                    DomainError::AudioCaptureFailed(format!("cpal thread panicked: {e:?}"))
                })?;
        }
        // Drain any frames still queued so subsequent next_frame() calls
        // return None promptly.
        self.rx.close();
        Ok(())
    }
}

impl Drop for StartedStream {
    fn drop(&mut self) {
        if let Some(tx) = self.stop.take() {
            let _ = tx.send(());
        }
        // Best-effort thread join from a sync context. We do not block
        // forever; cpal teardown is bounded by the host.
        if let Some(handle) = self.join.take() {
            let _ = handle.join();
        }
    }
}

fn start_capture(spec: &CaptureSpec) -> Result<StartedStream, DomainError> {
    let host = cpal::default_host();

    let device = pick_device(&host, spec.device_id.as_deref())?;
    let device_name = device
        .description()
        .map(|d| d.name().to_string())
        .unwrap_or_else(|_| "<unnamed>".to_string());
    let supported = pick_config(&device, spec.preferred_format)?;
    let format = AudioFormat {
        sample_rate_hz: supported.sample_rate(),
        channels: supported.channels(),
    };
    let sample_format = supported.sample_format();
    let config: StreamConfig = supported.into();

    info!(
        device = %device_name,
        requested.rate = spec.preferred_format.sample_rate_hz,
        requested.ch = spec.preferred_format.channels,
        actual.rate = format.sample_rate_hz,
        actual.ch = format.channels,
        sample_format = ?sample_format,
        "starting microphone capture"
    );

    let (tx, rx) = mpsc::channel::<AudioFrame>(CHANNEL_CAPACITY_HINT);
    let (stop_tx, stop_rx) = oneshot::channel::<()>();
    // Channel that lets the audio thread report a build error back to
    // the caller before the thread parks.
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), DomainError>>();

    let join = thread::Builder::new()
        .name("echo-audio-cpal".into())
        .spawn(move || {
            run_audio_thread(device, config, sample_format, format, tx, stop_rx, ready_tx);
        })
        .map_err(|e| DomainError::AudioCaptureFailed(format!("spawn audio thread: {e}")))?;

    // Wait for the thread to either start the cpal stream or fail. This
    // turns build errors into synchronous Result propagation rather than
    // silent stalls.
    match ready_rx.recv() {
        Ok(Ok(())) => Ok(StartedStream {
            rx,
            stop: Some(stop_tx),
            format,
            join: Some(join),
        }),
        Ok(Err(e)) => Err(e),
        Err(e) => Err(DomainError::AudioCaptureFailed(format!(
            "audio thread died before signalling readiness: {e}"
        ))),
    }
}

fn pick_device(host: &cpal::Host, device_id: Option<&str>) -> Result<Device, DomainError> {
    if let Some(id) = device_id {
        let devices = host
            .input_devices()
            .map_err(|e| DomainError::AudioDeviceUnavailable(format!("input_devices: {e}")))?;
        for dev in devices {
            if dev
                .description()
                .ok()
                .map(|d| d.name().to_string())
                .as_deref()
                == Some(id)
            {
                return Ok(dev);
            }
        }
        Err(DomainError::AudioDeviceUnavailable(format!(
            "no input device named {id:?}"
        )))
    } else {
        host.default_input_device().ok_or_else(|| {
            DomainError::AudioDeviceUnavailable("host has no default input device".into())
        })
    }
}

/// Choose the configuration closest to `preferred`.
///
/// Strategy:
/// 1. Pick the supported config whose channel count exactly matches
///    `preferred.channels`, else minimum channels available.
/// 2. From those, pick the config whose sample-rate range contains
///    `preferred.sample_rate_hz`, clamped to the supported range.
/// 3. Prefer `f32` sample format, falling back to `i16` then `u16`.
fn pick_config(
    device: &Device,
    preferred: AudioFormat,
) -> Result<SupportedStreamConfig, DomainError> {
    let supported_iter = device
        .supported_input_configs()
        .map_err(|e| DomainError::AudioFormatUnsupported(format!("supported_input_configs: {e}")))?
        .collect::<Vec<_>>();
    if supported_iter.is_empty() {
        return Err(DomainError::AudioFormatUnsupported(
            "device exposes no input configurations".into(),
        ));
    }

    // Preference order on sample format.
    fn rank_format(f: SampleFormat) -> u8 {
        match f {
            SampleFormat::F32 => 0,
            SampleFormat::I16 => 1,
            SampleFormat::U16 => 2,
            _ => 9,
        }
    }

    let target_channels = preferred.channels.max(1);

    // Pick the closest channel count: prefer exact match, else the
    // smallest supported count.
    let chosen_channels = supported_iter
        .iter()
        .map(|c| c.channels())
        .min_by_key(|&ch| (if ch == target_channels { 0 } else { 1 }, ch))
        .unwrap_or(target_channels);

    let mut candidates: Vec<_> = supported_iter
        .into_iter()
        .filter(|c| c.channels() == chosen_channels)
        .collect();
    candidates.sort_by_key(|c| rank_format(c.sample_format()));

    let cfg = candidates.into_iter().next().ok_or_else(|| {
        DomainError::AudioFormatUnsupported(format!(
            "no input config with {chosen_channels} channels"
        ))
    })?;

    let min = cfg.min_sample_rate();
    let max = cfg.max_sample_rate();
    let target = preferred.sample_rate_hz.clamp(min, max);
    Ok(cfg.with_sample_rate(target))
}

fn run_audio_thread(
    device: Device,
    config: StreamConfig,
    sample_format: SampleFormat,
    format: AudioFormat,
    tx: FrameSender,
    stop_rx: oneshot::Receiver<()>,
    ready_tx: std::sync::mpsc::Sender<Result<(), DomainError>>,
) {
    let start = std::time::Instant::now();
    let err_fn = |err: cpal::StreamError| error!(error = %err, "cpal stream error");

    let build_result: Result<cpal::Stream, DomainError> = match sample_format {
        SampleFormat::F32 => device
            .build_input_stream(
                &config,
                {
                    let tx = tx.clone();
                    move |data: &[f32], _info: &cpal::InputCallbackInfo| {
                        push_frame(&tx, format, start, data.iter().copied());
                    }
                },
                err_fn,
                Some(BUFFER_PERIOD),
            )
            .map_err(|e| DomainError::AudioCaptureFailed(format!("build f32 stream: {e}"))),
        SampleFormat::I16 => device
            .build_input_stream(
                &config,
                {
                    let tx = tx.clone();
                    move |data: &[i16], _info: &cpal::InputCallbackInfo| {
                        push_frame(
                            &tx,
                            format,
                            start,
                            data.iter().map(|s| f32::from(*s) / f32::from(i16::MAX)),
                        );
                    }
                },
                err_fn,
                Some(BUFFER_PERIOD),
            )
            .map_err(|e| DomainError::AudioCaptureFailed(format!("build i16 stream: {e}"))),
        SampleFormat::U16 => device
            .build_input_stream(
                &config,
                {
                    let tx = tx.clone();
                    move |data: &[u16], _info: &cpal::InputCallbackInfo| {
                        push_frame(
                            &tx,
                            format,
                            start,
                            data.iter().map(|s| (f32::from(*s) - 32_768.0) / 32_768.0),
                        );
                    }
                },
                err_fn,
                Some(BUFFER_PERIOD),
            )
            .map_err(|e| DomainError::AudioCaptureFailed(format!("build u16 stream: {e}"))),
        other => Err(DomainError::AudioFormatUnsupported(format!(
            "unsupported sample format: {other:?}"
        ))),
    };

    let stream = match build_result {
        Ok(s) => s,
        Err(e) => {
            let _ = ready_tx.send(Err(e));
            return;
        }
    };

    if let Err(e) = stream.play() {
        let _ = ready_tx.send(Err(DomainError::AudioCaptureFailed(format!(
            "stream.play: {e}"
        ))));
        return;
    }

    let _ = ready_tx.send(Ok(()));
    debug!("cpal stream live; parking thread until stop signal");

    // Block until a stop is requested. The Stream is dropped at the end
    // of this scope, which also stops the underlying audio thread.
    let _ = stop_rx.blocking_recv();
    drop(stream);
    info!("cpal microphone capture stopped");
}

/// Forwards a chunk of samples to the consumer, never blocking the audio
/// thread for longer than a single `try_send`. Drops are warned but do
/// not interrupt capture.
fn push_frame(
    tx: &FrameSender,
    format: AudioFormat,
    start: std::time::Instant,
    samples: impl Iterator<Item = Sample>,
) {
    let captured_at_ns = start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
    let frame = AudioFrame {
        samples: samples.collect(),
        format,
        captured_at_ns,
    };
    if let Err(mpsc::error::TrySendError::Full(_)) = tx.try_send(frame) {
        warn!("audio frame dropped — consumer is slower than capture");
    }
}
