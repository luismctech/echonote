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

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
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
            // tears down; offload to the blocking pool with a timeout so
            // we never hang the UI indefinitely.
            let join_future = tokio::task::spawn_blocking(move || handle.join());
            match tokio::time::timeout(Duration::from_secs(5), join_future).await {
                Ok(Ok(Ok(()))) => {}
                Ok(Ok(Err(e))) => {
                    warn!("cpal thread panicked: {e:?}");
                }
                Ok(Err(e)) => {
                    warn!("join task failed: {e}");
                }
                Err(_) => {
                    warn!("cpal thread join timed out after 5s — detaching");
                }
            }
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

    let preferred_format = spec.preferred_format;
    let device_id = spec.device_id.clone();

    let join = thread::Builder::new()
        .name("echo-audio-cpal".into())
        .spawn(move || {
            run_audio_thread(
                device,
                config,
                sample_format,
                format,
                tx,
                stop_rx,
                ready_tx,
                device_id,
                preferred_format,
            );
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

/// Choose the device configuration to use for capture.
///
/// **Strategy (macOS compatibility):**
///
/// On macOS, when another application (Teams, Zoom) activates Voice
/// Processing I/O (VPIO), it may reconfigure the device's nominal
/// sample rate. CoreAudio's internal AudioConverter inside the AUHAL
/// silently stops delivering callbacks if the stream was configured for
/// a rate that no longer matches the device's current operating rate.
///
/// To avoid this, we **prefer the device's current default config**
/// (which reflects the rate the hardware is actually running at right
/// now), and only fall back to the preference-based selection when the
/// default is unavailable.  The downstream resampler converts to
/// Whisper's 16 kHz mono regardless of capture rate, so capturing at
/// 48 kHz (the typical native rate) is perfectly fine.
///
/// Fallback strategy (if default_input_config fails):
/// 1. Pick the supported config whose channel count matches or is
///    minimal.
/// 2. Use the **highest** supported sample rate (closest to native).
/// 3. Prefer `f32` sample format.
fn pick_config(
    device: &Device,
    _preferred: AudioFormat,
) -> Result<SupportedStreamConfig, DomainError> {
    // Primary: use the device's current operating config. This is the
    // rate the hardware is delivering right now, even if another app
    // (Teams/Zoom VPIO) reconfigured it.
    if let Ok(default_cfg) = device.default_input_config() {
        info!(
            sample_rate = default_cfg.sample_rate(),
            channels = default_cfg.channels(),
            format = ?default_cfg.sample_format(),
            "using device default input config (native rate)"
        );
        return Ok(default_cfg);
    }

    // Fallback: enumerate supported configs and pick the best one.
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

    // Prefer minimum channel count (mono if possible).
    let chosen_channels = supported_iter
        .iter()
        .map(|c| c.channels())
        .min()
        .unwrap_or(1);

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

    // Use the highest supported rate (closest to native) so we avoid
    // relying on CoreAudio's internal rate converter.
    let rate = cfg.max_sample_rate();
    Ok(cfg.with_sample_rate(rate))
}

/// Maximum number of consecutive reconnection attempts before giving up.
const MAX_RECONNECT_ATTEMPTS: u32 = 10;

/// Base delay between reconnection attempts. Doubles on each retry
/// (exponential backoff) up to 5 s.
const RECONNECT_BASE_DELAY: Duration = Duration::from_millis(300);

/// Maximum delay between reconnection attempts.
const RECONNECT_MAX_DELAY: Duration = Duration::from_secs(5);

/// If no audio frame arrives within this duration, the device is
/// considered stalled (e.g. another app like Teams silently took it).
/// Triggers reconnection.
const STARVATION_TIMEOUT: Duration = Duration::from_secs(3);

#[allow(clippy::too_many_arguments)]
fn run_audio_thread(
    device: Device,
    config: StreamConfig,
    sample_format: SampleFormat,
    format: AudioFormat,
    tx: FrameSender,
    mut stop_rx: oneshot::Receiver<()>,
    ready_tx: std::sync::mpsc::Sender<Result<(), DomainError>>,
    device_id: Option<String>,
    preferred_format: AudioFormat,
) {
    let start = std::time::Instant::now();
    let stream_error = Arc::new(AtomicBool::new(false));
    // Epoch-millis of the last successfully pushed frame. Used by the
    // starvation watchdog to detect silent device loss.
    let last_frame_ms = Arc::new(AtomicU64::new(epoch_ms_now()));

    // Build and start the initial stream.
    let stream = match build_and_play_stream(
        &device,
        &config,
        sample_format,
        format,
        &tx,
        start,
        &stream_error,
        &last_frame_ms,
    ) {
        Ok(s) => s,
        Err(e) => {
            let _ = ready_tx.send(Err(e));
            return;
        }
    };

    let _ = ready_tx.send(Ok(()));
    debug!("cpal stream live; monitoring for errors with auto-reconnect");

    let mut current_stream = Some(stream);

    // Main loop: poll for stop signal, stream errors, or starvation.
    loop {
        match stop_rx.try_recv() {
            Ok(()) | Err(oneshot::error::TryRecvError::Closed) => break,
            Err(oneshot::error::TryRecvError::Empty) => {}
        }

        let needs_reconnect = if stream_error.load(Ordering::Acquire) {
            warn!("cpal stream error detected — attempting auto-reconnect");
            true
        } else {
            // Starvation watchdog: if no frame arrived in STARVATION_TIMEOUT,
            // the device silently stopped delivering audio (common on macOS
            // when Teams/Zoom takes the mic).
            let last = last_frame_ms.load(Ordering::Acquire);
            let now = epoch_ms_now();
            let stalled = now.saturating_sub(last) > STARVATION_TIMEOUT.as_millis() as u64;
            if stalled {
                warn!(
                    silence_ms = now.saturating_sub(last),
                    "audio starvation detected — device may have been taken by another app"
                );
            }
            stalled
        };

        if needs_reconnect {
            // Drop the broken/stalled stream immediately.
            current_stream.take();

            match reconnect_stream(
                &device_id,
                preferred_format,
                format,
                &tx,
                start,
                &stream_error,
                &last_frame_ms,
                &mut stop_rx,
            ) {
                Some(new_stream) => {
                    info!("microphone stream reconnected successfully");
                    current_stream = Some(new_stream);
                }
                None => {
                    warn!("microphone reconnection failed — ending capture");
                    break;
                }
            }
        }

        std::thread::sleep(Duration::from_millis(50));
    }

    drop(current_stream);
    drop(tx);
    info!("cpal microphone capture stopped");
}

/// Attempts to rebuild and play a CPAL stream on the same (or default)
/// device. Returns `None` if stop was requested or retries exhausted.
fn reconnect_stream(
    device_id: &Option<String>,
    preferred_format: AudioFormat,
    format: AudioFormat,
    tx: &FrameSender,
    start: std::time::Instant,
    stream_error: &Arc<AtomicBool>,
    last_frame_ms: &Arc<AtomicU64>,
    stop_rx: &mut oneshot::Receiver<()>,
) -> Option<cpal::Stream> {
    let mut delay = RECONNECT_BASE_DELAY;

    for attempt in 1..=MAX_RECONNECT_ATTEMPTS {
        // Check if stop was requested while we retry.
        match stop_rx.try_recv() {
            Ok(()) | Err(oneshot::error::TryRecvError::Closed) => return None,
            Err(oneshot::error::TryRecvError::Empty) => {}
        }

        info!(
            attempt,
            max = MAX_RECONNECT_ATTEMPTS,
            ?delay,
            "reconnect attempt"
        );

        // Sleep in small increments so we can respond to stop requests quickly.
        let sleep_step = Duration::from_millis(50);
        let mut remaining = delay;
        while remaining > Duration::ZERO {
            let step = remaining.min(sleep_step);
            std::thread::sleep(step);
            remaining = remaining.saturating_sub(step);
            // Re-check stop during the wait.
            match stop_rx.try_recv() {
                Ok(()) | Err(oneshot::error::TryRecvError::Closed) => return None,
                Err(oneshot::error::TryRecvError::Empty) => {}
            }
        }

        // Re-acquire the device (it may have been reconfigured).
        let host = cpal::default_host();
        let device = match pick_device(&host, device_id.as_deref()) {
            Ok(d) => d,
            Err(e) => {
                warn!(attempt, error = %e, "device not available yet");
                delay = (delay * 2).min(RECONNECT_MAX_DELAY);
                continue;
            }
        };

        // Re-pick config in case the device's supported formats changed.
        let supported = match pick_config(&device, preferred_format) {
            Ok(c) => c,
            Err(e) => {
                warn!(attempt, error = %e, "cannot pick config on reconnect");
                delay = (delay * 2).min(RECONNECT_MAX_DELAY);
                continue;
            }
        };

        let new_sample_format = supported.sample_format();
        let new_config: StreamConfig = supported.into();

        // Reset error flag and frame timestamp before building the new stream.
        stream_error.store(false, Ordering::Release);
        last_frame_ms.store(epoch_ms_now(), Ordering::Release);

        match build_and_play_stream(
            &device,
            &new_config,
            new_sample_format,
            format,
            tx,
            start,
            stream_error,
            last_frame_ms,
        ) {
            Ok(stream) => return Some(stream),
            Err(e) => {
                warn!(attempt, error = %e, "rebuild stream failed");
                delay = (delay * 2).min(RECONNECT_MAX_DELAY);
            }
        }
    }

    None
}

/// Builds and starts a CPAL input stream. Shared by initial start and
/// reconnection logic.
fn build_and_play_stream(
    device: &Device,
    config: &StreamConfig,
    sample_format: SampleFormat,
    format: AudioFormat,
    tx: &FrameSender,
    start: std::time::Instant,
    stream_error: &Arc<AtomicBool>,
    last_frame_ms: &Arc<AtomicU64>,
) -> Result<cpal::Stream, DomainError> {
    let make_err_fn = || {
        let flag = stream_error.clone();
        move |err: cpal::StreamError| {
            error!(error = %err, "cpal stream error — will attempt reconnect");
            flag.store(true, Ordering::Release);
        }
    };

    let stream = match sample_format {
        SampleFormat::F32 => device
            .build_input_stream(
                config,
                {
                    let tx = tx.clone();
                    let error_flag = stream_error.clone();
                    let frame_ts = last_frame_ms.clone();
                    move |data: &[f32], _info: &cpal::InputCallbackInfo| {
                        if error_flag.load(Ordering::Acquire) {
                            return;
                        }
                        push_frame(&tx, format, start, &frame_ts, data.iter().copied());
                    }
                },
                make_err_fn(),
                Some(BUFFER_PERIOD),
            )
            .map_err(|e| DomainError::AudioCaptureFailed(format!("build f32 stream: {e}")))?,
        SampleFormat::I16 => device
            .build_input_stream(
                config,
                {
                    let tx = tx.clone();
                    let error_flag = stream_error.clone();
                    let frame_ts = last_frame_ms.clone();
                    move |data: &[i16], _info: &cpal::InputCallbackInfo| {
                        if error_flag.load(Ordering::Acquire) {
                            return;
                        }
                        push_frame(
                            &tx,
                            format,
                            start,
                            &frame_ts,
                            data.iter().map(|s| f32::from(*s) / f32::from(i16::MAX)),
                        );
                    }
                },
                make_err_fn(),
                Some(BUFFER_PERIOD),
            )
            .map_err(|e| DomainError::AudioCaptureFailed(format!("build i16 stream: {e}")))?,
        SampleFormat::U16 => device
            .build_input_stream(
                config,
                {
                    let tx = tx.clone();
                    let error_flag = stream_error.clone();
                    let frame_ts = last_frame_ms.clone();
                    move |data: &[u16], _info: &cpal::InputCallbackInfo| {
                        if error_flag.load(Ordering::Acquire) {
                            return;
                        }
                        push_frame(
                            &tx,
                            format,
                            start,
                            &frame_ts,
                            data.iter().map(|s| (f32::from(*s) - 32_768.0) / 32_768.0),
                        );
                    }
                },
                make_err_fn(),
                Some(BUFFER_PERIOD),
            )
            .map_err(|e| DomainError::AudioCaptureFailed(format!("build u16 stream: {e}")))?,
        other => {
            return Err(DomainError::AudioFormatUnsupported(format!(
                "unsupported sample format: {other:?}"
            )));
        }
    };

    stream
        .play()
        .map_err(|e| DomainError::AudioCaptureFailed(format!("stream.play: {e}")))?;

    Ok(stream)
}

/// Forwards a chunk of samples to the consumer, never blocking the audio
/// thread for longer than a single `try_send`. Drops are warned but do
/// not interrupt capture.
fn push_frame(
    tx: &FrameSender,
    format: AudioFormat,
    start: std::time::Instant,
    last_frame_ms: &AtomicU64,
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
    // Update watchdog timestamp on every successful callback invocation.
    last_frame_ms.store(epoch_ms_now(), Ordering::Release);
}

/// Returns the current time in milliseconds since an arbitrary but
/// consistent epoch (process start via `Instant` would be cleaner but
/// `Instant` is not available in const/static context, so we use
/// `SystemTime` which is cheap enough for a watchdog timer).
fn epoch_ms_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
