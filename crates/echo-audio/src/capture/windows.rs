//! Windows system-audio capture via WASAPI loopback.
//!
//! cpal's WASAPI backend automatically sets
//! `AUDCLNT_STREAMFLAGS_LOOPBACK` when you build an input stream on a
//! **render** (output) device. This means we can capture desktop audio
//! without any extra native bindings — just open the default output
//! device as an input stream through cpal.
//!
//! The adapter mirrors [`super::cpal_microphone::CpalMicrophoneCapture`]:
//! a dedicated `std::thread` owns the cpal `Stream` (which is `!Send`),
//! frames flow through a `tokio::sync::mpsc` channel, and a oneshot
//! signals the thread to tear down.
//!
//! ## Reconnect
//!
//! WASAPI never moves a stream to another endpoint: when the default
//! output changes or the device is removed, cpal reports a fatal error
//! (`StreamInvalidated` / `DeviceNotAvailable`) and the stream stops.
//! The thread then reopens the loopback on the current default output
//! with [`Backoff::DEVICE`]; frames carry the format of the stream that
//! produced them. Unlike the microphone there is no starvation watchdog:
//! a loopback stream delivers nothing while no application plays audio.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, StreamConfig, SupportedStreamConfig};
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, error, info, warn};

use echo_domain::{
    AudioCapture, AudioFormat, AudioFrame, AudioSource, AudioStream, CaptureSpec, DeviceInfo,
    DomainError, Sample,
};

use super::reconnect::{retry, stop_requested, Backoff};
use super::sample_convert::{self, rank_format};
use super::stream_error::{classify, Severity};

/// Synthetic device ID exposed via [`WasapiLoopbackCapture::list_devices`].
pub const SYSTEM_OUTPUT_DEVICE_ID: &str = "windows:wasapi:loopback-default";

const CHANNEL_CAPACITY_HINT: usize = 512;

/// Buffer period hint for WASAPI. Overrides the device default period
/// (which can be 20–100 ms on some drivers) to reduce callback latency.
const BUFFER_PERIOD: Duration = Duration::from_millis(10);

/// How often the audio thread checks for stop and stream failure.
const SUPERVISE_EVERY: Duration = Duration::from_millis(50);

/// Windows WASAPI loopback capture adapter.
///
/// Captures the system audio mix by opening an input stream on the
/// default output (render) device. cpal handles the WASAPI loopback
/// flag transparently.
#[derive(Debug, Default, Clone)]
pub struct WasapiLoopbackCapture;

impl WasapiLoopbackCapture {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl AudioCapture for WasapiLoopbackCapture {
    async fn list_devices(&self, source: AudioSource) -> Result<Vec<DeviceInfo>, DomainError> {
        if source != AudioSource::SystemOutput {
            return Err(DomainError::AudioDeviceUnavailable(format!(
                "{source:?} is not handled by the WASAPI loopback adapter; \
                 use the cpal microphone adapter instead"
            )));
        }

        tokio::task::spawn_blocking(enumerate_loopback_device)
            .await
            .map_err(|e| DomainError::AudioCaptureFailed(format!("device enumeration join: {e}")))?
    }

    async fn start(&self, spec: CaptureSpec) -> Result<Box<dyn AudioStream>, DomainError> {
        if spec.source != AudioSource::SystemOutput {
            return Err(DomainError::AudioDeviceUnavailable(format!(
                "{:?} is not handled by the WASAPI loopback adapter; \
                 use the cpal microphone adapter instead",
                spec.source
            )));
        }

        let started = tokio::task::spawn_blocking(move || start_capture(&spec))
            .await
            .map_err(|e| DomainError::AudioCaptureFailed(format!("start join: {e}")))??;

        Ok(Box::new(started))
    }
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

fn enumerate_loopback_device() -> Result<Vec<DeviceInfo>, DomainError> {
    let host = cpal::default_host();
    let device = host.default_output_device().ok_or_else(|| {
        DomainError::AudioDeviceUnavailable("no default output device for WASAPI loopback".into())
    })?;
    let name = device
        .description()
        .map(|d| d.name().to_string())
        .unwrap_or_else(|_| "System Audio (WASAPI Loopback)".to_string());

    Ok(vec![DeviceInfo {
        id: SYSTEM_OUTPUT_DEVICE_ID.to_string(),
        is_default: true,
        name,
    }])
}

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
            tokio::task::spawn_blocking(move || handle.join())
                .await
                .map_err(|e| DomainError::AudioCaptureFailed(format!("join task: {e}")))?
                .map_err(|e| {
                    DomainError::AudioCaptureFailed(format!("audio thread panicked: {e:?}"))
                })?;
        }
        self.rx.close();
        Ok(())
    }
}

impl Drop for StartedStream {
    fn drop(&mut self) {
        if let Some(tx) = self.stop.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.join.take() {
            let _ = handle.join();
        }
    }
}

fn start_capture(spec: &CaptureSpec) -> Result<StartedStream, DomainError> {
    let opened = open_default_loopback(spec.preferred_format)?;
    let format = opened.format;

    let (tx, rx) = mpsc::channel::<AudioFrame>(CHANNEL_CAPACITY_HINT);
    let (stop_tx, stop_rx) = oneshot::channel::<()>();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), DomainError>>();
    let preferred = spec.preferred_format;

    let join = thread::Builder::new()
        .name("echo-audio-wasapi-loopback".into())
        .spawn(move || run_audio_thread(opened, preferred, tx, stop_rx, ready_tx))
        .map_err(|e| DomainError::AudioCaptureFailed(format!("spawn audio thread: {e}")))?;

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

/// The default output device with the stream configuration picked for it.
struct LoopbackDevice {
    device: Device,
    config: StreamConfig,
    sample_format: SampleFormat,
    format: AudioFormat,
}

/// Resolves the current default output device and its loopback format.
fn open_default_loopback(preferred: AudioFormat) -> Result<LoopbackDevice, DomainError> {
    let host = cpal::default_host();
    let device = host.default_output_device().ok_or_else(|| {
        DomainError::AudioDeviceUnavailable("no default output device for WASAPI loopback".into())
    })?;
    let device_name = device
        .description()
        .map(|d| d.name().to_string())
        .unwrap_or_else(|_| "<unnamed>".to_string());
    let supported = pick_output_config(&device, preferred)?;
    let format = AudioFormat {
        sample_rate_hz: supported.sample_rate(),
        channels: supported.channels(),
    };
    let sample_format = supported.sample_format();

    info!(
        device = %device_name,
        actual.rate = format.sample_rate_hz,
        actual.ch = format.channels,
        sample_format = ?sample_format,
        "opening WASAPI loopback capture"
    );

    Ok(LoopbackDevice {
        device,
        config: supported.into(),
        sample_format,
        format,
    })
}

/// Pick the output device's config closest to `preferred`.
///
/// We query `supported_output_configs` because this is natively a render
/// device — cpal will use this format for the loopback input stream.
fn pick_output_config(
    device: &Device,
    preferred: AudioFormat,
) -> Result<SupportedStreamConfig, DomainError> {
    let supported_iter = device
        .supported_output_configs()
        .map_err(|e| DomainError::AudioFormatUnsupported(format!("supported_output_configs: {e}")))?
        .collect::<Vec<_>>();
    if supported_iter.is_empty() {
        return Err(DomainError::AudioFormatUnsupported(
            "output device exposes no configurations".into(),
        ));
    }

    let target_channels = preferred.channels.max(1);
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
            "no output config with {chosen_channels} channels"
        ))
    })?;

    let min = cfg.min_sample_rate();
    let max = cfg.max_sample_rate();
    let target = preferred.sample_rate_hz.clamp(min, max);
    Ok(cfg.with_sample_rate(target))
}

fn run_audio_thread(
    opened: LoopbackDevice,
    preferred: AudioFormat,
    tx: FrameSender,
    mut stop_rx: oneshot::Receiver<()>,
    ready_tx: std::sync::mpsc::Sender<Result<(), DomainError>>,
) {
    let start = Instant::now();
    let failed = Arc::new(AtomicBool::new(false));
    let mut format = opened.format;

    let stream = match play_loopback(&opened, &tx, start, &failed) {
        Ok(s) => s,
        Err(e) => {
            let _ = ready_tx.send(Err(e));
            return;
        }
    };
    let _ = ready_tx.send(Ok(()));
    debug!("WASAPI loopback stream live; watching for stop and stream failure");

    let mut current = Some(stream);
    loop {
        if stop_requested(&mut stop_rx) {
            break;
        }
        if failed.load(Ordering::Acquire) {
            current.take();
            warn!("WASAPI loopback stream ended — reopening the default output");
            let reopened = retry(Backoff::DEVICE, &mut stop_rx, |_| {
                let opened = open_default_loopback(preferred)?;
                failed.store(false, Ordering::Release);
                let stream = play_loopback(&opened, &tx, start, &failed)?;
                Ok((stream, opened.format))
            });
            match reopened {
                Some((stream, new_format)) => {
                    if new_format != format {
                        info!(
                            old.rate = format.sample_rate_hz,
                            old.ch = format.channels,
                            new.rate = new_format.sample_rate_hz,
                            new.ch = new_format.channels,
                            "WASAPI loopback reconnected with a different format"
                        );
                    } else {
                        info!("WASAPI loopback reconnected");
                    }
                    format = new_format;
                    current = Some(stream);
                }
                None => {
                    warn!("WASAPI loopback reconnection failed — ending capture");
                    break;
                }
            }
        }
        thread::sleep(SUPERVISE_EVERY);
    }

    drop(current);
    info!("WASAPI loopback capture stopped");
}

/// Builds and starts an input stream on the output device: cpal's WASAPI
/// backend detects the render endpoint and sets
/// `AUDCLNT_STREAMFLAGS_LOOPBACK`. A fatal stream error raises `failed`
/// and silences the stream until the thread replaces it.
fn play_loopback(
    opened: &LoopbackDevice,
    tx: &FrameSender,
    start: Instant,
    failed: &Arc<AtomicBool>,
) -> Result<cpal::Stream, DomainError> {
    let format = opened.format;
    let err_fn = {
        let failed = Arc::clone(failed);
        move |err: cpal::Error| match classify(&err) {
            Severity::Transient => debug!(error = %err, "transient WASAPI loopback stream error"),
            Severity::Fatal => {
                error!(error = %err, "WASAPI loopback stream error — will reconnect");
                failed.store(true, Ordering::Release);
            }
        }
    };
    let stream = sample_convert::build_input_stream(
        &opened.device,
        opened.config,
        opened.sample_format,
        {
            let tx = tx.clone();
            let failed = Arc::clone(failed);
            move |samples: Vec<Sample>| {
                if failed.load(Ordering::Acquire) {
                    return;
                }
                push_frame(&tx, format, start, samples.into_iter());
            }
        },
        err_fn,
        Some(BUFFER_PERIOD),
    )?;
    stream
        .play()
        .map_err(|e| DomainError::AudioCaptureFailed(format!("stream.play: {e}")))?;
    Ok(stream)
}

fn push_frame(
    tx: &FrameSender,
    format: AudioFormat,
    start: Instant,
    samples: impl Iterator<Item = Sample>,
) {
    let captured_at_ns = start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
    let frame = AudioFrame {
        samples: samples.collect(),
        format,
        captured_at_ns,
    };
    if let Err(mpsc::error::TrySendError::Full(_)) = tx.try_send(frame) {
        warn!("audio frame dropped — consumer is slower than WASAPI loopback capture");
    }
}
