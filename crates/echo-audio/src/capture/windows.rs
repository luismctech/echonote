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

/// Synthetic device ID exposed via [`WasapiLoopbackCapture::list_devices`].
pub const SYSTEM_OUTPUT_DEVICE_ID: &str = "windows:wasapi:loopback-default";

const CHANNEL_CAPACITY_HINT: usize = 512;

/// Buffer period hint for WASAPI. Overrides the device default period
/// (which can be 20–100 ms on some drivers) to reduce callback latency.
const BUFFER_PERIOD: Duration = Duration::from_millis(10);

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
        .name()
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
    let host = cpal::default_host();

    let device = host.default_output_device().ok_or_else(|| {
        DomainError::AudioDeviceUnavailable("no default output device for WASAPI loopback".into())
    })?;
    let device_name = device.name().unwrap_or_else(|_| "<unnamed>".to_string());
    let supported = pick_output_config(&device, spec.preferred_format)?;
    let format = AudioFormat {
        sample_rate_hz: supported.sample_rate().0,
        channels: supported.channels(),
    };
    let sample_format = supported.sample_format();
    let config: StreamConfig = supported.into();

    info!(
        device = %device_name,
        actual.rate = format.sample_rate_hz,
        actual.ch = format.channels,
        sample_format = ?sample_format,
        "starting WASAPI loopback capture"
    );

    let (tx, rx) = mpsc::channel::<AudioFrame>(CHANNEL_CAPACITY_HINT);
    let (stop_tx, stop_rx) = oneshot::channel::<()>();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), DomainError>>();

    let join = thread::Builder::new()
        .name("echo-audio-wasapi-loopback".into())
        .spawn(move || {
            run_audio_thread(device, config, sample_format, format, tx, stop_rx, ready_tx);
        })
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

    fn rank_format(f: SampleFormat) -> u8 {
        match f {
            SampleFormat::F32 => 0,
            SampleFormat::I16 => 1,
            SampleFormat::U16 => 2,
            _ => 9,
        }
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

    let min = cfg.min_sample_rate().0;
    let max = cfg.max_sample_rate().0;
    let target = preferred.sample_rate_hz.clamp(min, max);
    Ok(cfg.with_sample_rate(cpal::SampleRate(target)))
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
    let err_fn = |err: cpal::StreamError| error!(error = %err, "WASAPI loopback stream error");

    // Build an *input* stream on the *output* device — cpal's WASAPI
    // backend detects this and sets AUDCLNT_STREAMFLAGS_LOOPBACK.
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
    debug!("WASAPI loopback stream live; parking thread until stop signal");

    let _ = stop_rx.blocking_recv();
    drop(stream);
    info!("WASAPI loopback capture stopped");
}

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
        warn!("audio frame dropped — consumer is slower than WASAPI loopback capture");
    }
}
