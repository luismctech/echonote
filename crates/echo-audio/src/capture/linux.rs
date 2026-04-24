//! Linux system-audio capture via PulseAudio monitor sources.
//!
//! PulseAudio (and PipeWire's PulseAudio compatibility layer) exposes a
//! "Monitor of &lt;sink&gt;" source for every output sink. On most Linux
//! desktops the PulseAudio-ALSA bridge makes these appear as regular
//! input devices through cpal's default ALSA host. We find the monitor
//! source for the default output sink and open a standard cpal input
//! stream on it.
//!
//! The adapter mirrors [`super::cpal_microphone::CpalMicrophoneCapture`]:
//! a dedicated `std::thread` owns the cpal `Stream` (which is `!Send`),
//! frames flow through a `tokio::sync::mpsc` channel, and a oneshot
//! signals the thread to tear down.

use std::thread;

use async_trait::async_trait;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, StreamConfig, SupportedStreamConfig};
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, error, info, warn};

use echo_domain::{
    AudioCapture, AudioFormat, AudioFrame, AudioSource, AudioStream, CaptureSpec, DeviceInfo,
    DomainError, Sample,
};

/// Synthetic device ID for the default monitor source.
pub const SYSTEM_OUTPUT_DEVICE_ID: &str = "linux:pulseaudio:monitor-default";

const CHANNEL_CAPACITY_HINT: usize = 512;

/// Linux PulseAudio monitor capture adapter.
///
/// Captures the system audio mix by opening an input stream on the
/// PulseAudio "Monitor of ..." source that mirrors the default output
/// sink.
#[derive(Debug, Default, Clone)]
pub struct PulseMonitorCapture;

impl PulseMonitorCapture {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl AudioCapture for PulseMonitorCapture {
    async fn list_devices(&self, source: AudioSource) -> Result<Vec<DeviceInfo>, DomainError> {
        if source != AudioSource::SystemOutput {
            return Err(DomainError::AudioDeviceUnavailable(format!(
                "{source:?} is not handled by the PulseAudio monitor adapter; \
                 use the cpal microphone adapter instead"
            )));
        }

        tokio::task::spawn_blocking(enumerate_monitor_devices)
            .await
            .map_err(|e| DomainError::AudioCaptureFailed(format!("device enumeration join: {e}")))?
    }

    async fn start(&self, spec: CaptureSpec) -> Result<Box<dyn AudioStream>, DomainError> {
        if spec.source != AudioSource::SystemOutput {
            return Err(DomainError::AudioDeviceUnavailable(format!(
                "{:?} is not handled by the PulseAudio monitor adapter; \
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

/// PulseAudio monitor sources contain "Monitor of" in their name.
fn is_monitor_device(name: &str) -> bool {
    let lower = name.to_lowercase();
    lower.contains("monitor of") || lower.contains("monitor_of")
}

fn enumerate_monitor_devices() -> Result<Vec<DeviceInfo>, DomainError> {
    let host = cpal::default_host();
    let devices = host
        .input_devices()
        .map_err(|e| DomainError::AudioDeviceUnavailable(format!("input_devices: {e}")))?;

    let mut monitors = Vec::new();
    for dev in devices {
        let name = dev.name().unwrap_or_else(|_| "<unnamed>".to_string());
        if is_monitor_device(&name) {
            monitors.push(DeviceInfo {
                id: name.clone(),
                is_default: monitors.is_empty(),
                name,
            });
        }
    }

    if monitors.is_empty() {
        return Err(DomainError::AudioDeviceUnavailable(
            "no PulseAudio monitor sources found — is PulseAudio or PipeWire running?".into(),
        ));
    }

    Ok(monitors)
}

/// Find the best monitor device: prefer one whose name matches
/// `device_id`, else fall back to the first monitor source.
fn pick_monitor_device(host: &cpal::Host, device_id: Option<&str>) -> Result<Device, DomainError> {
    let devices = host
        .input_devices()
        .map_err(|e| DomainError::AudioDeviceUnavailable(format!("input_devices: {e}")))?;

    let mut fallback: Option<Device> = None;
    for dev in devices {
        let name = dev.name().unwrap_or_default();
        if !is_monitor_device(&name) {
            continue;
        }
        if let Some(target) = device_id {
            if name == target {
                return Ok(dev);
            }
        }
        if fallback.is_none() {
            fallback = Some(dev);
        }
    }

    fallback.ok_or_else(|| {
        DomainError::AudioDeviceUnavailable(
            "no PulseAudio monitor sources found — is PulseAudio or PipeWire running?".into(),
        )
    })
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

    let device = pick_monitor_device(&host, spec.device_id.as_deref())?;
    let device_name = device.name().unwrap_or_else(|_| "<unnamed>".to_string());
    let supported = pick_config(&device, spec.preferred_format)?;
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
        "starting PulseAudio monitor capture"
    );

    let (tx, rx) = mpsc::channel::<AudioFrame>(CHANNEL_CAPACITY_HINT);
    let (stop_tx, stop_rx) = oneshot::channel::<()>();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), DomainError>>();

    let join = thread::Builder::new()
        .name("echo-audio-pulse-monitor".into())
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

/// Choose the config closest to `preferred` from the device's input
/// configurations.
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
            "monitor device exposes no input configurations".into(),
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
            "no input config with {chosen_channels} channels"
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
    let err_fn = |err: cpal::StreamError| error!(error = %err, "PulseAudio monitor stream error");

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
                None,
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
                None,
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
                None,
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
    debug!("PulseAudio monitor stream live; parking thread until stop signal");

    let _ = stop_rx.blocking_recv();
    drop(stream);
    info!("PulseAudio monitor capture stopped");
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
        warn!("audio frame dropped — consumer is slower than PulseAudio monitor capture");
    }
}
