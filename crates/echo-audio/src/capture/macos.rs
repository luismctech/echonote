//! macOS system-audio (loopback) capture adapter backed by `ScreenCaptureKit`.
//!
//! ScreenCaptureKit is the modern Apple-supported way to capture system
//! audio without virtual devices (BlackHole, Loopback, etc.). It became
//! available in macOS 12.3 and gained audio-only support in 13.0
//! (Ventura). EchoNote targets macOS 13+ for system audio, in line with
//! `docs/ARCHITECTURE.md` §2.
//!
//! ## Why not `cpal`?
//!
//! cpal exposes only physical input devices on CoreAudio. To capture the
//! mix that any app sends to the speakers, the OS itself has to route it
//! through ScreenCaptureKit, which gives a privacy-aware, permission-
//! gated audio buffer. The user grants Screen Recording permission once,
//! after which the app receives PCM samples for system output (and,
//! optionally, individual app audio when the per-app filter is used).
//!
//! ## Threading
//!
//! `SCStream`'s output handler runs on a Grand Central Dispatch queue
//! owned by the framework. We capture only `Send` values inside that
//! handler — the `mpsc::Sender<AudioFrame>` half — and immediately
//! convert the raw PCM bytes into an owned [`AudioFrame`] before sending
//! it to the async runtime.
//!
//! `SCStream` itself is moved onto a dedicated `std::thread` (mirroring
//! [`crate::capture::cpal_microphone`]) and parked until a stop signal
//! arrives, so the [`AudioStream`] handle returned to the application
//! layer is `Send` even though the framework callback is not.
//!
//! ## Format
//!
//! `with_captures_audio(true)` always emits 32-bit float interleaved PCM
//! at the rate / channel count configured on [`SCStreamConfiguration`].
//! We default to **48 kHz / stereo**: this matches Apple's preferred
//! mixer rate, is what every conferencing app (Zoom, Teams, Meet)
//! produces, and is cheap to downsample to the 16 kHz mono Whisper
//! expects (see [`crate::preprocess::resample`]).

use std::sync::mpsc as sync_mpsc;
use std::thread;
use std::time::Instant;

use async_trait::async_trait;
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, error, info, warn};

use echo_domain::{
    AudioCapture, AudioFormat, AudioFrame, AudioSource, AudioStream, CaptureSpec, DeviceInfo,
    DomainError,
};

use screencapturekit::prelude::*;
use screencapturekit::shareable_content::SCShareableContent;

/// Default sample rate requested from ScreenCaptureKit. 48 kHz is the
/// CoreAudio mixer's native rate on every modern Mac, so picking it
/// avoids a kernel-side resample before our own [`crate::preprocess::resample`]
/// step.
const SCK_SAMPLE_RATE: u32 = 48_000;

/// Default channel count. Stereo because ScreenCaptureKit will refuse
/// mono on most configurations and the downstream resampler folds to
/// mono before Whisper anyway.
const SCK_CHANNELS: u16 = 2;

/// Mirrors the cpal adapter's bound. Sized for ~5 s of 48 kHz stereo
/// f32 (~2 MB) before backpressure warnings fire.
const CHANNEL_CAPACITY_HINT: usize = 512;

/// Synthetic device id reported by [`Self::list_devices`]. ScreenCaptureKit
/// does not expose enumerable "system output" devices: there is one
/// logical loopback per display. We surface a stable identifier so the
/// UI can offer a single, named option without having to special-case
/// `None`.
pub const SYSTEM_OUTPUT_DEVICE_ID: &str = "macos:screencapturekit:default-display";

/// macOS system-audio capture adapter.
///
/// Cheap to construct; touches no Apple APIs until [`Self::start`] or
/// [`Self::list_devices`] is called.
#[derive(Debug, Default, Clone)]
pub struct ScreenCaptureKitCapture;

impl ScreenCaptureKitCapture {
    /// Constructs a new adapter. No system calls are made.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl AudioCapture for ScreenCaptureKitCapture {
    async fn list_devices(&self, source: AudioSource) -> Result<Vec<DeviceInfo>, DomainError> {
        if source != AudioSource::SystemOutput {
            return Err(DomainError::AudioDeviceUnavailable(format!(
                "{source:?} is not handled by the ScreenCaptureKit adapter; \
                 use the cpal microphone adapter instead"
            )));
        }

        // Probing SCShareableContent triggers the screen-recording
        // permission prompt the first time it runs and synchronously
        // talks to WindowServer. Keep it off the async runtime.
        tokio::task::spawn_blocking(probe_default_display)
            .await
            .map_err(|e| DomainError::AudioCaptureFailed(format!("device probe join: {e}")))?
    }

    async fn start(&self, spec: CaptureSpec) -> Result<Box<dyn AudioStream>, DomainError> {
        if spec.source != AudioSource::SystemOutput {
            return Err(DomainError::AudioDeviceUnavailable(format!(
                "{:?} is not handled by the ScreenCaptureKit adapter; \
                 use the cpal microphone adapter instead",
                spec.source
            )));
        }

        // SCK setup talks to a Mach service and may block briefly while
        // the user authorizes screen recording the first time. Offload
        // to the blocking pool but report errors synchronously.
        let started = tokio::task::spawn_blocking(move || start_capture(&spec))
            .await
            .map_err(|e| DomainError::AudioCaptureFailed(format!("start join: {e}")))??;

        Ok(Box::new(started))
    }
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

fn probe_default_display() -> Result<Vec<DeviceInfo>, DomainError> {
    let content = SCShareableContent::get().map_err(|e| {
        DomainError::AudioDeviceUnavailable(format!(
            "ScreenCaptureKit unavailable (Screen Recording permission?): {e}"
        ))
    })?;

    if content.displays().is_empty() {
        return Err(DomainError::AudioDeviceUnavailable(
            "ScreenCaptureKit reported zero displays — Screen Recording \
             permission is likely missing"
                .into(),
        ));
    }

    Ok(vec![DeviceInfo {
        id: SYSTEM_OUTPUT_DEVICE_ID.to_string(),
        name: "System Output (ScreenCaptureKit)".to_string(),
        is_default: true,
    }])
}

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
                    DomainError::AudioCaptureFailed(format!("SCK thread panicked: {e:?}"))
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
    // Resolve a target display. Per-app filtering will land in a
    // follow-up issue; the MVP captures the full mix of the primary
    // display, which already isolates calls / browser tabs once paired
    // with VAD + diarization downstream.
    let content = SCShareableContent::get().map_err(|e| {
        DomainError::AudioDeviceUnavailable(format!(
            "ScreenCaptureKit unavailable (Screen Recording permission?): {e}"
        ))
    })?;

    let display = content.displays().into_iter().next().ok_or_else(|| {
        DomainError::AudioDeviceUnavailable(
            "no shareable display — grant Screen Recording in System Settings".into(),
        )
    })?;

    let filter = SCContentFilter::create()
        .with_display(&display)
        .with_excluding_windows(&[])
        .build();

    // We honor the caller's requested sample rate / channel count when
    // possible. Whisper-target preferred_format (16 kHz mono) is below
    // SCK's supported range, so we fall back to 48 kHz stereo and rely
    // on the resampler downstream.
    let (sample_rate, channels) = effective_format(spec.preferred_format);

    // SCStreamConfiguration's audio setters take `impl Into<i32>`. The
    // domain ports use unsigned types because negative rates / channel
    // counts are nonsense; we widen-then-cast here at the boundary.
    #[allow(clippy::cast_possible_wrap)]
    let config = SCStreamConfiguration::new()
        .with_captures_audio(true)
        .with_sample_rate(sample_rate as i32)
        .with_channel_count(i32::from(channels));

    let format = AudioFormat {
        sample_rate_hz: sample_rate,
        channels,
    };

    info!(
        requested.rate = spec.preferred_format.sample_rate_hz,
        requested.ch = spec.preferred_format.channels,
        actual.rate = format.sample_rate_hz,
        actual.ch = format.channels,
        "starting ScreenCaptureKit system-audio capture"
    );

    let (tx, rx) = mpsc::channel::<AudioFrame>(CHANNEL_CAPACITY_HINT);
    let (stop_tx, stop_rx) = oneshot::channel::<()>();
    let (ready_tx, ready_rx) = sync_mpsc::channel::<Result<(), DomainError>>();

    // Move SCStream construction onto a dedicated thread so the start
    // future can return a `Send` handle. SCStream's output handler runs
    // on its own dispatch queue regardless, so this thread just owns
    // lifetime + receives the stop signal.
    let join = thread::Builder::new()
        .name("echo-audio-sck".into())
        .spawn(move || {
            run_capture_thread(filter, config, format, tx, stop_rx, ready_tx);
        })
        .map_err(|e| DomainError::AudioCaptureFailed(format!("spawn SCK thread: {e}")))?;

    match ready_rx.recv() {
        Ok(Ok(())) => Ok(StartedStream {
            rx,
            stop: Some(stop_tx),
            format,
            join: Some(join),
        }),
        Ok(Err(e)) => Err(e),
        Err(e) => Err(DomainError::AudioCaptureFailed(format!(
            "SCK thread died before signalling readiness: {e}"
        ))),
    }
}

/// Maps the caller's preferred format onto something ScreenCaptureKit
/// will accept. The framework rejects rates below 16 kHz and prefers
/// stereo on most configurations, so we negotiate towards 48 kHz / 2 ch
/// when the request is below threshold.
fn effective_format(preferred: AudioFormat) -> (u32, u16) {
    let rate = preferred.sample_rate_hz.clamp(16_000, 48_000);
    let rate = if rate < 24_000 { SCK_SAMPLE_RATE } else { rate };
    let channels = match preferred.channels {
        0 => SCK_CHANNELS,
        n => n.min(2),
    };
    (rate, channels)
}

/// Audio output handler. Runs on a Grand Central Dispatch queue owned
/// by `SCStream`. Captures only `Send` data and pushes a freshly-owned
/// `AudioFrame` per sample buffer into the async channel.
struct AudioOutputHandler {
    tx: mpsc::Sender<AudioFrame>,
    format: AudioFormat,
    start: Instant,
}

impl SCStreamOutputTrait for AudioOutputHandler {
    fn did_output_sample_buffer(&self, sample: CMSampleBuffer, of_type: SCStreamOutputType) {
        // Single handler bound to multiple output types is a no-op for
        // anything that isn't the audio mix.
        if of_type != SCStreamOutputType::Audio {
            return;
        }

        let Some(buffer_list) = sample.audio_buffer_list() else {
            warn!("SCK audio sample carried no buffer list");
            return;
        };

        // ScreenCaptureKit always delivers float32 linear PCM. A single
        // AudioBufferList may contain one interleaved buffer (stereo)
        // or N planar buffers (one per channel). We coalesce both into
        // an interleaved Vec<f32> matching `self.format.channels`.
        let mut samples: Vec<f32> = Vec::new();
        for buf in buffer_list.iter() {
            let bytes = buf.data();
            // Trust the framework's claim of float32. Truncating div is
            // intentional — partial trailing samples are discarded.
            let chunk_len = bytes.len() / std::mem::size_of::<f32>();
            samples.reserve(chunk_len);
            for i in 0..chunk_len {
                let offset = i * 4;
                // SAFETY: `bytes` is at least `chunk_len * 4` bytes
                // long (checked via integer division above) and SCK
                // guarantees the data pointer is aligned to f32.
                let raw = u32::from_le_bytes([
                    bytes[offset],
                    bytes[offset + 1],
                    bytes[offset + 2],
                    bytes[offset + 3],
                ]);
                samples.push(f32::from_bits(raw));
            }
        }

        if samples.is_empty() {
            return;
        }

        let captured_at_ns = self.start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;

        let frame = AudioFrame {
            samples,
            format: self.format,
            captured_at_ns,
        };

        if let Err(mpsc::error::TrySendError::Full(_)) = self.tx.try_send(frame) {
            warn!("SCK frame dropped — consumer slower than capture");
        }
    }
}

fn run_capture_thread(
    filter: SCContentFilter,
    config: SCStreamConfiguration,
    format: AudioFormat,
    tx: mpsc::Sender<AudioFrame>,
    stop_rx: oneshot::Receiver<()>,
    ready_tx: sync_mpsc::Sender<Result<(), DomainError>>,
) {
    let start = Instant::now();
    let handler = AudioOutputHandler { tx, format, start };

    let mut stream = SCStream::new(&filter, &config);
    stream.add_output_handler(handler, SCStreamOutputType::Audio);

    if let Err(e) = stream.start_capture() {
        let _ = ready_tx.send(Err(DomainError::AudioCaptureFailed(format!(
            "SCStream::start_capture: {e}"
        ))));
        return;
    }

    let _ = ready_tx.send(Ok(()));
    debug!("SCK system-audio capture live; parking thread until stop signal");

    let _ = stop_rx.blocking_recv();

    if let Err(e) = stream.stop_capture() {
        error!(error = %e, "SCStream::stop_capture failed");
    }
    drop(stream);
    info!("SCK system-audio capture stopped");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn effective_format_promotes_whisper_rate_to_48k() {
        // Whisper asks for 16 kHz mono. SCK refuses anything below
        // 24 kHz, so we promote to the framework default (48 kHz) but
        // honor the caller's mono preference.
        let (rate, ch) = effective_format(AudioFormat::WHISPER);
        assert_eq!(rate, SCK_SAMPLE_RATE);
        assert_eq!(ch, 1);
    }

    #[test]
    fn effective_format_picks_stereo_when_unspecified() {
        let (rate, ch) = effective_format(AudioFormat {
            sample_rate_hz: 0,
            channels: 0,
        });
        assert_eq!(rate, SCK_SAMPLE_RATE);
        assert_eq!(ch, SCK_CHANNELS);
    }

    #[test]
    fn effective_format_keeps_24k_and_above() {
        let (rate, ch) = effective_format(AudioFormat {
            sample_rate_hz: 24_000,
            channels: 1,
        });
        assert_eq!(rate, 24_000);
        assert_eq!(ch, 1);
    }

    #[test]
    fn effective_format_caps_above_48k() {
        let (rate, ch) = effective_format(AudioFormat {
            sample_rate_hz: 96_000,
            channels: 6,
        });
        assert_eq!(rate, 48_000);
        assert_eq!(ch, 2);
    }

    #[test]
    fn effective_format_replaces_zero_channels() {
        let (_rate, ch) = effective_format(AudioFormat {
            sample_rate_hz: 48_000,
            channels: 0,
        });
        assert_eq!(ch, SCK_CHANNELS);
    }

    #[tokio::test]
    async fn microphone_source_is_rejected() {
        let cap = ScreenCaptureKitCapture::new();
        let err = cap
            .list_devices(AudioSource::Microphone)
            .await
            .expect_err("must reject microphone source");
        assert!(matches!(err, DomainError::AudioDeviceUnavailable(_)));
    }

    #[tokio::test]
    async fn microphone_start_is_rejected() {
        let cap = ScreenCaptureKitCapture::new();
        let outcome = cap.start(CaptureSpec::default_microphone()).await;
        match outcome {
            Ok(_) => panic!("must reject microphone source"),
            Err(DomainError::AudioDeviceUnavailable(_)) => {}
            Err(other) => panic!("unexpected error variant: {other:?}"),
        }
    }
}
