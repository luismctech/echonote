//! Audio capture port.
//!
//! Defines how the application layer drives platform-specific capture
//! engines. Concrete implementations live in [`echo_audio`] and are
//! compiled conditionally per target OS.
//!
//! ## Design
//!
//! The port is split in three concepts:
//!
//! - [`AudioCapture`] — factory that lists devices and starts a stream.
//! - [`AudioStream`] — the live, ordered sequence of [`AudioFrame`]s.
//! - [`AudioFormat`] / [`AudioFrame`] — value objects shared across layers.
//!
//! Streams expose `next_frame().await -> Option<AudioFrame>` so the
//! consumer can drive backpressure. Calling [`AudioStream::stop`] (or
//! dropping the stream) releases the underlying device.
//!
//! ## Threading
//!
//! Implementations typically own a real-time OS audio thread that writes
//! into a bounded channel. The port intentionally hides this so the
//! application layer does not need to know whether capture lives on a
//! cpal callback, a CoreAudio render thread or a WASAPI loopback.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::DomainError;

/// Sample type used everywhere downstream of the capture adapter.
///
/// EchoNote standardizes on 32-bit float PCM internally. Whisper,
/// Silero VAD and the diarizer all consume `f32`. Adapters convert from
/// the device-native format (typically `i16`/`i24`/`f32`) before pushing
/// frames downstream.
pub type Sample = f32;

/// Source from which audio is captured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioSource {
    /// Default or user-selected microphone input.
    Microphone,
    /// System audio output (loopback). Requires platform-specific
    /// permissions: ScreenCaptureKit on macOS, WASAPI loopback on
    /// Windows, PulseAudio monitor on Linux.
    SystemOutput,
}

/// Sample-rate / channel configuration of a stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AudioFormat {
    /// Sampling frequency in hertz.
    pub sample_rate_hz: u32,
    /// Number of interleaved channels (1 = mono, 2 = stereo, ...).
    pub channels: u16,
}

impl AudioFormat {
    /// Format expected by Whisper and the rest of the inference stack.
    pub const WHISPER: Self = Self {
        sample_rate_hz: 16_000,
        channels: 1,
    };

    /// Returns the size in bytes of a single PCM sample.
    #[must_use]
    pub const fn bytes_per_sample(&self) -> usize {
        std::mem::size_of::<Sample>()
    }
}

/// One chunk of captured PCM samples.
///
/// Samples are interleaved when `format.channels > 1` (LRLRLR…). The
/// frame length in seconds is `samples.len() / channels / sample_rate_hz`.
#[derive(Debug, Clone)]
pub struct AudioFrame {
    /// Interleaved PCM samples.
    pub samples: Vec<Sample>,
    /// Format describing `samples`.
    pub format: AudioFormat,
    /// Monotonic timestamp of the first sample, expressed in nanoseconds
    /// since the capture started. The clock is implementation-defined
    /// but guaranteed monotonic and gap-free within a single stream.
    pub captured_at_ns: u64,
}

impl AudioFrame {
    /// Number of frames per channel.
    #[must_use]
    pub fn frames_per_channel(&self) -> usize {
        if self.format.channels == 0 {
            return 0;
        }
        self.samples.len() / self.format.channels as usize
    }

    /// Duration of the frame, derived from sample count and rate.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        if self.format.sample_rate_hz == 0 {
            return 0;
        }
        (self.frames_per_channel() as u64 * 1_000) / u64::from(self.format.sample_rate_hz)
    }
}

/// Description of an addressable input device.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// Stable identifier used as `CaptureSpec::device_id`. Implementation
    /// chooses a format (e.g. CoreAudio UID, WASAPI endpoint id).
    pub id: String,
    /// Human-readable label suitable for UI.
    pub name: String,
    /// True when the host considers this the default device for the
    /// requested [`AudioSource`].
    pub is_default: bool,
}

/// Parameters for starting a capture stream.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CaptureSpec {
    /// Which physical or logical source to read from.
    pub source: AudioSource,
    /// Optional [`DeviceInfo::id`]. `None` selects the host default.
    pub device_id: Option<String>,
    /// Preferred format. Adapters may refuse exact matches and round to
    /// the closest supported configuration; the resulting format is
    /// reported through the first frame.
    pub preferred_format: AudioFormat,
}

impl CaptureSpec {
    /// Default microphone capture in Whisper-compatible format.
    #[must_use]
    pub fn default_microphone() -> Self {
        Self {
            source: AudioSource::Microphone,
            device_id: None,
            preferred_format: AudioFormat::WHISPER,
        }
    }
}

/// Live capture session. Yields frames in order and releases the device
/// when [`AudioStream::stop`] is called or the stream is dropped.
#[async_trait]
pub trait AudioStream: Send {
    /// Format of the underlying device. Available after [`AudioCapture::start`]
    /// returns; may differ from the [`CaptureSpec::preferred_format`].
    fn format(&self) -> AudioFormat;

    /// Returns the next captured frame, or `None` when the stream ends
    /// (device removed, stop requested, or an unrecoverable error).
    async fn next_frame(&mut self) -> Option<AudioFrame>;

    /// Stops the stream eagerly. Idempotent. The next call to
    /// [`AudioStream::next_frame`] returns `None`.
    async fn stop(&mut self) -> Result<(), DomainError>;
}

/// Factory for [`AudioStream`]s.
#[async_trait]
pub trait AudioCapture: Send + Sync {
    /// Lists the input devices the host exposes for `source`.
    async fn list_devices(&self, source: AudioSource) -> Result<Vec<DeviceInfo>, DomainError>;

    /// Starts capture according to `spec`. The returned stream is hot:
    /// frames are produced from the moment this future resolves.
    async fn start(&self, spec: CaptureSpec) -> Result<Box<dyn AudioStream>, DomainError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn whisper_format_is_16k_mono() {
        let f = AudioFormat::WHISPER;
        assert_eq!(f.sample_rate_hz, 16_000);
        assert_eq!(f.channels, 1);
        assert_eq!(f.bytes_per_sample(), 4);
    }

    #[test]
    fn frame_metrics_handle_typical_chunks() {
        let frame = AudioFrame {
            samples: vec![0.0; 16_000],
            format: AudioFormat::WHISPER,
            captured_at_ns: 0,
        };
        assert_eq!(frame.frames_per_channel(), 16_000);
        assert_eq!(frame.duration_ms(), 1_000);
    }

    #[test]
    fn stereo_frame_splits_samples_across_channels() {
        let frame = AudioFrame {
            samples: vec![0.0; 96_000],
            format: AudioFormat {
                sample_rate_hz: 48_000,
                channels: 2,
            },
            captured_at_ns: 0,
        };
        assert_eq!(frame.frames_per_channel(), 48_000);
        assert_eq!(frame.duration_ms(), 1_000);
    }

    #[test]
    fn empty_frame_reports_zero_duration() {
        let frame = AudioFrame {
            samples: vec![],
            format: AudioFormat::WHISPER,
            captured_at_ns: 0,
        };
        assert_eq!(frame.duration_ms(), 0);
        assert_eq!(frame.frames_per_channel(), 0);
    }
}
