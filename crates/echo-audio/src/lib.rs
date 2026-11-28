//! # echo-audio
//!
//! Audio capture and preprocessing for EchoNote. Owns the platform-specific
//! adapters that implement [`echo_domain::ports::audio`], a 30-second ring
//! buffer, a resampler to 16 kHz mono and a Silero-based VAD.
//!
//! Unsafe code is allowed in this crate because several native bindings
//! (cpal, ScreenCaptureKit, WASAPI) expose FFI surface. Unsafe blocks must
//! be documented and localized.
//!
//! Layout reflects `docs/ARCHITECTURE.md` §5.1:
//!
//! ```text
//! echo-audio/
//! ├── capture/   OS-specific adapters behind #[cfg(target_os = "...")]
//! ├── preprocess/ resample, denoise, VAD
//! └── buffer.rs  lock-free ring buffer
//! ```

#![warn(rust_2018_idioms, clippy::all)]

pub mod buffer;
pub mod capture;
pub mod output_device;
pub mod permission;
pub mod preprocess;
pub mod sink;

pub use buffer::SamplePool;
#[cfg(target_os = "macos")]
pub use capture::{CoreAudioTapCapture, ScreenCaptureKitCapture, SYSTEM_OUTPUT_DEVICE_ID};
pub use capture::{CpalMicrophoneCapture, MixControls, MixedStream, RoutingAudioCapture};
pub use output_device::{default_output_device_info, OutputDeviceInfo};
pub use permission::{request_system_audio_permission, SystemAudioPermission};
pub use preprocess::resample::{
    resample_to_whisper, FormatConverter, ResampleError, RubatoResamplerAdapter, StreamResampler,
    WHISPER_SAMPLE_RATE,
};
pub use preprocess::silero_vad::{
    SileroVad, SileroVadConfig, SILERO_FRAME_SAMPLES, SILERO_SAMPLE_RATE,
};
pub use preprocess::vad::{rms, EnergyVad, VadConfig};
pub use sink::{WavSink, WriteOptions};
