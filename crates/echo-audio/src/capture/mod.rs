//! Audio capture adapters.
//!
//! [`cpal_microphone`] is the cross-platform microphone implementation
//! used everywhere; the OS-specific submodules add **system audio
//! (loopback)** capture using native APIs (ScreenCaptureKit on macOS,
//! WASAPI loopback on Windows, PulseAudio monitor on Linux).
//!
//! Sprint 1 brings macOS up to functional parity with the cpal mic via
//! [`macos::ScreenCaptureKitCapture`]; the other OSes remain stubs and
//! return [`echo_domain::DomainError::AudioDeviceUnavailable`] until
//! their respective issues land.

pub mod cpal_microphone;
pub mod routing;

pub use cpal_microphone::CpalMicrophoneCapture;
pub use routing::RoutingAudioCapture;

#[cfg(target_os = "macos")]
pub mod macos;
#[cfg(target_os = "macos")]
pub use macos::{ScreenCaptureKitCapture, SYSTEM_OUTPUT_DEVICE_ID};

#[cfg(target_os = "linux")]
pub mod linux;

#[cfg(target_os = "windows")]
pub mod windows;
