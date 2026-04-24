//! Audio capture adapters.
//!
//! [`cpal_microphone`] is the cross-platform microphone implementation
//! used everywhere; the OS-specific submodules add **system audio
//! (loopback)** capture using native APIs:
//!
//! - **macOS**: [`macos::ScreenCaptureKitCapture`] via ScreenCaptureKit.
//! - **Windows**: [`windows::WasapiLoopbackCapture`] via WASAPI loopback.
//! - **Linux**: [`linux::PulseMonitorCapture`] via PulseAudio monitor sources.

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
#[cfg(target_os = "linux")]
pub use linux::{PulseMonitorCapture, SYSTEM_OUTPUT_DEVICE_ID as LINUX_SYSTEM_OUTPUT_DEVICE_ID};

#[cfg(target_os = "windows")]
pub mod windows;
#[cfg(target_os = "windows")]
pub use windows::{
    WasapiLoopbackCapture, SYSTEM_OUTPUT_DEVICE_ID as WINDOWS_SYSTEM_OUTPUT_DEVICE_ID,
};
