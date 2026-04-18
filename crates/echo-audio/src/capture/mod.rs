//! Audio capture adapters.
//!
//! [`cpal_microphone`] is the cross-platform microphone implementation used
//! on macOS, Windows and Linux during Sprint 0. The OS-specific submodules
//! exist as placeholders for **system audio (loopback)** capture, which
//! requires native APIs (ScreenCaptureKit, WASAPI loopback, PulseAudio
//! monitor) and lands later in Phase 1.

pub mod cpal_microphone;

pub use cpal_microphone::CpalMicrophoneCapture;

#[cfg(target_os = "macos")]
pub mod macos;

#[cfg(target_os = "linux")]
pub mod linux;

#[cfg(target_os = "windows")]
pub mod windows;
