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
pub(crate) mod interleave;
pub mod mixed;
// Used by the Linux adapter; built everywhere so its tests run on every CI runner.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) mod monitor_select;
pub(crate) mod reconnect;
pub mod routing;
pub mod sample_convert;
pub mod stream_error;

pub use cpal_microphone::CpalMicrophoneCapture;
pub use mixed::{MixControls, MixedStream};
pub use routing::RoutingAudioCapture;

/// Host for the microphone and the default-output query.
///
/// On Linux this stays ALSA: cpal's PulseAudio backend is compiled in for
/// the system-audio monitor ([`linux`]) and would otherwise become cpal's
/// default host, changing the microphone names and ids users have saved.
pub(crate) fn device_host() -> cpal::Host {
    #[cfg(target_os = "linux")]
    if let Ok(host) = cpal::host_from_id(cpal::HostId::Alsa) {
        return host;
    }
    cpal::default_host()
}

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
