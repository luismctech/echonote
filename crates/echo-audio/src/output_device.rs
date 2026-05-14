//! Default audio output device detection.
//!
//! Used by the UI to suggest Mixed mode when headphones are detected.

use cpal::traits::{DeviceTrait, HostTrait};

/// Information about the system's current default audio output device.
#[derive(Debug, Clone)]
pub struct OutputDeviceInfo {
    /// Human-readable name reported by the OS.
    pub name: String,
    /// Heuristic: `true` when the name suggests headphones / earbuds
    /// rather than built-in or external speakers.
    pub is_headphones: bool,
}

/// Query the default output device using cpal. Returns `None` when the host
/// has no output device or the device name cannot be read.
///
/// This call may briefly access the CoreAudio / WASAPI subsystem; run it
/// on a blocking thread if latency matters.
pub fn default_output_device_info() -> Option<OutputDeviceInfo> {
    let device = cpal::default_host().default_output_device()?;
    let name = device.description().ok()?.name().to_string();
    let is_headphones = looks_like_headphones(&name);
    Some(OutputDeviceInfo {
        name,
        is_headphones,
    })
}

fn looks_like_headphones(name: &str) -> bool {
    let lower = name.to_lowercase();
    lower.contains("airpod")
        || lower.contains("headphone")
        || lower.contains("headset")
        || lower.contains("earphone")
        || lower.contains("earpod")
        || lower.contains("earbuds")
        || lower.contains("jabra")
        || lower.contains("plantronics")
        || lower.contains("bose")
        || lower.contains("sennheiser")
        || lower.contains("sony wh")
        || lower.contains("sony wf")
        || lower.contains("beats")
}
