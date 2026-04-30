//! Hardware profiling for model recommendations.

use serde::Serialize;

/// Hardware profile detected at runtime.
#[derive(Debug, Clone, Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct HardwareProfile {
    /// Total physical RAM in bytes.
    pub total_ram_bytes: u64,
    /// Currently available (free) RAM in bytes.
    pub available_ram_bytes: u64,
    /// Number of logical CPU cores (threads).
    pub cpu_cores: usize,
    /// CPU brand string (e.g. "Apple M2 Pro").
    pub cpu_brand: String,
    /// GPU name (Metal device name on macOS, "Unknown" elsewhere).
    pub gpu_name: String,
    /// Recommended max GPU working set in bytes (Metal on macOS).
    /// On other platforms this is 0.
    pub gpu_max_working_set: u64,
    /// Whether the device has unified memory (CPU + GPU share RAM).
    pub unified_memory: bool,
}

/// Detect system hardware. Called once when the user opens the Models panel.
#[tauri::command]
#[specta::specta]
pub fn get_hardware_profile() -> HardwareProfile {
    use sysinfo::System;

    let mut sys = System::new();
    sys.refresh_memory();
    sys.refresh_cpu_all();

    let cpu_brand = sys
        .cpus()
        .first()
        .map(|c| c.brand().to_string())
        .unwrap_or_else(|| "Unknown CPU".to_string());

    let (gpu_name, gpu_max_working_set, unified_memory) = detect_gpu();

    HardwareProfile {
        total_ram_bytes: sys.total_memory(),
        available_ram_bytes: sys.available_memory(),
        cpu_cores: sys.cpus().len(),
        cpu_brand,
        gpu_name,
        gpu_max_working_set,
        unified_memory,
    }
}

#[cfg(target_os = "macos")]
fn detect_gpu() -> (String, u64, bool) {
    match metal::Device::system_default() {
        Some(device) => {
            let name = device.name().to_string();
            let max_working_set = device.recommended_max_working_set_size();
            let unified = device.has_unified_memory();
            (name, max_working_set, unified)
        }
        None => ("No Metal GPU".to_string(), 0, false),
    }
}

#[cfg(not(target_os = "macos"))]
fn detect_gpu() -> (String, u64, bool) {
    ("Unknown".to_string(), 0, false)
}
