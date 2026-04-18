//! OS-specific audio capture adapters.
//!
//! Each submodule is gated by `#[cfg(target_os = "...")]` so that only the
//! host-relevant implementation is compiled. Sprint 0 starts with macOS
//! (the primary development platform); Windows and Linux follow in Phase 1.

#[cfg(target_os = "macos")]
pub mod macos;

#[cfg(target_os = "linux")]
pub mod linux;

#[cfg(target_os = "windows")]
pub mod windows;
