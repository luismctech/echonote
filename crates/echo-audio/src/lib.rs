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
pub mod preprocess;
