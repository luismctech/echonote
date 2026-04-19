//! Domain ports.
//!
//! A port is a trait that describes *what* the domain needs from the outside
//! world, not *how* it is provided. Concrete adapters live in the
//! corresponding infrastructure crates (`echo-audio`, `echo-asr`, ...).
//!
//! Keeping ports here frees the domain from knowing anything about whisper,
//! llama, SQLite or platform-specific audio APIs.

pub mod audio;
pub mod diarizer;
pub mod llm;
pub mod resampler;
pub mod storage;
pub mod transcriber;
pub mod vad;
