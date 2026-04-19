//! # echo-asr
//!
//! Automatic Speech Recognition adapter. Wraps whisper.cpp through
//! [`whisper-rs`] and implements [`echo_domain::Transcriber`].
//!
//! ## Backends
//!
//! - macOS: built with `metal` so the GPU does the heavy lifting.
//! - Linux/Windows: pure CPU build. Acceleration features (CUDA,
//!   Vulkan, OpenBLAS) land later when those platforms become primary.
//!
//! ## Threading model
//!
//! `WhisperState::full` is a long-running, blocking call. The adapter
//! offloads it to [`tokio::task::spawn_blocking`] so the async runtime
//! stays responsive. The [`WhisperContext`] is `Send + Sync` and held
//! once per process; new [`WhisperState`]s are cheap and created per
//! call so concurrent transcriptions do not contend on a single state.

#![warn(rust_2018_idioms, clippy::all)]

pub mod whisper_cpp;

pub use whisper_cpp::WhisperCppTranscriber;
