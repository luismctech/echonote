//! # echo-asr
//!
//! Automatic Speech Recognition adapter. Wraps whisper.cpp (via `whisper-rs`)
//! and implements the `Transcriber` port declared in `echo-domain`. Exposes
//! both streaming (5 s chunks during recording) and refinement (full-file
//! pass after stop) pipelines.
//!
//! See `docs/ARCHITECTURE.md` §3.2.5 for the rationale.

#![warn(rust_2018_idioms, clippy::all)]
