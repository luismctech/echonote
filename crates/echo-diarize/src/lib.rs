//! # echo-diarize
//!
//! Speaker diarization for EchoNote: cluster an audio track's voice
//! segments by speaker identity using local ONNX embeddings followed by
//! agglomerative clustering. Implements the `Diarizer` port.
//!
//! Lands in Sprint 2 per `docs/DEVELOPMENT_PLAN.md` §5.3.

#![warn(rust_2018_idioms, clippy::all)]
