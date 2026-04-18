//! # echo-app
//!
//! The application layer of EchoNote: use cases that coordinate ports
//! defined by [`echo_domain`]. Every case of use (CU-01..CU-08 in the
//! development plan) will land as a struct in [`use_cases`] with an
//! `execute` method.
//!
//! The application layer is ignorant of Tauri, React and concrete
//! adapters. The only thing it needs is a set of `Arc<dyn Port>` values
//! injected at startup by `src-tauri`.

#![forbid(unsafe_code)]
#![warn(missing_docs, rust_2018_idioms, clippy::all)]

pub mod services;
pub mod use_cases;

pub use use_cases::start_recording::{FrameSink, RecordToSink, RecordingReport};
