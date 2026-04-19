//! Sinks that consume an [`echo_domain::AudioStream`] and persist or
//! forward the captured frames.

pub mod wav;

pub use wav::{WavSink, WriteOptions};
