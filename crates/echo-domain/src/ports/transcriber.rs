//! ASR (Automatic Speech Recognition) port.
//!
//! The application layer feeds the [`Transcriber`] with PCM samples in
//! the canonical format ([`AudioFormat::WHISPER`]: 16 kHz mono `f32`)
//! and gets back a [`Transcript`] composed of [`Segment`]s.
//!
//! ## Preconditions
//!
//! Adapters MAY reject input that is not 16 kHz mono `f32`. Resampling
//! is the caller's responsibility (see `echo_audio::preprocess::resample`).
//! This keeps the port tight and avoids hidden CPU work behind
//! `transcribe()`.
//!
//! [`Segment`]: crate::entities::segment::Segment

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::entities::segment::Segment;
use crate::ports::audio::Sample;
use crate::DomainError;

/// Result of a transcription pass over a chunk of PCM samples.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Transcript {
    /// Decoded segments, in chronological order.
    pub segments: Vec<Segment>,
    /// ISO-639-1 language code detected (or echoed back from
    /// [`TranscribeOptions::language`]). `None` if the backend did not
    /// report it.
    pub language: Option<String>,
    /// Total duration of the audio that was transcribed, in
    /// milliseconds. Useful for benchmarking and progress UI.
    pub duration_ms: u32,
}

impl Transcript {
    /// Concatenated text of every segment, separated by single spaces.
    /// Convenience for early UX and tests.
    #[must_use]
    pub fn full_text(&self) -> String {
        self.segments
            .iter()
            .map(|s| s.text.trim())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Knobs the application layer can pass to the ASR backend.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TranscribeOptions {
    /// ISO-639-1 language hint (e.g. `"en"`, `"es"`). `None` lets the
    /// backend auto-detect.
    pub language: Option<String>,
    /// Optional context prepended to the audio. Whisper uses it to bias
    /// the decoder towards the meeting topic.
    pub initial_prompt: Option<String>,
    /// When `true`, translate the audio to English instead of
    /// transcribing in the source language.
    pub translate: bool,
    /// Maximum number of decoder threads. `None` lets the adapter
    /// pick a sensible default (typically `num_cpus / 2`).
    pub threads: Option<u16>,
}

/// Async ASR port. Adapters wrap whisper.cpp, faster-whisper or any
/// future engine and return a normalized [`Transcript`].
#[async_trait]
pub trait Transcriber: Send + Sync {
    /// Transcribes an in-memory chunk of PCM samples.
    ///
    /// Adapters MAY return [`DomainError::AudioFormatUnsupported`] if
    /// the caller did not resample to the canonical Whisper format
    /// ([`crate::AudioFormat::WHISPER`]).
    async fn transcribe(
        &self,
        samples: &[Sample],
        options: &TranscribeOptions,
    ) -> Result<Transcript, DomainError>;
}
