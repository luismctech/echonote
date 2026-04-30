//! Speaker segmentation port.
//!
//! A `Segmenter` detects **where** speaker changes happen inside an
//! audio chunk at a finer granularity than the chunk itself.
//!
//! ## How it fits in the pipeline
//!
//! The current streaming pipeline uses fixed 5-second chunks as the
//! basic unit of diarization. When two speakers alternate within a
//! single chunk the embedder receives a mixed representation, which
//! degrades cluster quality. A segmenter solves this by splitting each
//! chunk into speaker-homogeneous sub-regions *before* embedding:
//!
//! ```text
//! (old)  audio chunk  →  embed chunk  →  cluster
//! (new)  audio chunk  →  segment  →  embed each sub-region  →  cluster
//! ```
//!
//! ## Local vs global speakers
//!
//! The segmenter produces *local* speaker indices (`0`, `1`, `2`, …)
//! that are meaningful only within the current chunk. Mapping local
//! speaker `0` in chunk N to a global `SpeakerId` is the job of the
//! embedder + cluster layer that sits on top.
//!
//! ## Format
//!
//! All adapters consume mono `f32` PCM at the rate declared by
//! [`Segmenter::sample_rate_hz`], matching [`crate::Sample`].

use crate::{DomainError, Sample};

/// A contiguous span of audio dominated by a single local speaker.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerSegment {
    /// Start offset in samples from the start of the audio passed to
    /// [`Segmenter::segment`].
    pub start_sample: usize,
    /// End offset in samples (exclusive). Always `> start_sample`.
    pub end_sample: usize,
    /// Zero-based local speaker index within the chunk. Two segments
    /// with the same `local_speaker` value belong to the same voice
    /// inside this chunk; cross-chunk identity is resolved by the
    /// cluster layer.
    pub local_speaker: u8,
}

impl SpeakerSegment {
    /// Length of this segment in samples.
    #[must_use]
    pub fn len_samples(&self) -> usize {
        self.end_sample.saturating_sub(self.start_sample)
    }

    /// `true` when the segment contains no samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.end_sample <= self.start_sample
    }
}

/// Speaker boundary detector.
///
/// Implementations split an audio chunk into speaker-homogeneous
/// sub-regions and return one [`SpeakerSegment`] per contiguous span.
///
/// Segments may overlap when multiple speakers are simultaneously
/// active (the model detects overlapping speech). Non-overlapping
/// implementations should always return disjoint segments.
///
/// Methods are synchronous: neural segmenters run a single feed-
/// forward pass per chunk and do not benefit from async plumbing.
pub trait Segmenter: Send + Sync {
    /// Sample rate the model was trained on. Mixing rates silently
    /// degrades accuracy — convert with [`crate::Resampler`] first.
    fn sample_rate_hz(&self) -> u32;

    /// Maximum number of *simultaneous* speakers the model can detect
    /// per chunk. Pyannote-segmentation-3.0 supports up to 3.
    fn max_local_speakers(&self) -> u8;

    /// Detect speaker boundaries in `samples` and return a list of
    /// speaker-homogeneous segments in chronological order.
    ///
    /// Returns an empty `Vec` when the chunk is too short or entirely
    /// silent. Returns [`DomainError::DiarizationFailed`] on inference
    /// errors.
    fn segment(&mut self, samples: &[Sample]) -> Result<Vec<SpeakerSegment>, DomainError>;
}
