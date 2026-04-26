//! Speaker diarization port.
//!
//! The diarizer answers "*who* spoke this chunk?". It does not split
//! audio into utterances — that's the VAD's job — and it does not
//! transcribe — that's whisper's. Given a contiguous span of mono
//! 16 kHz audio that the VAD already classified as voiced, it
//! returns a stable [`SpeakerId`] keyed off an internal cluster of
//! voice embeddings.
//!
//! ## Per-track contract (Sprint 1)
//!
//! Each `Diarizer` instance owns the speaker space for a single
//! audio source (microphone *or* system output). Cross-track
//! unification — recognizing that "Alice" on the system side is
//! the same person whose mic we're capturing — is a Sprint 2
//! follow-up. Callers should construct one diarizer per track and
//! merge results downstream.
//!
//! ## State and lifecycle
//!
//! Diarization is inherently stateful: every assignment grows the
//! cluster, may create a new speaker, or refines an existing
//! centroid. Callers must therefore feed chunks chronologically
//! through a single instance, and call [`Diarizer::reset`] before
//! reusing it across independent meetings.
//!
//! Adapter implementations live in `echo-diarize`.

use async_trait::async_trait;

use crate::{DomainError, Sample, Speaker, SpeakerId};

/// Stateful speaker assignment over a chunk-by-chunk audio stream.
///
/// Implementations must be `Send + Sync` so the streaming pipeline
/// can hold them behind `Arc<dyn Diarizer>` while moving them across
/// async tasks.
#[async_trait]
pub trait Diarizer: Send + Sync {
    /// Sample rate the embedder expects. Mixing rates is a bug —
    /// resample upstream with [`crate::Resampler`].
    fn sample_rate_hz(&self) -> u32;

    /// Process one audio chunk and return the speaker it belongs to.
    ///
    /// Returns `Ok(None)` when the chunk is too short to embed, or
    /// when the embedder reports a low-confidence frame (e.g. a
    /// trailing silence the VAD let through). Callers that get
    /// `None` should keep the chunk unlabelled rather than treating
    /// it as a new speaker.
    async fn assign(&mut self, samples: &[Sample]) -> Result<Option<SpeakerId>, DomainError>;

    /// Snapshot of the speakers identified so far in this session,
    /// in arrival order. The slot indices are stable across calls;
    /// the labels reflect any [`Diarizer::rename`] calls already
    /// applied.
    fn speakers(&self) -> Vec<Speaker>;

    /// Apply a user-supplied label to an existing speaker. Returns
    /// `Ok(false)` when the id is not part of this diarizer's
    /// internal cluster (already evicted, never created, …) — the
    /// caller decides whether that's an error to surface.
    fn rename(&mut self, id: SpeakerId, label: &str) -> Result<bool, DomainError>;

    /// Drop all internal state (centroids, embedder LSTM, counters).
    /// Use between independent meetings instead of allocating a
    /// fresh instance — adapters with heavy model state will reuse
    /// loaded weights.
    fn reset(&mut self);
}
