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

/// One speaker-attributed span within an audio chunk.
///
/// Emitted by [`Diarizer::assign_spans`]. Two `SpeakerSpan`s returned
/// from the same call whose `start_sample`/`end_sample` ranges overlap
/// represent **simultaneous talkers** — the caller must keep both
/// rather than collapsing to a single dominant speaker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeakerSpan {
    /// Speaker the diarizer assigned to this span.
    pub speaker_id: SpeakerId,
    /// Arrival-order slot of the assigned speaker. Mirrored from the
    /// diarizer's [`Diarizer::speakers`] view; included so callers
    /// don't have to walk the speakers list per span just to colour
    /// a chip in the UI.
    pub slot: u32,
    /// Inclusive sample offset, relative to the first sample of the
    /// chunk passed to `assign_spans`.
    pub start_sample: usize,
    /// Exclusive sample offset. Always `>= start_sample`.
    pub end_sample: usize,
}

impl SpeakerSpan {
    /// Length of the span in samples. `0` for degenerate spans.
    #[must_use]
    pub fn len_samples(&self) -> usize {
        self.end_sample.saturating_sub(self.start_sample)
    }

    /// Overlap (in samples) between this span and the half-open
    /// `[start, end)` window. `0` when disjoint.
    #[must_use]
    pub fn overlap_samples(&self, start: usize, end: usize) -> usize {
        let lo = self.start_sample.max(start);
        let hi = self.end_sample.min(end);
        hi.saturating_sub(lo)
    }
}

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
    ///
    /// This is the legacy, single-speaker entry point retained for
    /// adapters that don't model overlap. The streaming pipeline
    /// prefers [`Diarizer::assign_spans`] so it can attribute
    /// simultaneous talkers to different Whisper segments.
    async fn assign(&mut self, samples: &[Sample]) -> Result<Option<SpeakerId>, DomainError>;

    /// Process one audio chunk and return every speaker-attributed
    /// span within it.
    ///
    /// Two overlapping spans encode simultaneous speech. An empty vec
    /// means the diarizer chose not to embed (chunk too short, low
    /// confidence, etc.) — same semantics as `Ok(None)` from
    /// [`Diarizer::assign`].
    ///
    /// Default impl wraps [`Diarizer::assign`] in a single full-chunk
    /// span so adapters without overlap support keep working. Real
    /// diarizers with a sub-chunk segmenter override this to emit one
    /// span per speaker-homogeneous sub-region.
    async fn assign_spans(&mut self, samples: &[Sample]) -> Result<Vec<SpeakerSpan>, DomainError> {
        let Some(id) = self.assign(samples).await? else {
            return Ok(Vec::new());
        };
        let slot = self
            .speakers()
            .into_iter()
            .find(|s| s.id == id)
            .map(|s| s.slot)
            .unwrap_or(0);
        Ok(vec![SpeakerSpan {
            speaker_id: id,
            slot,
            start_sample: 0,
            end_sample: samples.len(),
        }])
    }

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

    /// Offline refinement pass executed once at the end of a session.
    ///
    /// Implementations that accumulate per-chunk embeddings during a
    /// session can re-cluster them in bulk here — full-session means
    /// are less noisy than the streaming EMA centroids, so a pair of
    /// speakers the online pass over-split typically collapses back
    /// into one. Returns `(absorbed_id, surviving_id)` pairs the
    /// streaming pipeline can apply to already-persisted segments.
    ///
    /// Default impl is a no-op (empty vec) so adapters without
    /// offline state — fakes, stubs, the in-memory test diarizer —
    /// satisfy the trait without extra code.
    fn refine_offline(&self) -> Vec<(SpeakerId, SpeakerId)> {
        Vec::new()
    }
}
