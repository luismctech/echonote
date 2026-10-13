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

use std::time::Instant;

use async_trait::async_trait;

use crate::entities::streaming::SpeakerRetirement;
use crate::{DomainError, Sample, Speaker, SpeakerId};

/// Where a chunk sits in the session, passed to
/// [`Diarizer::assign_spans_at`] so adapters can timestamp what they
/// remember about it (ADR 0001, D1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ChunkContext {
    /// 0-based chunk index, as carried by `TranscriptEvent::Chunk`.
    pub chunk_index: u32,
    /// Chunk start in session milliseconds (paused audio excluded).
    pub offset_ms: u32,
    /// 0 at start, incremented after every resume. Adapters must not
    /// treat two chunks of different epochs as contiguous audio.
    pub epoch: u16,
}

/// Identity of one span returned by [`Diarizer::assign_spans_at`]:
/// the chunk it came from and its position in that call's result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SpanRef {
    /// Chunk index from the [`ChunkContext`].
    pub chunk_index: u32,
    /// Index into the `Vec<SpeakerSpan>` returned for that chunk.
    pub span_index: u16,
}

/// Limits for [`Diarizer::refine_plan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefineBudget {
    /// The refine must give up (and return an error) once this instant
    /// has passed. It never returns a partial plan.
    pub deadline: Instant,
}

impl RefineBudget {
    /// `Ok` while the deadline is still ahead.
    ///
    /// # Errors
    ///
    /// [`DomainError::DiarizationFailed`] once `deadline` has passed: the
    /// error every refine returns instead of a late or partial plan.
    pub fn check(&self) -> Result<(), DomainError> {
        if Instant::now() >= self.deadline {
            return Err(DomainError::DiarizationFailed(
                "offline refine deadline exceeded".into(),
            ));
        }
        Ok(())
    }
}

/// Result of the end-of-session refinement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefineOutcome {
    /// Nothing to change; the online result stands.
    Unchanged,
    /// Legacy merge-only result: `(absorbed, surviving)` pairs.
    Merges(Vec<(SpeakerId, SpeakerId)>),
    /// Span-level relabelling.
    Relabel(SpanRelabelPlan),
}

/// Span-level relabelling produced by a re-segmenting refine.
///
/// Ids of speakers the online pass created are kept wherever a new
/// cluster matches them, so their slot, colour and user label survive.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpanRelabelPlan {
    /// Final speaker for every logged span.
    pub spans: Vec<(SpanRef, SpeakerId)>,
    /// Speakers the plan introduces (fresh id, next free slot, no label).
    pub created: Vec<Speaker>,
    /// Online speakers left with no span, each with the speaker that
    /// took most of its duration.
    pub retired: Vec<SpeakerRetirement>,
}

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
    ///
    /// Superseded by [`Diarizer::refine_plan`], which can also move
    /// spans between speakers; this stays as the merge-only path.
    fn refine_offline(&self) -> Vec<(SpeakerId, SpeakerId)> {
        Vec::new()
    }

    /// Same as [`Diarizer::assign_spans`], with the chunk's place in the
    /// session so the adapter can keep a timestamped record of it.
    ///
    /// Default impl ignores `ctx` and forwards to `assign_spans`, so
    /// adapters without an offline pass need no change.
    async fn assign_spans_at(
        &mut self,
        samples: &[Sample],
        ctx: ChunkContext,
    ) -> Result<Vec<SpeakerSpan>, DomainError> {
        let _ = ctx;
        self.assign_spans(samples).await
    }

    /// End-of-session refinement as a plan the pipeline can apply to
    /// segments it already emitted.
    ///
    /// Implementations must check `budget.deadline` and return an error
    /// when it passes, never a partial plan. Default impl wraps
    /// [`Diarizer::refine_offline`]: [`RefineOutcome::Merges`], or
    /// [`RefineOutcome::Unchanged`] when it found nothing to merge.
    /// `refine_offline` cannot be interrupted, so the default checks the
    /// budget before and after it ([`RefineBudget::check`]).
    fn refine_plan(&self, budget: RefineBudget) -> Result<RefineOutcome, DomainError> {
        budget.check()?;
        let merges = self.refine_offline();
        budget.check()?;
        Ok(if merges.is_empty() {
            RefineOutcome::Unchanged
        } else {
            RefineOutcome::Merges(merges)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Minimal diarizer: one speaker for every chunk, optional merges.
    struct OneSpeaker {
        speaker: Speaker,
        merges: Vec<(SpeakerId, SpeakerId)>,
    }

    #[async_trait]
    impl Diarizer for OneSpeaker {
        fn sample_rate_hz(&self) -> u32 {
            16_000
        }
        async fn assign(&mut self, _: &[Sample]) -> Result<Option<SpeakerId>, DomainError> {
            Ok(Some(self.speaker.id))
        }
        fn speakers(&self) -> Vec<Speaker> {
            vec![self.speaker.clone()]
        }
        fn rename(&mut self, _: SpeakerId, _: &str) -> Result<bool, DomainError> {
            Ok(false)
        }
        fn reset(&mut self) {}
        fn refine_offline(&self) -> Vec<(SpeakerId, SpeakerId)> {
            self.merges.clone()
        }
    }

    fn budget() -> RefineBudget {
        RefineBudget {
            deadline: Instant::now() + Duration::from_secs(1),
        }
    }

    #[tokio::test]
    async fn assign_spans_at_defaults_to_assign_spans() {
        let mut d = OneSpeaker {
            speaker: Speaker::anonymous(3),
            merges: vec![],
        };
        let ctx = ChunkContext {
            chunk_index: 7,
            offset_ms: 35_000,
            epoch: 1,
        };
        let spans = d.assign_spans_at(&[0.0; 160], ctx).await.unwrap();
        assert_eq!(spans, d.assign_spans(&[0.0; 160]).await.unwrap());
        assert_eq!(spans[0].slot, 3);
    }

    #[test]
    fn refine_plan_defaults_to_unchanged_without_merges() {
        let d = OneSpeaker {
            speaker: Speaker::anonymous(0),
            merges: vec![],
        };
        assert_eq!(d.refine_plan(budget()).unwrap(), RefineOutcome::Unchanged);
    }

    #[test]
    fn default_refine_plan_fails_once_the_deadline_has_passed() {
        let d = OneSpeaker {
            speaker: Speaker::anonymous(0),
            merges: vec![(SpeakerId::new(), SpeakerId::new())],
        };
        let past = RefineBudget {
            deadline: Instant::now() - Duration::from_millis(1),
        };
        assert!(matches!(
            d.refine_plan(past),
            Err(DomainError::DiarizationFailed(_))
        ));
    }

    #[test]
    fn budget_check_passes_before_the_deadline_only() {
        assert!(budget().check().is_ok());
        let past = RefineBudget {
            deadline: Instant::now() - Duration::from_millis(1),
        };
        assert!(matches!(
            past.check(),
            Err(DomainError::DiarizationFailed(_))
        ));
    }

    #[test]
    fn refine_plan_defaults_to_the_legacy_merges() {
        let pair = (SpeakerId::new(), SpeakerId::new());
        let d = OneSpeaker {
            speaker: Speaker::anonymous(0),
            merges: vec![pair],
        };
        assert_eq!(
            d.refine_plan(budget()).unwrap(),
            RefineOutcome::Merges(vec![pair])
        );
    }
}
