//! Streaming-friendly [`Diarizer`] adapter.
//!
//! [`OnlineDiarizer`] glues a [`SpeakerEmbedder`] (the model — ERes2Net
//! in production, a stub in tests) to an [`OnlineCluster`] (the
//! threshold-based assignment policy). It owns no audio I/O and no
//! VAD; callers should hand it pre-segmented voiced chunks at the
//! embedder's expected sample rate.
//!
//! ## Threading
//!
//! The diarizer is not internally locked. Callers are expected to
//! drive it from a single task per audio track, which matches the
//! per-track contract documented on [`Diarizer`].

use async_trait::async_trait;
use echo_domain::{Diarizer, DomainError, Sample, Segmenter, Speaker, SpeakerId, SpeakerSpan};

use crate::cluster::{OnlineCluster, OnlineClusterConfig};
use crate::embedding::SpeakerEmbedder;
use crate::offline_refine::{refine_speakers, OfflineRefineConfig};

/// Streaming diarizer backed by a swappable embedder + online cluster.
///
/// In addition to real-time assignment, the diarizer optionally
/// accumulates every embedding it computes so that an offline AHC
/// refinement pass can be run after the session ends (see
/// [`OnlineDiarizer::refine_offline`]).  Accumulation is disabled by
/// default to avoid a memory cost on long recordings when the caller
/// does not intend to use the offline pass.
pub struct OnlineDiarizer {
    embedder: Box<dyn SpeakerEmbedder>,
    cluster: OnlineCluster,
    /// When `Some`, every embedding computed by `assign` is appended
    /// here alongside the online-assigned [`SpeakerId`].
    embedding_log: Option<Vec<(Vec<f32>, SpeakerId)>>,
    /// Optional sub-chunk speaker boundary detector. When present,
    /// each audio chunk is split into speaker-homogeneous segments
    /// before embedding, improving accuracy when multiple speakers
    /// alternate within a single VAD chunk.
    segmenter: Option<Box<dyn Segmenter>>,
    /// Configuration [`Diarizer::refine_offline`] runs the offline pass
    /// with. Set by [`Self::with_offline_refinement_config`].
    refine_config: OfflineRefineConfig,
}

impl OnlineDiarizer {
    /// Wire up an embedder with the supplied cluster configuration.
    /// Construction is infallible; failures live in
    /// [`SpeakerEmbedder::embed`] (model inference) and surface
    /// through [`Diarizer::assign`].
    #[must_use]
    pub fn new(embedder: Box<dyn SpeakerEmbedder>, cluster_config: OnlineClusterConfig) -> Self {
        Self {
            embedder,
            cluster: OnlineCluster::new(cluster_config),
            embedding_log: None,
            segmenter: None,
            refine_config: OfflineRefineConfig::default(),
        }
    }

    /// Convenience constructor with default cluster tuning.
    #[must_use]
    pub fn with_defaults(embedder: Box<dyn SpeakerEmbedder>) -> Self {
        Self::new(embedder, OnlineClusterConfig::default())
    }

    /// Attach a [`Segmenter`] that will split each chunk into
    /// speaker-homogeneous sub-regions before embedding. The dominant
    /// speaker (by sample count) is returned from [`Diarizer::assign`].
    ///
    /// When the segmenter returns an empty result (e.g. a silent chunk),
    /// the diarizer falls back to whole-chunk embedding.
    #[must_use]
    pub fn with_segmenter(mut self, segmenter: Box<dyn Segmenter>) -> Self {
        self.segmenter = Some(segmenter);
        self
    }

    /// Enable embedding accumulation for the offline refinement pass.
    ///
    /// When called before the session starts, every embedding computed
    /// by [`Diarizer::assign`] is stored in memory alongside its
    /// online-assigned speaker ID.  Call [`Self::refine_offline`] after
    /// the session ends to re-cluster the accumulated embeddings and
    /// obtain a merge map.
    ///
    /// Not enabled by default because long meetings can accumulate
    /// several hundred embeddings (192 floats each), adding a modest
    /// but non-zero memory cost.
    #[must_use]
    pub fn with_offline_refinement(mut self) -> Self {
        self.embedding_log = Some(Vec::new());
        self
    }

    /// Like [`Self::with_offline_refinement`], but the end-of-session
    /// pass behind [`Diarizer::refine_offline`] uses `config` instead of
    /// [`OfflineRefineConfig::default`]. Per-embedder values live in
    /// [`crate::EmbedderFamily::refine_config`].
    #[must_use]
    pub fn with_offline_refinement_config(mut self, config: OfflineRefineConfig) -> Self {
        self.refine_config = config;
        self.with_offline_refinement()
    }

    /// Run the offline AHC refinement pass over all accumulated
    /// embeddings and return a list of `(absorbed_id, surviving_id)`
    /// speaker merges.
    ///
    /// Returns an empty `Vec` when:
    /// - offline accumulation was not enabled ([`Self::with_offline_refinement`]),
    /// - fewer than two speakers were detected, or
    /// - the AHC pass found no pairs above `config.merge_threshold`.
    ///
    /// The caller is responsible for applying the returned merges to
    /// the stored segment records (typically in the stop-recording use
    /// case, before persisting to the database).
    pub fn refine_offline(&self, config: OfflineRefineConfig) -> Vec<(SpeakerId, SpeakerId)> {
        let Some(ref log) = self.embedding_log else {
            return Vec::new();
        };
        refine_speakers(log, &self.cluster.speakers(), config)
    }

    /// Convenience overload using [`OfflineRefineConfig::default`].
    pub fn refine_offline_default(&self) -> Vec<(SpeakerId, SpeakerId)> {
        self.refine_offline(OfflineRefineConfig::default())
    }

    /// Direct cluster handle for tests and observability. Not part
    /// of the public diarizer contract.
    #[cfg(test)]
    pub fn cluster(&self) -> &OnlineCluster {
        &self.cluster
    }
}

#[async_trait]
impl Diarizer for OnlineDiarizer {
    fn sample_rate_hz(&self) -> u32 {
        self.embedder.sample_rate_hz()
    }

    async fn assign(&mut self, samples: &[Sample]) -> Result<Option<SpeakerId>, DomainError> {
        // Legacy single-speaker path: forward to assign_spans and pick
        // the dominant speaker by sample-weighted majority. Kept so
        // callers that haven't migrated to spans still get a sensible
        // answer; the streaming pipeline calls assign_spans directly.
        let spans = self.assign_spans(samples).await?;
        if spans.is_empty() {
            return Ok(None);
        }
        let mut totals: Vec<(SpeakerId, usize)> = Vec::with_capacity(spans.len());
        for span in &spans {
            let len = span.len_samples();
            if let Some((_, acc)) = totals.iter_mut().find(|(id, _)| *id == span.speaker_id) {
                *acc += len;
            } else {
                totals.push((span.speaker_id, len));
            }
        }
        let dominant = totals.into_iter().max_by_key(|(_, n)| *n).map(|(id, _)| id);
        Ok(dominant)
    }

    async fn assign_spans(&mut self, samples: &[Sample]) -> Result<Vec<SpeakerSpan>, DomainError> {
        // Segmented path: emit one span per speaker-homogeneous
        // sub-region the segmenter detected. Overlapping pyannote
        // outputs (frames where two speaker channels are active at
        // once) come through as two SpeakerSegments with overlapping
        // sample ranges, which we preserve verbatim — the streaming
        // pipeline distributes them across Whisper segments.
        if let Some(ref mut segmenter) = self.segmenter {
            let sub_segments = segmenter.segment(samples)?;
            if !sub_segments.is_empty() {
                let mut spans: Vec<SpeakerSpan> = Vec::with_capacity(sub_segments.len());
                for seg in &sub_segments {
                    let end = seg.end_sample.min(samples.len());
                    if end <= seg.start_sample {
                        continue;
                    }
                    let sub = &samples[seg.start_sample..end];
                    let Some(embedding) = self.embedder.embed(sub)? else {
                        continue;
                    };
                    let id = self.cluster.assign(embedding.clone());
                    if let Some(ref mut log) = self.embedding_log {
                        log.push((embedding, id));
                    }
                    let slot = self
                        .cluster
                        .speakers()
                        .into_iter()
                        .find(|s| s.id == id)
                        .map(|s| s.slot)
                        .unwrap_or(0);
                    spans.push(SpeakerSpan {
                        speaker_id: id,
                        slot,
                        start_sample: seg.start_sample,
                        end_sample: end,
                    });
                }
                return Ok(spans);
            }
        }
        // Whole-chunk path (no segmenter, or segmenter returned nothing).
        let Some(embedding) = self.embedder.embed(samples)? else {
            return Ok(Vec::new());
        };
        let id = self.cluster.assign(embedding.clone());
        if let Some(ref mut log) = self.embedding_log {
            log.push((embedding, id));
        }
        let slot = self
            .cluster
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

    fn speakers(&self) -> Vec<Speaker> {
        self.cluster.speakers()
    }

    fn rename(&mut self, id: SpeakerId, label: &str) -> Result<bool, DomainError> {
        Ok(self.cluster.rename(id, label))
    }

    fn reset(&mut self) {
        self.cluster.reset();
        if let Some(ref mut log) = self.embedding_log {
            log.clear();
        }
    }

    fn refine_offline(&self) -> Vec<(SpeakerId, SpeakerId)> {
        // Forward to the inherent helper, which checks whether
        // [`with_offline_refinement`] was opted into. When it wasn't
        // (embedding_log is None) the helper returns an empty vec.
        self.refine_offline(self.refine_config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use std::collections::HashMap;

    /// Deterministic embedder: every chunk maps to a 4-D vector
    /// derived from the *first sample* of the chunk. Same first
    /// sample ⇒ identical embedding ⇒ same cluster, regardless of
    /// the rest of the audio. Useful for asserting that the
    /// diarizer faithfully passes embeddings through to the
    /// cluster.
    struct StubEmbedder {
        dim: usize,
        sample_rate_hz: u32,
    }

    impl StubEmbedder {
        fn new() -> Self {
            Self {
                dim: 4,
                sample_rate_hz: 16_000,
            }
        }
    }

    impl SpeakerEmbedder for StubEmbedder {
        fn sample_rate_hz(&self) -> u32 {
            self.sample_rate_hz
        }

        fn dim(&self) -> usize {
            self.dim
        }

        fn embed(&mut self, samples: &[Sample]) -> Result<Option<Vec<f32>>, DomainError> {
            if samples.is_empty() {
                return Ok(None);
            }
            // Map the leading sample to a direction on the unit
            // hyper-cube: sign(s) along axis (s.abs() % 4).
            let s = samples[0];
            let axis = (s.abs() * 1000.0) as usize % self.dim;
            let mut v = vec![0.0_f32; self.dim];
            v[axis] = if s >= 0.0 { 1.0 } else { -1.0 };
            Ok(Some(v))
        }
    }

    fn chunk(seed: f32) -> Vec<Sample> {
        vec![seed; 480]
    }

    #[tokio::test]
    async fn empty_chunk_yields_no_speaker() {
        let mut d = OnlineDiarizer::with_defaults(Box::new(StubEmbedder::new()));
        let r = d.assign(&[]).await.expect("embedder shouldn't error");
        assert!(r.is_none());
        assert!(d.speakers().is_empty());
    }

    #[tokio::test]
    async fn distinct_seeds_produce_distinct_speakers() {
        let mut d = OnlineDiarizer::with_defaults(Box::new(StubEmbedder::new()));
        // Each seed lands on a different axis ⇒ orthogonal embeddings.
        let seeds = [0.001_f32, 0.002, 0.003, 0.004];
        let mut ids: Vec<SpeakerId> = Vec::new();
        for s in seeds {
            ids.push(d.assign(&chunk(s)).await.unwrap().unwrap());
        }
        assert_eq!(
            ids.iter().collect::<std::collections::HashSet<_>>().len(),
            4
        );
        assert_eq!(d.speakers().len(), 4);
    }

    #[tokio::test]
    async fn same_seed_collapses_into_same_speaker_across_turns() {
        let mut d = OnlineDiarizer::with_defaults(Box::new(StubEmbedder::new()));
        let a1 = d.assign(&chunk(0.001)).await.unwrap().unwrap();
        let _b = d.assign(&chunk(0.002)).await.unwrap().unwrap();
        let a2 = d.assign(&chunk(0.001)).await.unwrap().unwrap();
        assert_eq!(a1, a2);
        assert_eq!(d.speakers().len(), 2);
    }

    #[tokio::test]
    async fn rename_round_trips_through_diarizer_port() {
        let mut d = OnlineDiarizer::with_defaults(Box::new(StubEmbedder::new()));
        let id = d.assign(&chunk(0.001)).await.unwrap().unwrap();
        assert!(d.rename(id, "Alice").unwrap());
        let names: HashMap<SpeakerId, String> = d
            .speakers()
            .into_iter()
            .map(|s| (s.id, s.display_name()))
            .collect();
        assert_eq!(names.get(&id).map(String::as_str), Some("Alice"));
    }

    #[tokio::test]
    async fn reset_clears_speakers_but_keeps_embedder_state() {
        let mut d = OnlineDiarizer::with_defaults(Box::new(StubEmbedder::new()));
        d.assign(&chunk(0.001)).await.unwrap();
        d.assign(&chunk(0.002)).await.unwrap();
        assert_eq!(d.speakers().len(), 2);
        d.reset();
        assert!(d.speakers().is_empty());

        // After reset, slot numbering restarts from zero.
        let id = d.assign(&chunk(0.003)).await.unwrap().unwrap();
        let snap = d.speakers();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].slot, 0);
        assert_eq!(snap[0].id, id);
    }

    #[tokio::test]
    async fn sample_rate_is_derived_from_embedder() {
        let d = OnlineDiarizer::with_defaults(Box::new(StubEmbedder::new()));
        assert_eq!(d.sample_rate_hz(), 16_000);
    }

    /// Two orthogonal speakers, two chunks each, logged for refinement.
    async fn two_logged_speakers(d: &mut OnlineDiarizer) {
        for s in [0.001_f32, 0.002, 0.001, 0.002] {
            d.assign(&chunk(s)).await.unwrap();
        }
        assert_eq!(d.speakers().len(), 2);
    }

    #[tokio::test]
    async fn port_refinement_uses_the_configured_threshold() {
        // Orthogonal speakers have average-linkage cosine 0: the default
        // threshold keeps them apart, a negative one merges them.
        let mut lenient = OnlineDiarizer::with_defaults(Box::new(StubEmbedder::new()))
            .with_offline_refinement_config(OfflineRefineConfig {
                merge_threshold: -0.5,
                ..OfflineRefineConfig::default()
            });
        two_logged_speakers(&mut lenient).await;
        assert_eq!(Diarizer::refine_offline(&lenient).len(), 1);

        let mut default =
            OnlineDiarizer::with_defaults(Box::new(StubEmbedder::new())).with_offline_refinement();
        two_logged_speakers(&mut default).await;
        assert!(Diarizer::refine_offline(&default).is_empty());
    }
}
