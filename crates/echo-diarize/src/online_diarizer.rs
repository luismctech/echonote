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

use std::sync::Arc;

use async_trait::async_trait;
use echo_domain::{
    ChunkContext, Diarizer, DomainError, RefineBudget, RefineOutcome, Sample, Segmenter, SpanRef,
    Speaker, SpeakerId, SpeakerSpan,
};

use crate::cluster::{OnlineCluster, OnlineClusterConfig};
use crate::embedding::SpeakerEmbedder;
use crate::embedding_log::{EmbeddingLog, EmbeddingLogEntry};
use crate::offline_refine::{refine_speakers, OfflineRefineConfig};
use crate::plda::Plda;
use crate::presets::{OfflineRefineMode, DEFAULT_RESEGMENT_THRESHOLD};
use crate::resegment::{relabel_plan, too_short, OfflineReclusterer};
use crate::vbx::{VbxConfig, VbxReclusterer};

/// Streaming diarizer backed by a swappable embedder + online cluster.
///
/// In addition to real-time assignment, the diarizer optionally
/// accumulates every embedding it computes so that an offline
/// refinement pass can be run after the session ends (see
/// [`Diarizer::refine_plan`]). Accumulation is disabled by default to
/// avoid a memory cost on long recordings when the caller does not
/// intend to use the offline pass.
///
/// The end-of-session pass is chosen by one setting, the effective
/// [`OfflineRefineMode`] ([`Self::refine_mode`]): the embedding log is
/// kept exactly when it is not [`OfflineRefineMode::Off`]. The builder
/// methods only record what was asked, so their call order does not
/// matter.
pub struct OnlineDiarizer {
    embedder: Box<dyn SpeakerEmbedder>,
    cluster: OnlineCluster,
    /// Every finite embedding computed by `assign` is appended here with
    /// its session time and online-assigned [`SpeakerId`] while the
    /// effective refine mode is not `Off`. RAM only; never serialised
    /// or logged (ADR 0001, D1).
    embedding_log: EmbeddingLog,
    /// Optional sub-chunk speaker boundary detector. When present,
    /// each audio chunk is split into speaker-homogeneous segments
    /// before embedding, improving accuracy when multiple speakers
    /// alternate within a single VAD chunk.
    segmenter: Option<Box<dyn Segmenter>>,
    /// Configuration [`Diarizer::refine_offline`] runs the offline pass
    /// with. Set by [`Self::with_offline_refinement_config`].
    refine_config: OfflineRefineConfig,
    /// Mode set with [`Self::with_refine_mode`]; it always wins.
    explicit_refine_mode: Option<OfflineRefineMode>,
    /// Legacy opt-in ([`Self::with_offline_refinement`]): selects
    /// [`OfflineRefineMode::MergeOnly`] when no mode was set.
    refinement_requested: bool,
    /// AHC cut of the re-segmentation's VBx initialisation (fixed cut, or
    /// fallback of the calibrated one).
    resegment_threshold: f32,
    /// PLDA model of the embedder; re-segmentation needs it (VBx) and
    /// falls back to the merge-only refine without it.
    plda: Option<Arc<Plda>>,
    /// VBx settings of the re-segmentation.
    vbx_config: VbxConfig,
    /// Context the next plain [`Diarizer::assign_spans`] call is logged
    /// under: callers that do not pass a [`ChunkContext`] get
    /// consecutive chunk indices and back-to-back session times.
    next_ctx: ChunkContext,
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
            embedding_log: EmbeddingLog::default(),
            segmenter: None,
            refine_config: OfflineRefineConfig::default(),
            explicit_refine_mode: None,
            refinement_requested: false,
            resegment_threshold: DEFAULT_RESEGMENT_THRESHOLD,
            plda: None,
            vbx_config: VbxConfig::default(),
            next_ctx: ChunkContext::default(),
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

    /// Request the offline refinement pass: unless
    /// [`Self::with_refine_mode`] picks another mode (before or after
    /// this call), the diarizer runs [`OfflineRefineMode::MergeOnly`].
    ///
    /// Every embedding computed by [`Diarizer::assign`] is then stored in
    /// memory alongside its online-assigned speaker ID. Not enabled by
    /// default because long meetings can accumulate thousands of
    /// embeddings (192 floats each), adding a modest but non-zero memory
    /// cost.
    #[must_use]
    pub fn with_offline_refinement(mut self) -> Self {
        self.refinement_requested = true;
        self
    }

    /// Choose the end-of-session pass. This is the single source of
    /// truth: [`OfflineRefineMode::Off`] keeps no embedding log even if
    /// [`Self::with_offline_refinement`] is called later, and the other
    /// modes keep one.
    #[must_use]
    pub fn with_refine_mode(mut self, mode: OfflineRefineMode) -> Self {
        self.explicit_refine_mode = Some(mode);
        self
    }

    /// Effective end-of-session mode [`Diarizer::refine_plan`] runs: the
    /// mode from [`Self::with_refine_mode`], else `MergeOnly` after
    /// [`Self::with_offline_refinement`], else `Off`.
    #[must_use]
    pub fn refine_mode(&self) -> OfflineRefineMode {
        match (self.explicit_refine_mode, self.refinement_requested) {
            (Some(mode), _) => mode,
            (None, true) => OfflineRefineMode::MergeOnly,
            (None, false) => OfflineRefineMode::Off,
        }
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

    /// Cosine similarity at which the re-segmenting AHC
    /// ([`OfflineRefineMode::Resegment`]) stops merging. Independent of
    /// the merge-only threshold in [`OfflineRefineConfig`]; per-embedder
    /// values live in [`crate::EmbedderFamily::resegment_threshold`].
    #[must_use]
    pub fn with_resegment_threshold(mut self, threshold: f32) -> Self {
        self.resegment_threshold = threshold;
        self
    }

    /// PLDA model of this embedder, which
    /// [`OfflineRefineMode::Resegment`] (VBx) requires. Per-embedder
    /// models come from [`crate::EmbedderFamily::plda`].
    #[must_use]
    pub fn with_plda(mut self, plda: Arc<Plda>) -> Self {
        self.plda = Some(plda);
        self
    }

    /// VBx settings of the re-segmentation; per-embedder values live in
    /// [`crate::EmbedderFamily::vbx_config`].
    #[must_use]
    pub fn with_vbx_config(mut self, config: VbxConfig) -> Self {
        self.vbx_config = config;
        self
    }

    /// Re-segment the log with other VBx settings and AHC cut, whatever
    /// the configured mode. Benchmarks use it to score several settings
    /// on one session; the app calls [`Diarizer::refine_plan`].
    ///
    /// # Errors
    ///
    /// [`DomainError::DiarizationFailed`] without a PLDA model, past the
    /// deadline, or when the clustering fails.
    pub fn resegment_with(
        &self,
        config: VbxConfig,
        init_threshold: f32,
        budget: RefineBudget,
    ) -> Result<RefineOutcome, DomainError> {
        let Some(plda) = &self.plda else {
            return Err(DomainError::DiarizationFailed(
                "re-segmentation needs a PLDA model".into(),
            ));
        };
        let entries = self.embedding_log.entries();
        if too_short(entries) {
            return Ok(RefineOutcome::Unchanged);
        }
        let labels = VbxReclusterer {
            plda: Arc::clone(plda),
            config,
            init_threshold,
        }
        .recluster(entries, budget)?;
        budget.check()?;
        Ok(relabel_plan(entries, &self.cluster.speakers(), &labels))
    }

    fn merge_only_plan(&self, budget: RefineBudget) -> Result<RefineOutcome, DomainError> {
        budget.check()?;
        let merges = self.refine_offline(self.refine_config);
        budget.check()?;
        Ok(if merges.is_empty() {
            RefineOutcome::Unchanged
        } else {
            RefineOutcome::Merges(merges)
        })
    }

    fn keeps_log(&self) -> bool {
        self.refine_mode() != OfflineRefineMode::Off
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
        if !self.keeps_log() {
            return Vec::new();
        }
        let records: Vec<(&[f32], SpeakerId)> = self
            .embedding_log
            .entries()
            .iter()
            .map(|e| (&*e.embedding, e.online_speaker))
            .collect();
        refine_speakers(&records, &self.cluster.speakers(), config)
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

    #[cfg(test)]
    fn log(&self) -> &[EmbeddingLogEntry] {
        self.embedding_log.entries()
    }

    fn samples_to_ms(&self, samples: usize) -> u32 {
        let rate = u64::from(self.embedder.sample_rate_hz().max(1));
        u32::try_from(samples as u64 * 1_000 / rate).unwrap_or(u32::MAX)
    }

    /// Embed one span, assign it online and log it. `None` when the
    /// embedder declined the span (too short, low confidence).
    fn embed_span(
        &mut self,
        samples: &[Sample],
        range: std::ops::Range<usize>,
        local_speaker: Option<u8>,
        span: SpanRef,
        ctx: ChunkContext,
    ) -> Result<Option<SpeakerSpan>, DomainError> {
        let Some(embedding) = self.embedder.embed(&samples[range.clone()])? else {
            return Ok(None);
        };
        let start_ms = ctx
            .offset_ms
            .saturating_add(self.samples_to_ms(range.start));
        let end_ms = ctx.offset_ms.saturating_add(self.samples_to_ms(range.end));
        // A non-finite vector (broken model output) is still assigned
        // online as before, but never logged: it would poison the
        // end-of-session clustering.
        let id = if self.keeps_log() && embedding.iter().all(|x| x.is_finite()) {
            let id = self.cluster.assign(embedding.clone());
            self.embedding_log.push(EmbeddingLogEntry::new(
                vec![span],
                start_ms..end_ms,
                ctx.epoch,
                local_speaker,
                id,
                embedding.into_boxed_slice(),
            ));
            id
        } else {
            self.cluster.assign(embedding)
        };
        let slot = self
            .cluster
            .speakers()
            .into_iter()
            .find(|s| s.id == id)
            .map(|s| s.slot)
            .unwrap_or(0);
        Ok(Some(SpeakerSpan {
            speaker_id: id,
            slot,
            start_sample: range.start,
            end_sample: range.end,
        }))
    }

    async fn assign_with_context(
        &mut self,
        samples: &[Sample],
        ctx: ChunkContext,
    ) -> Result<Vec<SpeakerSpan>, DomainError> {
        self.next_ctx = ChunkContext {
            chunk_index: ctx.chunk_index.saturating_add(1),
            offset_ms: ctx
                .offset_ms
                .saturating_add(self.samples_to_ms(samples.len())),
            epoch: ctx.epoch,
        };
        let span_ref = |spans: &[SpeakerSpan]| SpanRef {
            chunk_index: ctx.chunk_index,
            span_index: u16::try_from(spans.len()).unwrap_or(u16::MAX),
        };
        // Segmented path: emit one span per speaker-homogeneous
        // sub-region the segmenter detected. Overlapping pyannote
        // outputs (frames where two speaker channels are active at
        // once) come through as two SpeakerSegments with overlapping
        // sample ranges, which we preserve verbatim — the streaming
        // pipeline distributes them across Whisper segments.
        let sub_segments = match self.segmenter {
            Some(ref mut segmenter) => segmenter.segment(samples)?,
            None => Vec::new(),
        };
        if !sub_segments.is_empty() {
            let mut spans: Vec<SpeakerSpan> = Vec::with_capacity(sub_segments.len());
            for seg in &sub_segments {
                let end = seg.end_sample.min(samples.len());
                if end <= seg.start_sample {
                    continue;
                }
                let r = span_ref(&spans);
                if let Some(span) = self.embed_span(
                    samples,
                    seg.start_sample..end,
                    Some(seg.local_speaker),
                    r,
                    ctx,
                )? {
                    spans.push(span);
                }
            }
            return Ok(spans);
        }
        // Whole-chunk path (no segmenter, or segmenter returned nothing).
        let r = span_ref(&[]);
        Ok(self
            .embed_span(samples, 0..samples.len(), None, r, ctx)?
            .into_iter()
            .collect())
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
        let ctx = self.next_ctx;
        self.assign_with_context(samples, ctx).await
    }

    async fn assign_spans_at(
        &mut self,
        samples: &[Sample],
        ctx: ChunkContext,
    ) -> Result<Vec<SpeakerSpan>, DomainError> {
        self.assign_with_context(samples, ctx).await
    }

    fn speakers(&self) -> Vec<Speaker> {
        self.cluster.speakers()
    }

    fn rename(&mut self, id: SpeakerId, label: &str) -> Result<bool, DomainError> {
        Ok(self.cluster.rename(id, label))
    }

    fn reset(&mut self) {
        self.cluster.reset();
        self.next_ctx = ChunkContext::default();
        self.embedding_log.clear();
    }

    fn refine_offline(&self) -> Vec<(SpeakerId, SpeakerId)> {
        // Forward to the inherent helper, which returns an empty vec when
        // the effective refine mode is `Off` (no log is kept).
        self.refine_offline(self.refine_config)
    }

    fn refine_plan(&self, budget: RefineBudget) -> Result<RefineOutcome, DomainError> {
        match self.refine_mode() {
            OfflineRefineMode::Off => Ok(RefineOutcome::Unchanged),
            OfflineRefineMode::MergeOnly => self.merge_only_plan(budget),
            OfflineRefineMode::Resegment => {
                if self
                    .plda
                    .as_ref()
                    .is_some_and(|p| p.dim() == self.embedder.dim())
                {
                    self.resegment_with(self.vbx_config, self.resegment_threshold, budget)
                } else {
                    tracing::info!(
                        "no PLDA model for this embedder; re-segmentation falls back to merge-only"
                    );
                    self.merge_only_plan(budget)
                }
            }
        }
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
    /// 3 s of audio at 16 kHz whose stub embedding follows `seed`.
    fn long_chunk(seed: f32) -> Vec<Sample> {
        vec![seed; 48_000]
    }

    fn budget() -> RefineBudget {
        RefineBudget {
            deadline: std::time::Instant::now() + std::time::Duration::from_secs(10),
        }
    }

    /// VBx settings for the 4-dim stub: a fixed AHC cut and a light
    /// speaker regularisation, so a handful of frames keep two speakers.
    fn test_vbx() -> VbxConfig {
        VbxConfig {
            params: crate::vbx::VbxParams {
                fa: 1.0,
                fb: 1.0,
                loop_p: 0.5,
                max_iters: 10,
                epsilon: 1e-6,
            },
            init_offset: None,
            ..VbxConfig::default()
        }
    }

    fn diarizer(mode: OfflineRefineMode, threshold: f32) -> OnlineDiarizer {
        OnlineDiarizer::with_defaults(Box::new(StubEmbedder::new()))
            .with_offline_refinement_config(OfflineRefineConfig {
                merge_threshold: threshold,
                ..OfflineRefineConfig::default()
            })
            .with_resegment_threshold(threshold)
            .with_plda(Arc::new(crate::plda::tests::identity(4, 25.0)))
            .with_vbx_config(test_vbx())
            .with_refine_mode(mode)
    }

    async fn feed(d: &mut OnlineDiarizer, seeds: &[f32]) {
        for (i, &seed) in seeds.iter().enumerate() {
            let ctx = ChunkContext {
                chunk_index: i as u32,
                offset_ms: i as u32 * 3_000,
                epoch: 0,
            };
            d.assign_spans_at(&long_chunk(seed), ctx).await.unwrap();
        }
    }

    #[tokio::test]
    async fn assign_spans_at_logs_session_time_and_span_identity() {
        let mut d = diarizer(OfflineRefineMode::Resegment, 0.5);
        let ctx = ChunkContext {
            chunk_index: 7,
            offset_ms: 60_000,
            epoch: 2,
        };
        let spans = d.assign_spans_at(&long_chunk(0.001), ctx).await.unwrap();

        let log = d.log();
        assert_eq!(log.len(), 1);
        let e = &log[0];
        assert_eq!(
            e.spans,
            vec![SpanRef {
                chunk_index: 7,
                span_index: 0
            }]
        );
        assert_eq!((e.start_ms, e.end_ms, e.epoch), (60_000, 63_000, 2));
        assert_eq!(e.local_speaker, None, "whole-chunk path has no hint");
        assert_eq!(e.online_speaker, spans[0].speaker_id);
    }

    /// Embedder whose model is broken: every vector is NaN.
    struct NanEmbedder;

    impl SpeakerEmbedder for NanEmbedder {
        fn sample_rate_hz(&self) -> u32 {
            16_000
        }
        fn dim(&self) -> usize {
            4
        }
        fn embed(&mut self, _: &[Sample]) -> Result<Option<Vec<f32>>, DomainError> {
            Ok(Some(vec![f32::NAN; 4]))
        }
    }

    #[tokio::test]
    async fn non_finite_embeddings_are_never_logged() {
        let mut d = OnlineDiarizer::with_defaults(Box::new(NanEmbedder))
            .with_refine_mode(OfflineRefineMode::Resegment);
        d.assign_spans(&long_chunk(0.001)).await.unwrap();
        assert!(d.log().is_empty());
    }

    #[tokio::test]
    async fn plain_assign_spans_logs_back_to_back_chunks() {
        let mut d = diarizer(OfflineRefineMode::MergeOnly, 0.5);
        d.assign_spans(&long_chunk(0.001)).await.unwrap();
        d.assign_spans(&long_chunk(0.002)).await.unwrap();
        let times: Vec<_> = d
            .log()
            .iter()
            .map(|e| (e.spans[0].chunk_index, e.start_ms))
            .collect();
        assert_eq!(times, vec![(0, 0), (1, 3_000)]);
    }

    #[tokio::test]
    async fn off_mode_keeps_no_log_and_never_refines() {
        let mut d = diarizer(OfflineRefineMode::Off, -0.5);
        feed(&mut d, &[0.001, 0.002, 0.001, 0.002, 0.001]).await;
        assert!(d.log().is_empty());
        assert_eq!(d.refine_plan(budget()).unwrap(), RefineOutcome::Unchanged);
        assert!(Diarizer::refine_offline(&d).is_empty());
    }

    #[tokio::test]
    async fn off_mode_wins_whatever_the_builder_order() {
        let mut d = OnlineDiarizer::with_defaults(Box::new(StubEmbedder::new()))
            .with_refine_mode(OfflineRefineMode::Off)
            .with_offline_refinement_config(OfflineRefineConfig {
                merge_threshold: -0.5,
                ..OfflineRefineConfig::default()
            })
            .with_offline_refinement();
        assert_eq!(d.refine_mode(), OfflineRefineMode::Off);
        feed(&mut d, &[0.001, 0.002, 0.001, 0.002]).await;
        assert!(d.log().is_empty());
        assert!(Diarizer::refine_offline(&d).is_empty());
    }

    #[test]
    fn refine_mode_is_off_until_requested_and_merge_only_by_opt_in() {
        let plain = OnlineDiarizer::with_defaults(Box::new(StubEmbedder::new()));
        assert_eq!(plain.refine_mode(), OfflineRefineMode::Off);
        let opted =
            OnlineDiarizer::with_defaults(Box::new(StubEmbedder::new())).with_offline_refinement();
        assert_eq!(opted.refine_mode(), OfflineRefineMode::MergeOnly);
    }

    #[tokio::test]
    async fn resegment_cut_is_independent_of_the_merge_threshold() {
        // The merge-only threshold would merge everything; the
        // re-segmenting cut keeps the orthogonal speakers apart.
        let mut d = OnlineDiarizer::with_defaults(Box::new(StubEmbedder::new()))
            .with_offline_refinement_config(OfflineRefineConfig {
                merge_threshold: -0.5,
                ..OfflineRefineConfig::default()
            })
            .with_resegment_threshold(0.5)
            .with_plda(Arc::new(crate::plda::tests::identity(4, 25.0)))
            .with_vbx_config(test_vbx())
            .with_refine_mode(OfflineRefineMode::Resegment);
        feed(&mut d, &[0.001, 0.002, 0.001, 0.002, 0.001]).await;
        assert_eq!(d.refine_plan(budget()).unwrap(), RefineOutcome::Unchanged);
    }

    #[tokio::test]
    async fn resegment_without_a_plda_model_falls_back_to_merge_only() {
        let mut d = OnlineDiarizer::with_defaults(Box::new(StubEmbedder::new()))
            .with_offline_refinement_config(OfflineRefineConfig {
                merge_threshold: -0.5,
                ..OfflineRefineConfig::default()
            })
            .with_resegment_threshold(-0.5)
            .with_refine_mode(OfflineRefineMode::Resegment);
        feed(&mut d, &[0.001, 0.002, 0.001, 0.002, 0.001]).await;
        assert!(matches!(
            d.refine_plan(budget()).unwrap(),
            RefineOutcome::Merges(m) if m.len() == 1
        ));
    }

    #[tokio::test]
    async fn resegment_with_a_plda_of_another_dimension_falls_back_to_merge_only() {
        let mut d = diarizer(OfflineRefineMode::Resegment, -0.5)
            .with_plda(Arc::new(crate::plda::tests::identity(3, 25.0)));
        feed(&mut d, &[0.001, 0.002, 0.001, 0.002, 0.001]).await;
        assert!(matches!(
            d.refine_plan(budget()).unwrap(),
            RefineOutcome::Merges(_)
        ));
    }

    #[tokio::test]
    async fn resegment_with_moves_a_misattributed_span_back() {
        // Online, the third chunk (axis 1) joins speaker 0 because the
        // test cluster is lenient; VBx separates the two voices again.
        let mut d = OnlineDiarizer::new(
            Box::new(StubEmbedder::new()),
            crate::cluster::OnlineClusterConfig {
                similarity_threshold: -0.5,
                ..crate::cluster::OnlineClusterConfig::default()
            },
        )
        .with_plda(Arc::new(crate::plda::tests::identity(4, 25.0)))
        .with_vbx_config(test_vbx())
        .with_resegment_threshold(0.5)
        .with_refine_mode(OfflineRefineMode::Resegment);
        feed(&mut d, &[0.001, 0.001, 0.002, 0.002, 0.001, 0.002]).await;
        assert_eq!(d.speakers().len(), 1, "online put every chunk together");
        let RefineOutcome::Relabel(plan) = d.refine_plan(budget()).unwrap() else {
            panic!("expected a relabel plan");
        };
        let ids: Vec<_> = plan.spans.iter().map(|(_, id)| *id).collect();
        assert_eq!(ids[0], ids[1]);
        assert_eq!(ids[2], ids[3]);
        assert_ne!(ids[0], ids[2]);
        assert_eq!(plan.created.len(), 1);
    }

    #[tokio::test]
    async fn merge_only_mode_returns_the_legacy_merges() {
        let mut d = diarizer(OfflineRefineMode::MergeOnly, -0.5);
        feed(&mut d, &[0.001, 0.002, 0.001, 0.002]).await;
        assert!(matches!(
            d.refine_plan(budget()).unwrap(),
            RefineOutcome::Merges(m) if m.len() == 1
        ));
    }

    #[tokio::test]
    async fn resegment_mode_relabels_spans_and_retires_the_absorbed_speaker() {
        // A negative threshold puts every span in one cluster: the
        // speaker with the most speech keeps its id, the other retires.
        let mut d = diarizer(OfflineRefineMode::Resegment, -0.5);
        feed(&mut d, &[0.001, 0.002, 0.001, 0.002, 0.001]).await;
        let speakers = d.speakers();
        let (a, b) = (speakers[0].id, speakers[1].id);

        let RefineOutcome::Relabel(plan) = d.refine_plan(budget()).unwrap() else {
            panic!("expected a relabel plan");
        };
        assert_eq!(plan.spans.len(), 5);
        assert!(plan.spans.iter().all(|(_, id)| *id == a));
        assert!(plan.created.is_empty());
        assert_eq!(plan.retired.len(), 1);
        assert_eq!(plan.retired[0].speaker, b);
        assert_eq!(plan.retired[0].majority_recipient, Some(a));
    }

    #[tokio::test]
    async fn resegment_mode_keeps_a_correct_online_result_unchanged() {
        let mut d = diarizer(OfflineRefineMode::Resegment, 0.5);
        feed(&mut d, &[0.001, 0.002, 0.001, 0.002, 0.001]).await;
        assert_eq!(d.refine_plan(budget()).unwrap(), RefineOutcome::Unchanged);
    }

    #[tokio::test]
    async fn resegment_mode_skips_short_sessions() {
        // 3 chunks x 3 s: below both the entry and the speech floor.
        let mut d = diarizer(OfflineRefineMode::Resegment, -0.5);
        feed(&mut d, &[0.001, 0.002, 0.001]).await;
        assert_eq!(d.refine_plan(budget()).unwrap(), RefineOutcome::Unchanged);
    }

    #[tokio::test]
    async fn reset_clears_the_log_and_the_implicit_context() {
        let mut d = diarizer(OfflineRefineMode::Resegment, 0.5);
        feed(&mut d, &[0.001, 0.002]).await;
        d.reset();
        assert!(d.log().is_empty());
        d.assign_spans(&long_chunk(0.001)).await.unwrap();
        assert_eq!(d.log()[0].start_ms, 0);
    }
}
