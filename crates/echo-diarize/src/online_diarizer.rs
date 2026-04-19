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
use echo_domain::{Diarizer, DomainError, Sample, Speaker, SpeakerId};

use crate::cluster::{OnlineCluster, OnlineClusterConfig};
use crate::embedding::SpeakerEmbedder;

/// Streaming diarizer backed by a swappable embedder + online cluster.
pub struct OnlineDiarizer {
    embedder: Box<dyn SpeakerEmbedder>,
    cluster: OnlineCluster,
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
        }
    }

    /// Convenience constructor with default cluster tuning.
    #[must_use]
    pub fn with_defaults(embedder: Box<dyn SpeakerEmbedder>) -> Self {
        Self::new(embedder, OnlineClusterConfig::default())
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
        let Some(embedding) = self.embedder.embed(samples)? else {
            return Ok(None);
        };
        Ok(Some(self.cluster.assign(embedding)))
    }

    fn speakers(&self) -> Vec<Speaker> {
        self.cluster.speakers()
    }

    fn rename(&mut self, id: SpeakerId, label: &str) -> Result<bool, DomainError> {
        Ok(self.cluster.rename(id, label))
    }

    fn reset(&mut self) {
        self.cluster.reset();
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
}
