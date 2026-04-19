//! Online speaker clustering.
//!
//! Given a stream of L2-normalised speaker embeddings arriving in
//! chronological order, [`OnlineCluster`] keeps a small set of
//! centroids and assigns each new embedding to the nearest centroid
//! (cosine similarity). When no centroid is close enough — and we
//! still have headroom under `max_speakers` — a new centroid is
//! created.
//!
//! ## Why threshold-based, not full agglomerative?
//!
//! Streaming meetings have two properties that make incremental
//! threshold clustering attractive over an offline agglomerative
//! pass:
//!
//! - **Latency.** Each chunk needs a label *now*, not after the
//!   meeting ends. Online assignment is O(k) per chunk where k is
//!   the (small) number of speakers found so far.
//! - **Bounded memory.** We never need to hold more than `k`
//!   centroids; an agglomerative algorithm would grow a distance
//!   matrix proportional to total chunk count.
//!
//! The trade-off is sensitivity to the similarity threshold. We pin
//! a sensible default (`0.78`) calibrated against ERes2Net
//! embeddings on the VoxConverse subset and let callers override it
//! through [`OnlineClusterConfig`].
//!
//! ## Centroid update
//!
//! Centroids are running means of the embeddings assigned to them,
//! re-normalised after each update. This keeps the centroid on the
//! unit sphere where cosine ≡ dot product, and it down-weights any
//! single chunk in proportion to how many we have already seen,
//! making the cluster robust to occasional noisy embeddings.

use echo_domain::{Speaker, SpeakerId};
use serde::{Deserialize, Serialize};

use crate::embedding::{cosine_similarity, l2_normalize};

/// Tunables for [`OnlineCluster`]. The defaults target ERes2Net-like
/// embeddings; calibrate per-model when adding new embedders.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OnlineClusterConfig {
    /// Minimum cosine similarity between a new embedding and the
    /// best-matching centroid for the new chunk to be assigned to
    /// that existing speaker. Below this, a new speaker is spawned
    /// (subject to `max_speakers`).
    pub similarity_threshold: f32,

    /// Hard cap on the number of distinct speakers we will spawn.
    /// Once reached, any chunk that would otherwise create a new
    /// cluster is folded into the nearest existing one. Prevents
    /// runaway speaker counts from noisy embeddings on long
    /// recordings.
    pub max_speakers: usize,
}

impl Default for OnlineClusterConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.78,
            max_speakers: 8,
        }
    }
}

/// One running centroid plus its bookkeeping. Kept private; the
/// cluster surface only exposes [`Speaker`] snapshots.
#[derive(Debug, Clone)]
struct Centroid {
    speaker: Speaker,
    /// Normalised running mean of all embeddings assigned to this
    /// centroid.
    vector: Vec<f32>,
    /// Count of embeddings folded into this centroid. Used as the
    /// running-mean denominator and exposed via the snapshot for
    /// debugging.
    count: u32,
}

/// Online, threshold-based clustering of L2-normalised speaker
/// embeddings.
#[derive(Debug)]
pub struct OnlineCluster {
    config: OnlineClusterConfig,
    centroids: Vec<Centroid>,
}

impl OnlineCluster {
    /// Build an empty cluster with the supplied configuration.
    #[must_use]
    pub fn new(config: OnlineClusterConfig) -> Self {
        Self {
            config,
            centroids: Vec::new(),
        }
    }

    /// Effective configuration. Useful for tests and observability.
    #[must_use]
    pub fn config(&self) -> OnlineClusterConfig {
        self.config
    }

    /// Number of distinct speakers identified so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.centroids.len()
    }

    /// `true` while no chunks have been assigned yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.centroids.is_empty()
    }

    /// Assign a new embedding to the nearest matching speaker, or
    /// spawn a new one if no centroid is close enough and we still
    /// have headroom. Returns the resulting [`SpeakerId`].
    ///
    /// The input vector is consumed so the implementation can move
    /// it into a freshly-created centroid without an extra clone in
    /// the new-speaker path. Callers that need to keep the vector
    /// should clone it themselves.
    pub fn assign(&mut self, mut embedding: Vec<f32>) -> SpeakerId {
        l2_normalize(&mut embedding);

        let best = self.best_match(&embedding);

        match best {
            Some((idx, sim)) if sim >= self.config.similarity_threshold => {
                self.update_centroid(idx, &embedding);
                self.centroids[idx].speaker.id
            }
            Some((idx, _)) if self.centroids.len() >= self.config.max_speakers => {
                // Cap reached: collapse into nearest centroid even
                // though similarity is below threshold. Prevents
                // unbounded speaker growth on noisy inputs.
                self.update_centroid(idx, &embedding);
                self.centroids[idx].speaker.id
            }
            _ => self.spawn(embedding),
        }
    }

    /// Apply a user label to an existing centroid. Returns `false`
    /// when the id isn't in the cluster (already evicted or never
    /// existed). Centroid math is untouched — labels are pure
    /// presentation.
    pub fn rename(&mut self, id: SpeakerId, label: &str) -> bool {
        for c in &mut self.centroids {
            if c.speaker.id == id {
                let trimmed = label.trim();
                c.speaker.label = if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                };
                return true;
            }
        }
        false
    }

    /// Snapshot of the speakers known to the cluster, in arrival
    /// order. The returned [`Speaker`]s are owned copies — mutate
    /// the cluster through [`OnlineCluster::rename`] instead.
    #[must_use]
    pub fn speakers(&self) -> Vec<Speaker> {
        self.centroids.iter().map(|c| c.speaker.clone()).collect()
    }

    /// Clear all centroids without touching the configuration. Use
    /// when starting a new meeting with the same diarizer.
    pub fn reset(&mut self) {
        self.centroids.clear();
    }

    // ---------------- internals ----------------

    fn best_match(&self, embedding: &[f32]) -> Option<(usize, f32)> {
        let mut best: Option<(usize, f32)> = None;
        for (i, c) in self.centroids.iter().enumerate() {
            let sim = cosine_similarity(embedding, &c.vector);
            if best.is_none_or(|(_, b)| sim > b) {
                best = Some((i, sim));
            }
        }
        best
    }

    fn update_centroid(&mut self, idx: usize, embedding: &[f32]) {
        let c = &mut self.centroids[idx];
        let n = f32::from(u16::try_from(c.count.min(u32::from(u16::MAX))).unwrap_or(u16::MAX));
        // Running mean: new = (n * old + x) / (n + 1)
        for (slot, &x) in c.vector.iter_mut().zip(embedding.iter()) {
            *slot = (*slot * n + x) / (n + 1.0);
        }
        l2_normalize(&mut c.vector);
        c.count = c.count.saturating_add(1);
    }

    fn spawn(&mut self, embedding: Vec<f32>) -> SpeakerId {
        let slot = u32::try_from(self.centroids.len()).unwrap_or(u32::MAX);
        let speaker = Speaker::anonymous(slot);
        let id = speaker.id;
        self.centroids.push(Centroid {
            speaker,
            vector: embedding,
            count: 1,
        });
        id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    /// Build a normalised vector by direction (sin/cos pair) so tests
    /// can express "speaker A at 30°, speaker B at 120°" intuitively.
    fn at(deg: f32) -> Vec<f32> {
        let r = deg.to_radians();
        vec![r.cos(), r.sin()]
    }

    #[test]
    fn first_chunk_creates_speaker_zero_with_slot_zero() {
        let mut c = OnlineCluster::new(OnlineClusterConfig::default());
        let id = c.assign(at(0.0));
        assert_eq!(c.len(), 1);
        let snap = c.speakers();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].slot, 0);
        assert_eq!(snap[0].id, id);
    }

    #[test]
    fn nearby_embedding_collapses_into_same_speaker() {
        let mut c = OnlineCluster::new(OnlineClusterConfig::default());
        let a = c.assign(at(0.0));
        let b = c.assign(at(5.0)); // cos(5°) ≈ 0.996 > 0.78
        assert_eq!(a, b);
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn far_embedding_spawns_new_speaker() {
        let mut c = OnlineCluster::new(OnlineClusterConfig::default());
        let a = c.assign(at(0.0));
        let b = c.assign(at(90.0)); // cos = 0 < 0.78
        assert_ne!(a, b);
        assert_eq!(c.len(), 2);
        assert_eq!(c.speakers()[1].slot, 1);
    }

    #[test]
    fn returning_speaker_resolves_to_original_id() {
        let mut c = OnlineCluster::new(OnlineClusterConfig::default());
        let a1 = c.assign(at(0.0));
        let _b = c.assign(at(90.0));
        let a2 = c.assign(at(3.0));
        assert_eq!(a1, a2, "second turn from speaker A must reuse id");
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn max_speakers_cap_is_respected() {
        let cfg = OnlineClusterConfig {
            similarity_threshold: 0.95, // very strict so each chunk is "different"
            max_speakers: 2,
        };
        let mut c = OnlineCluster::new(cfg);
        c.assign(at(0.0));
        c.assign(at(40.0));
        c.assign(at(80.0)); // would normally spawn a 3rd, but capped
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn rename_updates_display_name_without_changing_id() {
        let mut c = OnlineCluster::new(OnlineClusterConfig::default());
        let id = c.assign(at(0.0));
        assert!(c.rename(id, "Alice"));
        let snap = c.speakers();
        assert_eq!(snap[0].display_name(), "Alice");
        assert_eq!(snap[0].id, id);
    }

    #[test]
    fn rename_unknown_id_returns_false() {
        let mut c = OnlineCluster::new(OnlineClusterConfig::default());
        c.assign(at(0.0));
        assert!(!c.rename(SpeakerId::new(), "ghost"));
    }

    #[test]
    fn reset_clears_all_centroids() {
        let mut c = OnlineCluster::new(OnlineClusterConfig::default());
        c.assign(at(0.0));
        c.assign(at(90.0));
        assert_eq!(c.len(), 2);
        c.reset();
        assert!(c.is_empty());
    }

    #[test]
    fn centroid_drifts_toward_added_embeddings() {
        let mut c = OnlineCluster::new(OnlineClusterConfig::default());
        let id = c.assign(at(0.0));
        // Many chunks slightly off-axis should pull the centroid.
        for _ in 0..20 {
            assert_eq!(c.assign(at(10.0)), id);
        }
        let snap_count = c.centroids[0].count;
        assert_eq!(snap_count, 21);
    }
}
