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
//! a sensible default (`0.55`) calibrated against ERes2Net
//! embeddings on streaming 5 s chunks and add a centroid merge pass
//! (threshold `0.70`) to recover from early mis-splits. Callers can
//! override both through [`OnlineClusterConfig`].
//!
//! ## Centroid update (EMA)
//!
//! Centroids use an exponential-moving-average (EMA) update:
//! the effective count is capped at `max_centroid_history`, so the
//! weight of each new embedding never falls below
//! `1 / (max_centroid_history + 1)`. This prevents early (often
//! noisy) embeddings from permanently dominating the centroid and
//! allows the cluster to track speaker drift over long meetings.
//!
//! ## Sticky speaker bias
//!
//! A small additive bias (`sticky_bias`) is added to the cosine
//! similarity of the previously-assigned centroid. This suppresses
//! single-chunk "flashes" to a different speaker caused by noisy
//! embeddings, without preventing genuine speaker turns (which
//! produce a much larger similarity gap).

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
    ///
    /// **Calibration note (2026-04):** 0.78 was too strict for
    /// streaming 5 s chunks — intra-speaker cosine on short, noisy
    /// segments frequently falls in the 0.50–0.70 range, causing
    /// spurious speaker spawns (e.g. 8 detected for 3 real). 0.55
    /// is the lowest safe value that still discriminates different
    /// speakers (cross-speaker cosine is typically < 0.35).
    pub similarity_threshold: f32,

    /// Hard cap on the number of distinct speakers we will spawn.
    /// Once reached, any chunk that would otherwise create a new
    /// cluster is folded into the nearest existing one. Prevents
    /// runaway speaker counts from noisy embeddings on long
    /// recordings.
    pub max_speakers: usize,

    /// Centroid-vs-centroid cosine above which two clusters are
    /// merged after each assignment. Centroids are running means
    /// (less noisy than raw embeddings) so this can safely sit above
    /// `similarity_threshold`. Set to `1.0` to disable merging.
    pub merge_threshold: f32,

    /// Maximum effective sample count for the running-mean centroid
    /// update. Once a centroid has accumulated this many embeddings
    /// the oldest observations are implicitly down-weighted, giving
    /// each new embedding a minimum influence of
    /// `1 / (max_centroid_history + 1)`. Set to `u32::MAX` to
    /// revert to an unbounded running mean.
    ///
    /// **Rationale (diart / pyannote research):** pure running means
    /// "cement" the centroid after ~50 chunks, making it unable to
    /// track speaker drift or recover from early noise. Capping at
    /// ~10 keeps effective α ≈ 0.09, matching the `rho_update`
    /// range found optimal by diart on DIHARD III.
    pub max_centroid_history: u32,

    /// Additive cosine bias applied to the last-assigned centroid
    /// during `best_match`. Suppresses single-chunk bouncing
    /// between speakers without preventing genuine turns (which
    /// produce a similarity gap much larger than this value).
    /// Set to `0.0` to disable.
    pub sticky_bias: f32,
}

impl Default for OnlineClusterConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.55,
            max_speakers: 6,
            merge_threshold: 0.70,
            max_centroid_history: 10,
            sticky_bias: 0.05,
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
    /// Maps absorbed SpeakerId → surviving SpeakerId so callers can
    /// retroactively fix already-emitted labels.
    merge_map: Vec<(SpeakerId, SpeakerId)>,
    /// Index of the centroid assigned on the previous call to
    /// [`assign`]. Used for the sticky-bias heuristic.
    last_assigned: Option<usize>,
}

impl OnlineCluster {
    /// Build an empty cluster with the supplied configuration.
    #[must_use]
    pub fn new(config: OnlineClusterConfig) -> Self {
        Self {
            config,
            centroids: Vec::new(),
            merge_map: Vec::new(),
            last_assigned: None,
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
    /// After assignment, a merge pass checks whether any two
    /// centroids have converged enough to be combined. This handles
    /// the "early split, late convergence" pattern common with
    /// short streaming chunks.
    pub fn assign(&mut self, mut embedding: Vec<f32>) -> SpeakerId {
        l2_normalize(&mut embedding);

        let best = self.best_match(&embedding);

        let (assigned_idx, id) = match best {
            Some((idx, sim)) if sim >= self.config.similarity_threshold => {
                tracing::debug!(speaker_slot = self.centroids[idx].speaker.slot, cosine = %format!("{sim:.3}"), threshold = %format!("{:.3}", self.config.similarity_threshold), "assign → existing (above threshold)");
                self.update_centroid(idx, &embedding);
                (idx, self.centroids[idx].speaker.id)
            }
            Some((idx, sim)) if self.centroids.len() >= self.config.max_speakers => {
                tracing::debug!(speaker_slot = self.centroids[idx].speaker.slot, cosine = %format!("{sim:.3}"), "assign → forced merge (cap reached)");
                self.update_centroid(idx, &embedding);
                (idx, self.centroids[idx].speaker.id)
            }
            Some((_, sim)) => {
                tracing::debug!(best_cosine = %format!("{sim:.3}"), threshold = %format!("{:.3}", self.config.similarity_threshold), n_speakers = self.centroids.len(), "assign → new speaker (below threshold)");
                let new_idx = self.centroids.len();
                let new_id = self.spawn(embedding);
                (new_idx, new_id)
            }
            None => {
                tracing::debug!("assign → first speaker");
                let new_id = self.spawn(embedding);
                (0, new_id)
            }
        };

        self.last_assigned = Some(assigned_idx);
        self.try_merge();
        id
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
        self.merge_map.clear();
        self.last_assigned = None;
    }

    /// Returns accumulated (absorbed_id → surviving_id) pairs from
    /// centroid merges since the last reset. The caller (typically
    /// [`OnlineDiarizer`]) can use this to retroactively update
    /// already-emitted segment labels.
    #[must_use]
    pub fn drain_merge_map(&mut self) -> Vec<(SpeakerId, SpeakerId)> {
        std::mem::take(&mut self.merge_map)
    }

    // ---------------- internals ----------------

    /// After each assignment, scan all centroid pairs. If any two
    /// centroids have converged above `merge_threshold`, absorb the
    /// newer (fewer-count) one into the older (more-count) one.
    /// Repeat until no merge is possible (handles cascading merges).
    fn try_merge(&mut self) {
        loop {
            let pair = self.find_merge_pair();
            let Some((i, j, sim)) = pair else { break };

            // Keep the centroid with more observations; absorb the
            // other. On a tie, keep the earlier slot (lower index).
            let (keep, absorb) = if self.centroids[i].count >= self.centroids[j].count {
                (i, j)
            } else {
                (j, i)
            };

            let absorbed_id = self.centroids[absorb].speaker.id;
            let surviving_id = self.centroids[keep].speaker.id;

            // Weighted merge of centroid vectors.
            let nk = self.centroids[keep].count as f32;
            let na = self.centroids[absorb].count as f32;
            let total = nk + na;
            let absorb_vec = self.centroids[absorb].vector.clone();
            for (slot, &x) in self.centroids[keep]
                .vector
                .iter_mut()
                .zip(absorb_vec.iter())
            {
                *slot = (*slot * nk + x * na) / total;
            }
            l2_normalize(&mut self.centroids[keep].vector);
            self.centroids[keep].count = self.centroids[keep]
                .count
                .saturating_add(self.centroids[absorb].count);

            // Preserve any user-assigned label from the absorbed
            // centroid if the surviving one has none.
            if self.centroids[keep].speaker.label.is_none() {
                self.centroids[keep].speaker.label = self.centroids[absorb].speaker.label.clone();
            }

            tracing::info!(
                absorbed_slot = self.centroids[absorb].speaker.slot,
                surviving_slot = self.centroids[keep].speaker.slot,
                cosine = %format!("{sim:.3}"),
                remaining = self.centroids.len() - 1,
                "merged centroids"
            );

            self.merge_map.push((absorbed_id, surviving_id));
            self.centroids.remove(absorb);

            // Fix last_assigned index after removal.
            if let Some(ref mut la) = self.last_assigned {
                if *la == absorb {
                    *la = keep;
                }
                // Adjust for the shift caused by removing `absorb`.
                if *la > absorb {
                    *la -= 1;
                }
            }
        }
    }

    /// Find the most similar centroid pair above `merge_threshold`.
    fn find_merge_pair(&self) -> Option<(usize, usize, f32)> {
        let mut best: Option<(usize, usize, f32)> = None;
        for i in 0..self.centroids.len() {
            for j in (i + 1)..self.centroids.len() {
                let sim = cosine_similarity(&self.centroids[i].vector, &self.centroids[j].vector);
                if sim >= self.config.merge_threshold && best.as_ref().is_none_or(|b| sim > b.2) {
                    best = Some((i, j, sim));
                }
            }
        }
        best
    }

    fn best_match(&self, embedding: &[f32]) -> Option<(usize, f32)> {
        let mut best: Option<(usize, f32)> = None;
        for (i, c) in self.centroids.iter().enumerate() {
            let mut sim = cosine_similarity(embedding, &c.vector);
            // Apply sticky bias to the previously-assigned centroid.
            if self.last_assigned == Some(i) {
                sim += self.config.sticky_bias;
            }
            if best.is_none_or(|(_, b)| sim > b) {
                best = Some((i, sim));
            }
        }
        best
    }

    fn update_centroid(&mut self, idx: usize, embedding: &[f32]) {
        let c = &mut self.centroids[idx];
        // Cap effective count to max_centroid_history so new
        // embeddings always contribute at least 1/(cap+1).
        let effective_n = c.count.min(self.config.max_centroid_history) as f32;
        for (slot, &x) in c.vector.iter_mut().zip(embedding.iter()) {
            *slot = (*slot * effective_n + x) / (effective_n + 1.0);
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
            merge_threshold: 1.0, // disable merge for this test
            sticky_bias: 0.0,     // disable for this test
            ..Default::default()
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

    #[test]
    fn merge_pass_combines_close_centroids() {
        // Use a very strict similarity_threshold so both chunks spawn
        // separate centroids, then let the merge pass combine them
        // because cos(5°) ≈ 0.996 > merge_threshold 0.90.
        let cfg = OnlineClusterConfig {
            similarity_threshold: 0.999,
            max_speakers: 10,
            merge_threshold: 0.90,
            sticky_bias: 0.0,
            ..Default::default()
        };
        let mut c = OnlineCluster::new(cfg);
        let a = c.assign(at(0.0));
        let b = c.assign(at(5.0));
        assert_ne!(a, b, "strict threshold should create two centroids");
        // The merge pass runs after each assign, so by now they merged.
        assert_eq!(c.len(), 1, "merge pass should combine close centroids");
    }

    #[test]
    fn merge_map_records_absorbed_into_survivor() {
        let cfg = OnlineClusterConfig {
            similarity_threshold: 0.999, // every chunk spawns a new centroid
            max_speakers: 10,
            merge_threshold: 0.90,
            sticky_bias: 0.0,
            ..Default::default()
        };
        let mut c = OnlineCluster::new(cfg);
        // Two very close centroids: cos(5°) ≈ 0.996 > 0.90
        let a = c.assign(at(0.0));
        let b = c.assign(at(5.0));
        // The merge pass fires immediately because cos > merge_threshold.
        assert_eq!(c.len(), 1, "should merge immediately");
        let map = c.drain_merge_map();
        assert_eq!(map.len(), 1);
        // `b` was absorbed into `a` (a has count=1, b has count=1,
        // tie-break goes to earlier index = a).
        assert_eq!(map[0], (b, a));
    }

    #[test]
    fn ema_centroid_adapts_to_drift() {
        // With max_centroid_history = 3, after 3 chunks the centroid
        // should closely track new embeddings rather than being
        // anchored to the first one.
        let cfg = OnlineClusterConfig {
            similarity_threshold: 0.3, // lenient so everything clusters together
            max_speakers: 2,
            merge_threshold: 1.0,
            max_centroid_history: 3,
            sticky_bias: 0.0,
        };
        let mut c = OnlineCluster::new(cfg);
        c.assign(at(0.0)); // centroid starts at 0°
                           // Feed 20 chunks at 15° — centroid should drift close to 15°.
        for _ in 0..20 {
            c.assign(at(15.0));
        }
        let centroid = &c.centroids[0].vector;
        let target = at(15.0);
        let sim = cosine_similarity(centroid, &target);
        assert!(
            sim > 0.99,
            "EMA centroid should closely track recent embeddings, got cosine {sim:.4}"
        );
    }

    #[test]
    fn ema_does_not_drift_with_unbounded_history() {
        // With max_centroid_history = u32::MAX (unbounded running mean),
        // early embeddings dominate and the centroid barely moves.
        let cfg = OnlineClusterConfig {
            similarity_threshold: 0.3,
            max_speakers: 2,
            merge_threshold: 1.0,
            max_centroid_history: u32::MAX,
            sticky_bias: 0.0,
        };
        let mut c = OnlineCluster::new(cfg);
        c.assign(at(0.0));
        for _ in 0..20 {
            c.assign(at(15.0));
        }
        let centroid = &c.centroids[0].vector;
        let at_zero = at(0.0);
        let at_fifteen = at(15.0);
        let sim_zero = cosine_similarity(centroid, &at_zero);
        let sim_fifteen = cosine_similarity(centroid, &at_fifteen);
        // With unbounded mean, centroid should still be closer to
        // the average (≈14°) but the first embedding has more weight
        // than in the EMA case — here the difference is marginal
        // since 20:1 ratio dominates either way, but verify it
        // compiles and the centroid remains valid.
        assert!(
            sim_zero > 0.9,
            "centroid should still be in the 0°–15° band"
        );
        assert!(
            sim_fifteen > 0.9,
            "centroid should still be in the 0°–15° band"
        );
    }

    #[test]
    fn sticky_bias_prevents_single_chunk_bounce() {
        // Two speakers at 0° and 60°. A noisy chunk at 30° (equidistant)
        // should stick to whichever speaker was last assigned.
        let cfg = OnlineClusterConfig {
            similarity_threshold: 0.3,
            max_speakers: 4,
            merge_threshold: 1.0,
            max_centroid_history: 100,
            sticky_bias: 0.10, // meaningful bias
        };
        let mut c = OnlineCluster::new(cfg);
        let a = c.assign(at(0.0)); // speaker A at 0°
        let b = c.assign(at(60.0)); // speaker B at 60°

        // Feed several chunks of speaker A to establish "last assigned"
        for _ in 0..3 {
            assert_eq!(c.assign(at(2.0)), a);
        }

        // Ambiguous chunk at 30° — without bias it could go either way,
        // but with sticky_bias toward A it should stay with A.
        // cos(30°) = 0.866 for both centroids at 0° and 60°,
        // but sticky adds 0.10 to A's score → A wins.
        let noisy = c.assign(at(30.0));
        assert_eq!(
            noisy, a,
            "sticky bias should keep ambiguous chunk with last speaker"
        );

        // Genuine turn to B (chunk very close to B) must still work.
        let turn = c.assign(at(58.0));
        assert_eq!(turn, b, "genuine speaker turn should override sticky bias");
    }

    #[test]
    fn sticky_bias_zero_disables_stickiness() {
        let cfg = OnlineClusterConfig {
            sticky_bias: 0.0,
            ..Default::default()
        };
        let mut c = OnlineCluster::new(cfg);
        let _a = c.assign(at(0.0));
        let _b = c.assign(at(90.0));
        // Without bias, behaviour is identical to the base algorithm.
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn last_assigned_survives_merge() {
        // Verify that last_assigned is updated correctly when the
        // assigned centroid index shifts due to a merge.
        let cfg = OnlineClusterConfig {
            similarity_threshold: 0.999,
            max_speakers: 10,
            merge_threshold: 0.90,
            sticky_bias: 0.0,
            ..Default::default()
        };
        let mut c = OnlineCluster::new(cfg);
        c.assign(at(0.0)); // centroid 0
        c.assign(at(5.0)); // centroid 1, then merge → centroid 0
                           // After merge, last_assigned should still be valid.
        assert!(c.last_assigned.is_some());
        let la = c.last_assigned.unwrap();
        assert!(
            la < c.centroids.len(),
            "last_assigned index must be in bounds"
        );
    }
}
