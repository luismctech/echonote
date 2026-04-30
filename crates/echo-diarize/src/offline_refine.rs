//! Offline speaker re-clustering pass for finished sessions.
//!
//! During streaming, the [`OnlineDiarizer`] assigns speaker labels to
//! each chunk using an online threshold-based algorithm that has two
//! known biases:
//!
//! - **EMA centroid drift** — early embeddings anchor the centroid via
//!   an exponential moving average; a slightly noisy first chunk can
//!   permanently skew the speaker fingerprint.
//! - **Sticky bias** — a per-chunk additive bonus that prevents
//!   single-chunk bouncing between speakers, but can delay genuine
//!   turn detection by one chunk.
//!
//! Once the session finishes we hold *all* embeddings and can do a
//! proper **agglomerative hierarchical clustering (AHC)** pass:
//!
//! 1. Compute the unweighted mean embedding for each online speaker.
//! 2. Build a speaker × speaker cosine-similarity matrix.
//! 3. Repeatedly merge the most-similar pair above `merge_threshold`
//!    until no such pair exists (complete linkage AHC).
//! 4. Return a list of `(absorbed_speaker_id, surviving_speaker_id)`
//!    pairs. The caller applies these to all stored segments.
//!
//! ## When this helps
//!
//! - A speaker with a noisy first 5-second chunk gets incorrectly
//!   split into two online clusters, then the AHC mean rebalances them.
//! - Two speakers talk in close alternation and the online algorithm
//!   merges them; the AHC mean (across the whole session) separates
//!   them more cleanly when enough embeddings are present.
//! - Long meetings where one speaker's voice drifts (fatigue, emotion)
//!   — the full-session mean is more stable than the EMA centroid.
//!
//! ## Limitations
//!
//! This pass only *merges* clusters; it does not split them.  A
//! genuinely mis-merged pair from the online pass cannot be recovered
//! here.  Splitting requires a segmentation model (see the planned
//! pyannote adapter in the roadmap).

use echo_domain::{Speaker, SpeakerId};
use tracing::info;

use crate::embedding::{cosine_similarity, l2_normalize};

/// Configuration for [`refine_speakers`].
#[derive(Debug, Clone, Copy)]
pub struct OfflineRefineConfig {
    /// Cosine similarity above which two offline cluster means are
    /// merged. Should be slightly higher than the online
    /// `merge_threshold` because offline means are less noisy than
    /// EMA centroids on short chunks.
    pub merge_threshold: f32,
    /// Minimum number of embeddings a speaker must have contributed
    /// to participate in the offline pass. Speakers with fewer
    /// embeddings than this are left unchanged (they may be noise or
    /// very short utterances where the mean would be unreliable).
    pub min_embeddings: usize,
}

impl Default for OfflineRefineConfig {
    fn default() -> Self {
        Self {
            merge_threshold: 0.75,
            min_embeddings: 2,
        }
    }
}

/// One per online speaker: the unweighted mean of all embeddings
/// assigned to that speaker during streaming.
#[derive(Debug)]
struct OfflineCluster {
    id: SpeakerId,
    /// L2-normalised mean embedding computed from all records assigned
    /// to this speaker. `None` when the speaker had fewer embeddings
    /// than `min_embeddings`.
    mean: Option<Vec<f32>>,
    count: usize,
}

/// Re-cluster the accumulated streaming embeddings and return a list
/// of `(absorbed_id, surviving_id)` merges.
///
/// # Arguments
///
/// * `records` — All `(L2-normalised embedding, online SpeakerId)`
///   pairs collected during the session. Empty records → empty return.
/// * `speakers` — The online speaker list from
///   [`OnlineDiarizer::speakers()`]. Used to preserve ordering so the
///   AHC tie-break (keep the earlier slot) is deterministic.
/// * `config` — Merge threshold and minimum embedding count.
///
/// # Returns
///
/// A list of `(absorbed_id, surviving_id)` pairs in merge order.
/// The caller should walk the list and update segment speaker labels
/// accordingly, replacing every occurrence of `absorbed_id` with
/// `surviving_id`. The list may be empty when no merges were found.
pub fn refine_speakers(
    records: &[(Vec<f32>, SpeakerId)],
    speakers: &[Speaker],
    config: OfflineRefineConfig,
) -> Vec<(SpeakerId, SpeakerId)> {
    if records.is_empty() || speakers.len() < 2 {
        return Vec::new();
    }

    // Build unweighted mean embeddings per speaker, in slot order so
    // the AHC tie-break is deterministic.
    let mut clusters: Vec<OfflineCluster> = speakers
        .iter()
        .map(|s| OfflineCluster {
            id: s.id,
            mean: None,
            count: 0,
        })
        .collect();

    let dim = records[0].0.len();

    for (emb, speaker_id) in records {
        let Some(c) = clusters.iter_mut().find(|c| c.id == *speaker_id) else {
            continue;
        };
        let mean = c.mean.get_or_insert_with(|| vec![0.0_f32; dim]);
        for (m, &x) in mean.iter_mut().zip(emb.iter()) {
            *m += x;
        }
        c.count += 1;
    }

    // Normalise each mean.
    for c in &mut clusters {
        if c.count < config.min_embeddings {
            c.mean = None;
            continue;
        }
        if let Some(ref mut m) = c.mean {
            let inv = 1.0 / c.count as f32;
            for x in m.iter_mut() {
                *x *= inv;
            }
            l2_normalize(m);
        }
    }

    // AHC: greedily merge the most-similar pair above the threshold.
    let mut merges: Vec<(SpeakerId, SpeakerId)> = Vec::new();

    loop {
        // Find the pair (i, j) with highest cosine similarity, i < j.
        let mut best: Option<(usize, usize, f32)> = None;
        for i in 0..clusters.len() {
            for j in (i + 1)..clusters.len() {
                let (Some(mi), Some(mj)) = (&clusters[i].mean, &clusters[j].mean) else {
                    continue;
                };
                let sim = cosine_similarity(mi, mj);
                if sim >= config.merge_threshold && best.as_ref().is_none_or(|&(_, _, b)| sim > b) {
                    best = Some((i, j, sim));
                }
            }
        }

        let Some((i, j, sim)) = best else { break };

        // Keep the cluster with more observations (earlier slot on tie).
        let (keep, absorb) = if clusters[i].count >= clusters[j].count {
            (i, j)
        } else {
            (j, i)
        };

        let absorbed_id = clusters[absorb].id;
        let surviving_id = clusters[keep].id;

        // Weighted merge of the means.
        let nk = clusters[keep].count as f32;
        let na = clusters[absorb].count as f32;
        let total = nk + na;
        let absorb_mean = clusters[absorb].mean.clone().unwrap_or_default();
        if let Some(ref mut km) = clusters[keep].mean {
            for (slot, &x) in km.iter_mut().zip(absorb_mean.iter()) {
                *slot = (*slot * nk + x * na) / total;
            }
            l2_normalize(km);
        }
        clusters[keep].count = clusters[keep].count.saturating_add(clusters[absorb].count);
        clusters.remove(absorb);

        info!(
            absorbed = %absorbed_id,
            surviving = %surviving_id,
            cosine = %format!("{sim:.3}"),
            remaining = clusters.len(),
            "offline AHC merged speakers"
        );

        merges.push((absorbed_id, surviving_id));
    }

    merges
}

#[cfg(test)]
mod tests {
    use super::*;
    use echo_domain::Speaker;

    fn unit_vec(angle_deg: f32) -> Vec<f32> {
        let r = angle_deg.to_radians();
        vec![r.cos(), r.sin()]
    }

    fn make_speaker(slot: u32) -> Speaker {
        Speaker::anonymous(slot)
    }

    #[test]
    fn no_records_returns_empty() {
        let speakers = vec![make_speaker(0), make_speaker(1)];
        let result = refine_speakers(&[], &speakers, OfflineRefineConfig::default());
        assert!(result.is_empty());
    }

    #[test]
    fn single_speaker_returns_empty() {
        let s = make_speaker(0);
        let records = vec![(unit_vec(0.0), s.id), (unit_vec(5.0), s.id)];
        let result = refine_speakers(&records, &[s], OfflineRefineConfig::default());
        assert!(result.is_empty(), "single speaker cannot be merged");
    }

    #[test]
    fn clearly_separated_speakers_are_not_merged() {
        let a = make_speaker(0);
        let b = make_speaker(1);
        // 90° apart → cosine = 0, well below any merge threshold.
        let records = vec![
            (unit_vec(0.0), a.id),
            (unit_vec(2.0), a.id),
            (unit_vec(90.0), b.id),
            (unit_vec(92.0), b.id),
        ];
        let result = refine_speakers(
            &records,
            &[a, b],
            OfflineRefineConfig {
                merge_threshold: 0.75,
                min_embeddings: 2,
            },
        );
        assert!(result.is_empty(), "orthogonal speakers must not be merged");
    }

    #[test]
    fn over_split_speakers_are_merged() {
        let a = make_speaker(0);
        let b = make_speaker(1);
        // Both clusters are very close (5° apart) — the online pass
        // over-split them; the offline pass should merge them.
        let records = vec![
            (unit_vec(0.0), a.id),
            (unit_vec(1.0), a.id),
            (unit_vec(4.0), b.id),
            (unit_vec(5.0), b.id),
        ];
        let result = refine_speakers(
            &records,
            &[a.clone(), b.clone()],
            OfflineRefineConfig {
                merge_threshold: 0.95, // cos(5°) ≈ 0.996 → merge
                min_embeddings: 2,
            },
        );
        assert_eq!(result.len(), 1, "should produce exactly one merge");
        let (absorbed, surviving) = result[0];
        // a has count=2, b has count=2 — tie → keep earlier slot (a).
        assert_eq!(surviving, a.id);
        assert_eq!(absorbed, b.id);
    }

    #[test]
    fn sparse_speaker_below_min_embeddings_is_skipped() {
        let a = make_speaker(0);
        let b = make_speaker(1);
        // Speaker b has only 1 embedding → below min_embeddings=2.
        // Even if their means would be close, b must not participate.
        let records = vec![
            (unit_vec(0.0), a.id),
            (unit_vec(2.0), a.id),
            (unit_vec(1.0), b.id), // only one
        ];
        let result = refine_speakers(
            &records,
            &[a, b],
            OfflineRefineConfig {
                merge_threshold: 0.95,
                min_embeddings: 2,
            },
        );
        assert!(
            result.is_empty(),
            "speaker with fewer than min_embeddings must be excluded"
        );
    }
}
