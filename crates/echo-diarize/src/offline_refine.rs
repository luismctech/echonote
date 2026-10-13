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
//! Once the session finishes we hold *all* segment-level embeddings and
//! can do a proper **agglomerative hierarchical clustering (AHC)** pass:
//!
//! 1. Group every accumulated embedding under its online speaker,
//!    dropping speakers below `min_embeddings`.
//! 2. Repeatedly merge the pair of groups with the highest
//!    **average-linkage** similarity (the mean cosine over *all*
//!    cross-group segment pairs) while it clears `merge_threshold`.
//! 3. Return a list of `(absorbed_speaker_id, surviving_speaker_id)`
//!    pairs. The caller applies these to all stored segments.
//!
//! ## Why average-linkage over segments, not mean-to-mean?
//!
//! An earlier version collapsed each speaker to a single mean embedding
//! and compared means. That throws away intra-speaker variance and lets
//! one skewed segment drag the centroid across the merge threshold —
//! the exact failure mode of single-mic Spanish audio, where distinct
//! speakers already sit in the hot 0.80–0.90 cosine band. Average-
//! linkage scores a candidate merge from the *whole distribution* of
//! cross-speaker segment pairs, so a handful of coincidentally-close
//! segments no longer forces a merge, and a genuine over-split (where
//! most pairs are close) still collapses.
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
//! here — splitting would need per-segment reassignment and a storage
//! contract richer than the `(absorbed, surviving)` merge pairs this
//! pass emits (roadmap item).

use echo_domain::{Speaker, SpeakerId};
use tracing::info;

use crate::embedding::cosine_similarity;

/// Configuration for [`refine_speakers`].
#[derive(Debug, Clone, Copy)]
pub struct OfflineRefineConfig {
    /// Average-linkage cosine similarity above which two speaker groups
    /// are merged. Should be slightly higher than the online
    /// `merge_threshold` because the full-session view is less noisy
    /// than EMA centroids on short chunks.
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
            // Conservative fallback for embedders without a calibrated
            // preset (EN CAM++, ERes2Net): it matches the EN CAM++ online
            // threshold and rarely merges. Calibrated families override
            // it in `echo_diarize::presets` (multilingual CAM++: 0.50,
            // DER-calibrated on held-out data; at 0.85 the pass never
            // merged anything in the T6 benchmark).
            merge_threshold: 0.85,
            min_embeddings: 2,
        }
    }
}

/// A group of segment-level embeddings the AHC pass treats as one
/// speaker. Starts as a single online speaker and accretes members as
/// groups merge. Similarity is computed from the members directly
/// (average-linkage), so a skewed mean can't drag a merge the way a
/// centroid-only comparison could.
#[derive(Debug)]
struct SegmentGroup {
    /// Surviving speaker id — the representative that absorbed ids fold
    /// into.
    id: SpeakerId,
    /// Compact indices into the participating-embeddings array.
    members: Vec<usize>,
}

/// Mean cosine over every cross-group segment pair (average-linkage).
/// `sim` is a flat `n × n` symmetric matrix of pairwise cosines.
fn average_linkage(sim: &[f32], n: usize, a: &[usize], b: &[usize]) -> f32 {
    debug_assert!(!a.is_empty() && !b.is_empty());
    let mut total = 0.0_f32;
    for &x in a {
        for &y in b {
            total += sim[x * n + y];
        }
    }
    total / (a.len() * b.len()) as f32
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

    // Gather the participating embeddings (those whose online speaker
    // cleared `min_embeddings`) into one compact array, and build one
    // group per qualifying speaker in slot order so tie-breaks are
    // deterministic.
    let mut emb: Vec<&[f32]> = Vec::new();
    let mut groups: Vec<SegmentGroup> = Vec::new();
    for s in speakers {
        let start = emb.len();
        for (e, sid) in records {
            if *sid == s.id {
                emb.push(e.as_slice());
            }
        }
        if emb.len() - start >= config.min_embeddings {
            groups.push(SegmentGroup {
                id: s.id,
                members: (start..emb.len()).collect(),
            });
        }
        // Speakers below the floor keep the embeddings we pushed in the
        // array (harmless — they're simply never referenced by a group).
    }

    if groups.len() < 2 {
        return Vec::new();
    }

    // Precompute the symmetric pairwise-cosine matrix once so the AHC
    // loop only sums cached values.
    let n = emb.len();
    let mut sim = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let s = cosine_similarity(emb[i], emb[j]);
            sim[i * n + j] = s;
            sim[j * n + i] = s;
        }
    }

    // AHC: greedily merge the highest average-linkage pair above the
    // threshold until none remain.
    let mut merges: Vec<(SpeakerId, SpeakerId)> = Vec::new();

    loop {
        // Find the pair (i, j), i < j, with the highest average-linkage.
        let mut best: Option<(usize, usize, f32)> = None;
        for i in 0..groups.len() {
            for j in (i + 1)..groups.len() {
                let link = average_linkage(&sim, n, &groups[i].members, &groups[j].members);
                if link >= config.merge_threshold && best.as_ref().is_none_or(|&(_, _, b)| link > b)
                {
                    best = Some((i, j, link));
                }
            }
        }

        let Some((i, j, link)) = best else { break };

        // Keep the larger group (earlier slot on tie, since i < j).
        let (keep, absorb) = if groups[i].members.len() >= groups[j].members.len() {
            (i, j)
        } else {
            (j, i)
        };

        let absorbed_id = groups[absorb].id;
        let surviving_id = groups[keep].id;

        let absorbed_members = std::mem::take(&mut groups[absorb].members);
        groups[keep].members.extend(absorbed_members);
        groups.remove(absorb);

        info!(
            absorbed = %absorbed_id,
            surviving = %surviving_id,
            avg_linkage = %format!("{link:.3}"),
            remaining = groups.len(),
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
    fn average_linkage_does_not_merge_on_a_close_mean_alone() {
        let a = make_speaker(0);
        let b = make_speaker(1);
        // Group A sits at 0°; group B straddles 20°/40°, so its *mean*
        // lands at 30° (cos ≈ 0.866 vs A) — a mean-to-mean pass would
        // merge at threshold 0.86. But averaged over every A×B segment
        // pair the linkage is only ≈0.853, so average-linkage must not.
        let records = vec![
            (unit_vec(0.0), a.id),
            (unit_vec(0.0), a.id),
            (unit_vec(20.0), b.id),
            (unit_vec(40.0), b.id),
        ];
        let result = refine_speakers(
            &records,
            &[a, b],
            OfflineRefineConfig {
                merge_threshold: 0.86,
                min_embeddings: 2,
            },
        );
        assert!(
            result.is_empty(),
            "average-linkage must not merge on a coincidentally-close mean"
        );
    }

    #[test]
    fn chained_over_splits_collapse_into_one() {
        let a = make_speaker(0);
        let b = make_speaker(1);
        let c = make_speaker(2);
        // Three online clusters of the same speaker within a few degrees
        // — every cross pair clears the threshold, so all three collapse.
        let records = vec![
            (unit_vec(0.0), a.id),
            (unit_vec(1.0), a.id),
            (unit_vec(2.0), b.id),
            (unit_vec(3.0), b.id),
            (unit_vec(4.0), c.id),
            (unit_vec(5.0), c.id),
        ];
        let result = refine_speakers(
            &records,
            &[a.clone(), b.clone(), c.clone()],
            OfflineRefineConfig {
                merge_threshold: 0.9,
                min_embeddings: 2,
            },
        );
        // Two merges fold three groups into one; the earliest slot (a)
        // survives every merge.
        assert_eq!(result.len(), 2, "three groups need two merges");
        for (_, surviving) in &result {
            assert_eq!(*surviving, a.id, "everything folds into slot 0");
        }
        let absorbed: std::collections::HashSet<_> = result.iter().map(|(x, _)| *x).collect();
        assert!(
            absorbed.contains(&b.id) && absorbed.contains(&c.id),
            "both b and c must be absorbed"
        );
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
