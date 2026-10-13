//! End-of-session re-segmentation over the embedding log (ADR 0001, D2).
//!
//! Unlike the merge-only refine ([`crate::offline_refine`]), which can
//! only fold whole online speakers together, re-segmentation gives every
//! logged span a new label. The labels are then mapped back to the online
//! speaker ids by maximum-overlap (duration-weighted Hungarian) matching,
//! so a matched speaker keeps its id, slot, colour and user label. Only
//! unmatched clusters get a fresh id.
//!
//! The clustering itself sits behind [`OfflineReclusterer`]. T9 ships
//! [`AhcReclusterer`] (average-linkage AHC over the log entries); VBx is
//! the planned second implementation.

use std::time::Instant;

use echo_domain::{
    DomainError, RefineBudget, RefineOutcome, SpanRelabelPlan, Speaker, SpeakerId,
    SpeakerRetirement,
};

use crate::embedding::{cosine_similarity, l2_normalize};
use crate::embedding_log::EmbeddingLogEntry;
use crate::metrics::max_weight_assignment;

/// Fewer logged entries than this: the session is too short to
/// re-segment and the refine returns [`RefineOutcome::Unchanged`].
/// Initial value from ADR 0001 (D7); T10 calibrates it.
pub const RESEGMENT_MIN_ENTRIES: usize = 4;

/// Less logged speech than this (milliseconds): too short to
/// re-segment. Initial value from ADR 0001 (D7); T10 calibrates it.
pub const RESEGMENT_MIN_SPEECH_MS: u64 = 10_000;

/// At most this many entries take part in the AHC itself (a condensed
/// f32 matrix of about 33.5 MB). Longer logs are strided; the remaining
/// entries join the nearest cluster centroid.
pub const AHC_MAX_REPRESENTATIVES: usize = 4_096;

/// Re-clusters the logged embeddings. One label (cluster index) per
/// entry, in entry order.
pub(crate) trait OfflineReclusterer {
    /// Cluster `entries`. Must return an error, never a partial result,
    /// once `budget.deadline` has passed.
    fn recluster(
        &self,
        entries: &[EmbeddingLogEntry],
        budget: RefineBudget,
    ) -> Result<Vec<usize>, DomainError>;
}

/// Average-linkage AHC over the log entries, cut at a cosine similarity.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AhcReclusterer {
    /// Two clusters merge while their average-linkage cosine similarity
    /// is at least this value.
    pub merge_threshold: f32,
}

impl OfflineReclusterer for AhcReclusterer {
    fn recluster(
        &self,
        entries: &[EmbeddingLogEntry],
        budget: RefineBudget,
    ) -> Result<Vec<usize>, DomainError> {
        let n = entries.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let stride = n.div_ceil(AHC_MAX_REPRESENTATIVES);
        let reps: Vec<usize> = (0..n).step_by(stride).collect();
        let rep_emb: Vec<&[f32]> = reps.iter().map(|&i| &*entries[i].embedding).collect();
        let rep_labels = average_linkage_labels(&rep_emb, self.merge_threshold, budget.deadline)?;
        if stride == 1 {
            return Ok(rep_labels);
        }

        // Strided: every other entry joins the nearest centroid.
        let k = rep_labels.iter().max().map_or(0, |m| m + 1);
        let dim = rep_emb[0].len();
        let mut centroids = vec![vec![0.0_f32; dim]; k];
        for (e, &label) in rep_emb.iter().zip(&rep_labels) {
            for (c, x) in centroids[label].iter_mut().zip(e.iter()) {
                *c += x;
            }
        }
        for c in &mut centroids {
            l2_normalize(c);
        }
        check_deadline(budget.deadline)?;
        let mut labels = vec![0_usize; n];
        for (r, &i) in reps.iter().enumerate() {
            labels[i] = rep_labels[r];
        }
        for (i, label) in labels.iter_mut().enumerate() {
            if i % stride != 0 {
                *label = nearest(&centroids, &entries[i].embedding);
            }
        }
        Ok(labels)
    }
}

fn nearest(centroids: &[Vec<f32>], v: &[f32]) -> usize {
    centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, cosine_similarity(c, v)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map_or(0, |(i, _)| i)
}

fn check_deadline(deadline: Instant) -> Result<(), DomainError> {
    if Instant::now() >= deadline {
        return Err(DomainError::DiarizationFailed(
            "offline refine deadline exceeded".into(),
        ));
    }
    Ok(())
}

/// Index into a condensed upper-triangular matrix, `i != j`.
fn tri(n: usize, i: usize, j: usize) -> usize {
    let (i, j) = if i < j { (i, j) } else { (j, i) };
    i * n - i * (i + 1) / 2 + (j - i - 1)
}

/// Average-linkage (UPGMA) agglomerative clustering of `emb`, cut where
/// the linkage cosine similarity drops below `threshold`.
///
/// Nearest-neighbour-chain algorithm with Lance-Williams updates:
/// O(n^2) time and one condensed f32 distance matrix. Average linkage
/// is reducible, so the merges it emits form a monotone dendrogram and
/// cutting it is a union-find over the merges below the cut.
/// Labels are compact (`0..k`) in order of first appearance.
fn average_linkage_labels(
    emb: &[&[f32]],
    threshold: f32,
    deadline: Instant,
) -> Result<Vec<usize>, DomainError> {
    let n = emb.len();
    if n <= 1 {
        return Ok(vec![0; n]);
    }
    let mut dist = vec![0.0_f32; n * (n - 1) / 2];
    for i in 0..n {
        for j in (i + 1)..n {
            dist[tri(n, i, j)] = 1.0 - cosine_similarity(emb[i], emb[j]);
        }
        if i % 256 == 0 {
            check_deadline(deadline)?;
        }
    }
    let cut = 1.0 - threshold;

    let mut size = vec![1_u32; n];
    let mut active = vec![true; n];
    let mut parent: Vec<usize> = (0..n).collect();
    let mut chain: Vec<usize> = Vec::with_capacity(n);
    for _ in 1..n {
        check_deadline(deadline)?;
        if chain.is_empty() {
            chain.push(active.iter().position(|&a| a).unwrap_or(0));
        }
        // Grow the chain until two clusters are reciprocal nearest
        // neighbours. Preferring the previous chain element on ties
        // guarantees termination.
        let (a, b) = loop {
            let a = chain[chain.len() - 1];
            let prev = chain.len().checked_sub(2).map(|p| chain[p]);
            let mut best = prev;
            let mut best_d = prev.map_or(f32::INFINITY, |p| dist[tri(n, a, p)]);
            for c in 0..n {
                if c != a && active[c] && Some(c) != prev {
                    let d = dist[tri(n, a, c)];
                    if d < best_d {
                        best_d = d;
                        best = Some(c);
                    }
                }
            }
            let b = best.expect("at least two active clusters");
            if Some(b) == prev {
                break (a, b);
            }
            chain.push(b);
        };
        chain.truncate(chain.len() - 2);

        let d_ab = dist[tri(n, a, b)];
        // `b` absorbs `a`: Lance-Williams average-linkage update.
        let (sa, sb) = (size[a] as f32, size[b] as f32);
        for c in 0..n {
            if c != a && c != b && active[c] {
                let merged = (sa * dist[tri(n, a, c)] + sb * dist[tri(n, b, c)]) / (sa + sb);
                dist[tri(n, b, c)] = merged;
            }
        }
        active[a] = false;
        size[b] += size[a];
        if d_ab <= cut {
            union(&mut parent, a, b);
        }
    }

    let mut labels = vec![usize::MAX; n];
    let mut ids: Vec<(usize, usize)> = Vec::new();
    for (i, label) in labels.iter_mut().enumerate() {
        let root = find(&mut parent, i);
        *label = match ids.iter().find(|(r, _)| *r == root) {
            Some(&(_, id)) => id,
            None => {
                ids.push((root, ids.len()));
                ids.len() - 1
            }
        };
    }
    Ok(labels)
}

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

fn union(parent: &mut [usize], a: usize, b: usize) {
    let (ra, rb) = (find(parent, a), find(parent, b));
    if ra != rb {
        parent[ra] = rb;
    }
}

/// `true` when the log is too short to re-segment (ADR 0001, D7).
pub(crate) fn too_short(entries: &[EmbeddingLogEntry]) -> bool {
    let speech_ms: u64 = entries.iter().map(|e| u64::from(e.duration_ms())).sum();
    entries.len() < RESEGMENT_MIN_ENTRIES || speech_ms < RESEGMENT_MIN_SPEECH_MS
}

/// Turn per-entry cluster labels into a span plan with stable ids.
///
/// Clusters are matched to online speakers by maximum total shared
/// duration (Hungarian). A matched cluster keeps the online id; an
/// unmatched one gets a fresh id at the next free slot. Online speakers
/// left without spans are retired, each pointing at the speaker that
/// took most of its duration. Returns [`RefineOutcome::Unchanged`] when
/// no span changes speaker.
pub(crate) fn relabel_plan(
    entries: &[EmbeddingLogEntry],
    speakers: &[Speaker],
    labels: &[usize],
) -> RefineOutcome {
    debug_assert_eq!(entries.len(), labels.len());
    let k = labels.iter().max().map_or(0, |m| m + 1);
    let column = |id: SpeakerId| speakers.iter().position(|s| s.id == id);

    let mut weights = vec![vec![0.0_f64; speakers.len()]; k];
    for (e, &label) in entries.iter().zip(labels) {
        if let Some(col) = column(e.online_speaker) {
            weights[label][col] += f64::from(e.duration_ms().max(1));
        }
    }
    let matching = max_weight_assignment(&weights);

    let mut next_slot = speakers.iter().map(|s| s.slot + 1).max().unwrap_or(0);
    let mut created = Vec::new();
    let cluster_ids: Vec<SpeakerId> = (0..k)
        .map(|c| match matching.get(c).copied().flatten() {
            Some(col) if weights[c][col] > 0.0 => speakers[col].id,
            _ => {
                let speaker = Speaker::anonymous(next_slot);
                next_slot += 1;
                created.push(speaker.clone());
                speaker.id
            }
        })
        .collect();

    let mut changed = false;
    let mut spans = Vec::new();
    for (e, &label) in entries.iter().zip(labels) {
        let id = cluster_ids[label];
        changed |= id != e.online_speaker;
        spans.extend(e.spans.iter().map(|&s| (s, id)));
    }
    if !changed {
        return RefineOutcome::Unchanged;
    }

    let retired = speakers
        .iter()
        .filter(|s| !cluster_ids.contains(&s.id))
        .map(|s| {
            let mut taken: Vec<(SpeakerId, u64)> = Vec::new();
            for (e, &label) in entries.iter().zip(labels) {
                if e.online_speaker == s.id {
                    let to = cluster_ids[label];
                    let ms = u64::from(e.duration_ms());
                    match taken.iter_mut().find(|(id, _)| *id == to) {
                        Some((_, acc)) => *acc += ms,
                        None => taken.push((to, ms)),
                    }
                }
            }
            SpeakerRetirement {
                speaker: s.id,
                majority_recipient: taken
                    .into_iter()
                    .max_by_key(|(_, ms)| *ms)
                    .map(|(id, _)| id),
            }
        })
        .collect();

    RefineOutcome::Relabel(SpanRelabelPlan {
        spans,
        created,
        retired,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use echo_domain::SpanRef;
    use pretty_assertions::assert_eq;
    use std::time::Duration;

    fn unit(angle_deg: f32) -> Vec<f32> {
        let r = angle_deg.to_radians();
        vec![r.cos(), r.sin()]
    }

    fn entry(i: u32, speaker: SpeakerId, angle: f32) -> EmbeddingLogEntry {
        EmbeddingLogEntry {
            spans: vec![SpanRef {
                chunk_index: i,
                span_index: 0,
            }],
            start_ms: i * 3_000,
            end_ms: (i + 1) * 3_000,
            epoch: 0,
            local_speaker: None,
            online_speaker: speaker,
            embedding: unit(angle).into_boxed_slice(),
        }
    }

    fn far_budget() -> RefineBudget {
        RefineBudget {
            deadline: Instant::now() + Duration::from_secs(30),
        }
    }

    #[test]
    fn ahc_groups_close_embeddings_and_separates_far_ones() {
        let emb = [unit(0.0), unit(5.0), unit(90.0), unit(2.0), unit(95.0)];
        let refs: Vec<&[f32]> = emb.iter().map(Vec::as_slice).collect();
        let labels = average_linkage_labels(&refs, 0.5, far_budget().deadline).unwrap();
        assert_eq!(labels, vec![0, 0, 1, 0, 1]);
    }

    #[test]
    fn ahc_with_a_strict_threshold_keeps_every_entry_apart() {
        let emb = [unit(0.0), unit(30.0), unit(60.0)];
        let refs: Vec<&[f32]> = emb.iter().map(Vec::as_slice).collect();
        let labels = average_linkage_labels(&refs, 0.99, far_budget().deadline).unwrap();
        assert_eq!(labels, vec![0, 1, 2]);
    }

    #[test]
    fn expired_deadline_is_an_error_not_a_partial_result() {
        let entries: Vec<_> = (0..8).map(|i| entry(i, SpeakerId::new(), 0.0)).collect();
        let past = RefineBudget {
            deadline: Instant::now() - Duration::from_millis(1),
        };
        let r = AhcReclusterer {
            merge_threshold: 0.5,
        }
        .recluster(&entries, past);
        assert!(matches!(r, Err(DomainError::DiarizationFailed(_))));
    }

    #[test]
    fn strided_logs_assign_the_rest_to_the_nearest_centroid() {
        let (a, b) = (SpeakerId::new(), SpeakerId::new());
        let n = AHC_MAX_REPRESENTATIVES * 2 + 3;
        let entries: Vec<_> = (0..n as u32)
            .map(|i| {
                if i % 4 < 2 {
                    entry(i, a, 0.0)
                } else {
                    entry(i, b, 90.0)
                }
            })
            .collect();
        let labels = AhcReclusterer {
            merge_threshold: 0.5,
        }
        .recluster(&entries, far_budget())
        .unwrap();
        for (e, &l) in entries.iter().zip(&labels) {
            let expected = if e.online_speaker == a {
                labels[0]
            } else {
                labels[2]
            };
            assert_eq!(l, expected);
        }
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn matched_clusters_keep_their_online_ids() {
        let (a, b) = (Speaker::anonymous(0), Speaker::anonymous(1));
        let entries = vec![
            entry(0, a.id, 0.0),
            entry(1, b.id, 90.0),
            entry(2, b.id, 0.0), // mis-attributed online: belongs to a
            entry(3, b.id, 90.0),
        ];
        // Cluster 0 = a's voice, cluster 1 = b's voice.
        let RefineOutcome::Relabel(plan) =
            relabel_plan(&entries, &[a.clone(), b.clone()], &[1, 0, 1, 0])
        else {
            panic!("expected a relabel");
        };
        let ids: Vec<_> = plan.spans.iter().map(|(_, id)| *id).collect();
        assert_eq!(ids, vec![a.id, b.id, a.id, b.id]);
        assert!(plan.created.is_empty());
        assert!(plan.retired.is_empty());
    }

    #[test]
    fn merged_speaker_is_retired_to_its_majority_recipient() {
        let (a, b) = (Speaker::anonymous(0), Speaker::anonymous(1));
        let entries = vec![
            entry(0, a.id, 0.0),
            entry(1, b.id, 2.0),
            entry(2, a.id, 1.0),
        ];
        let RefineOutcome::Relabel(plan) =
            relabel_plan(&entries, &[a.clone(), b.clone()], &[0, 0, 0])
        else {
            panic!("expected a relabel");
        };
        assert!(plan.spans.iter().all(|(_, id)| *id == a.id));
        assert_eq!(
            plan.retired,
            vec![SpeakerRetirement {
                speaker: b.id,
                majority_recipient: Some(a.id),
            }]
        );
    }

    #[test]
    fn split_speaker_gets_a_fresh_id_at_the_next_slot() {
        let a = Speaker::anonymous(0);
        let b = Speaker::anonymous(3);
        let entries = vec![
            entry(0, a.id, 0.0),
            entry(1, a.id, 0.0),
            entry(2, a.id, 90.0),
            entry(3, b.id, 45.0),
        ];
        let RefineOutcome::Relabel(plan) =
            relabel_plan(&entries, &[a.clone(), b.clone()], &[0, 0, 1, 2])
        else {
            panic!("expected a relabel");
        };
        assert_eq!(plan.created.len(), 1);
        assert_eq!(plan.created[0].slot, 4);
        assert!(plan.created[0].label.is_none());
        assert_eq!(plan.spans[2].1, plan.created[0].id);
        assert_eq!(plan.spans[3].1, b.id);
    }

    #[test]
    fn identical_labelling_is_unchanged() {
        let (a, b) = (Speaker::anonymous(0), Speaker::anonymous(1));
        let entries = vec![entry(0, a.id, 0.0), entry(1, b.id, 90.0)];
        assert_eq!(
            relabel_plan(&entries, &[a, b], &[0, 1]),
            RefineOutcome::Unchanged
        );
    }

    #[test]
    fn short_sessions_are_too_short() {
        let a = SpeakerId::new();
        let three: Vec<_> = (0..3).map(|i| entry(i, a, 0.0)).collect();
        assert!(too_short(&three), "fewer than 4 entries");
        let mut four: Vec<_> = (0..4).map(|i| entry(i, a, 0.0)).collect();
        assert!(!too_short(&four), "4 x 3 s = 12 s of speech");
        for e in &mut four {
            e.end_ms = e.start_ms + 2_000;
        }
        assert!(too_short(&four), "4 x 2 s = 8 s < 10 s");
    }
}
