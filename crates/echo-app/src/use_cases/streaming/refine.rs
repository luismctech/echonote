//! End-of-session speaker refinement (ADR 0001, D3).
//!
//! While streaming, the [`AttributionLedger`] remembers how each emitted
//! segment was attributed: its chunk-local sample window and the
//! diarizer spans of its chunk. It holds ids and sample offsets only,
//! never embeddings. At stop, [`refine_at_stop`] runs the diarizer's
//! [`Diarizer::refine_plan`] off the async runtime under a deadline, and
//! [`AttributionLedger::reassign`] turns a span relabel plan into the
//! segment-level `SpeakersReassigned` payload by re-running the same
//! attribution rules (`dominant_span`, `chunk_dominant_speaker`).

use std::collections::HashMap;
use std::time::{Duration, Instant};

use tracing::{info, warn};

use echo_domain::{
    ChunkSpeaker, Diarizer, RefineBudget, RefineOutcome, SegmentId, SegmentSpeaker, SpanRef,
    SpanRelabelPlan, Speaker, SpeakerId, SpeakerRef, SpeakerRetirement, SpeakerSpan,
    StreamingSessionId,
};

use super::{chunk_dominant_speaker, dominant_span};

/// Hard limit on the end-of-session refine (ADR 0001, D8). `Stopped`
/// follows within this time even when the refine fails.
pub const DEFAULT_REFINE_TIMEOUT: Duration = Duration::from_secs(20);

/// Extra wait past the cooperative deadline before giving up on a
/// refine that ignores it. The blocking task is then left to finish on
/// its own; its result is discarded.
const JOIN_GRACE: Duration = Duration::from_secs(1);

/// One emitted segment, as attributed online.
#[derive(Debug, Clone, Copy)]
struct LedgerSegment {
    id: SegmentId,
    /// Chunk-local 16 kHz sample window.
    start: usize,
    end: usize,
    speaker: Option<SpeakerId>,
}

/// One emitted chunk that had diarizer spans.
#[derive(Debug, Clone)]
struct LedgerChunk {
    chunk_index: u32,
    spans: Vec<SpeakerSpan>,
    segments: Vec<LedgerSegment>,
    speaker: Option<SpeakerId>,
}

/// Vector-free record of every emitted chunk's spans and segments.
#[derive(Debug, Default)]
pub(super) struct AttributionLedger {
    chunks: Vec<LedgerChunk>,
}

/// Segment-level changes produced by a span relabel plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Reassignment {
    pub created: Vec<SpeakerRef>,
    pub retired: Vec<SpeakerRetirement>,
    pub segments: Vec<SegmentSpeaker>,
    pub chunks: Vec<ChunkSpeaker>,
}

impl AttributionLedger {
    /// Remember an emitted chunk. Chunks without spans are not
    /// recorded: nothing can move their segments.
    pub(super) fn record(
        &mut self,
        chunk_index: u32,
        spans: &[SpeakerSpan],
        segments: impl IntoIterator<Item = (SegmentId, usize, usize, Option<SpeakerId>)>,
        speaker: Option<SpeakerId>,
    ) {
        if spans.is_empty() {
            return;
        }
        self.chunks.push(LedgerChunk {
            chunk_index,
            spans: spans.to_vec(),
            segments: segments
                .into_iter()
                .map(|(id, start, end, speaker)| LedgerSegment {
                    id,
                    start,
                    end,
                    speaker,
                })
                .collect(),
            speaker,
        });
    }

    /// Apply `plan` to the recorded attribution. `online` is the
    /// diarizer's speaker list at stop (for slots).
    ///
    /// Returns `None` when no segment changes speaker, even if a chunk's
    /// dominant speaker changed: chunk rows only recolour live-pane
    /// lines, and without a moved segment there is nothing to persist
    /// or to report in the refine notice.
    ///
    /// The payload is self-consistent: `retired` is in speaker-id order,
    /// and every speaker a chunk row or a recipient names is either known
    /// online or listed in `created`. A created speaker that owns no
    /// segment is not sent, and no chunk row names it or a retired
    /// speaker: such a live-pane line takes the speaker owning most of
    /// its segment time (no speaker when it has no attributed segment).
    pub(super) fn reassign(
        &self,
        plan: &SpanRelabelPlan,
        online: &[Speaker],
    ) -> Option<Reassignment> {
        let relabel: HashMap<SpanRef, SpeakerId> = plan.spans.iter().copied().collect();
        let slot_of: HashMap<SpeakerId, u32> = online
            .iter()
            .chain(&plan.created)
            .map(|s| (s.id, s.slot))
            .collect();

        let mut segments = Vec::new();
        // Per chunk: (index, online speaker, span-dominant speaker and
        // slot, speaker owning most of its segment time after the plan).
        let mut chunk_views = Vec::with_capacity(self.chunks.len());
        // Per speaker: segment duration (samples) owned before and after.
        let mut before: HashMap<SpeakerId, usize> = HashMap::new();
        let mut after: HashMap<SpeakerId, usize> = HashMap::new();
        // Per old speaker: where its segments went, by duration.
        let mut flows: HashMap<SpeakerId, HashMap<SpeakerId, usize>> = HashMap::new();
        let mut known: Vec<SpeakerId> = Vec::new();

        for chunk in &self.chunks {
            let spans: Vec<SpeakerSpan> = chunk
                .spans
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    known.push(s.speaker_id);
                    let span = SpanRef {
                        chunk_index: chunk.chunk_index,
                        span_index: u16::try_from(i).unwrap_or(u16::MAX),
                    };
                    let id = relabel.get(&span).copied().unwrap_or(s.speaker_id);
                    SpeakerSpan {
                        speaker_id: id,
                        slot: slot_of.get(&id).copied().unwrap_or(s.slot),
                        ..*s
                    }
                })
                .collect();

            let mut owned: HashMap<SpeakerId, usize> = HashMap::new();
            for seg in &chunk.segments {
                let new = dominant_span(&spans, seg.start, seg.end).or(seg.speaker);
                let len = seg.end.saturating_sub(seg.start).max(1);
                if let Some(new) = new {
                    *owned.entry(new).or_default() += len;
                }
                if let Some(old) = seg.speaker {
                    *before.entry(old).or_default() += len;
                    if let Some(new) = new {
                        *flows.entry(old).or_default().entry(new).or_default() += len;
                    }
                }
                if let Some(new) = new {
                    *after.entry(new).or_default() += len;
                }
                if new != seg.speaker {
                    segments.push(SegmentSpeaker {
                        segment_id: seg.id,
                        speaker_id: new,
                    });
                }
            }

            let by_segments = owned
                .into_iter()
                .max_by_key(|(id, len)| (*len, std::cmp::Reverse(id.0)))
                .map(|(id, _)| id);
            chunk_views.push((
                chunk.chunk_index,
                chunk.speaker,
                chunk_dominant_speaker(&spans),
                by_segments,
            ));
        }

        if segments.is_empty() {
            return None;
        }

        // Retire every speaker left without segments: those the plan
        // retired and those that lost their last segment here. The
        // recipient takes the label rule's hand-over, so it must be a
        // speaker the store knows (seen in a span, or created below).
        let mut retired_ids: Vec<SpeakerId> = plan.retired.iter().map(|r| r.speaker).collect();
        for id in before.keys() {
            if !retired_ids.contains(id) {
                retired_ids.push(*id);
            }
        }
        retired_ids.sort_by_key(|id| id.0);
        let mut retired = Vec::new();
        for speaker in retired_ids {
            if after.contains_key(&speaker) {
                continue;
            }
            let by_segments = flows.get(&speaker).and_then(|to| {
                to.iter()
                    .max_by_key(|(id, len)| (**len, std::cmp::Reverse(id.0)))
                    .map(|(id, _)| *id)
            });
            let by_plan = plan
                .retired
                .iter()
                .find(|r| r.speaker == speaker)
                .and_then(|r| r.majority_recipient);
            retired.push(SpeakerRetirement {
                speaker,
                majority_recipient: by_segments.or(by_plan),
            });
        }

        let needed = |id: &SpeakerId| {
            after.contains_key(id) || retired.iter().any(|r| r.majority_recipient == Some(*id))
        };
        let created: Vec<SpeakerRef> = plan
            .created
            .iter()
            .filter(|s| needed(&s.id))
            .map(|s| SpeakerRef {
                id: s.id,
                slot: s.slot,
            })
            .collect();
        for r in &mut retired {
            let recipient_known = r
                .majority_recipient
                .is_some_and(|id| known.contains(&id) || created.iter().any(|c| c.id == id));
            if !recipient_known {
                r.majority_recipient = None;
            }
        }
        // A line's span-dominant speaker stands unless it is a created
        // speaker that is not sent or a retired one; the line then takes
        // the speaker owning most of its segment time, which owns a
        // segment and so is neither.
        let unusable = |id: SpeakerId| {
            retired.iter().any(|r| r.speaker == id)
                || (plan.created.iter().any(|c| c.id == id) && !created.iter().any(|c| c.id == id))
        };
        let mut chunks = Vec::new();
        for (chunk_index, online_speaker, dominant, by_segments) in chunk_views {
            let target = match dominant {
                Some((id, slot)) if !unusable(id) => Some((id, slot)),
                _ => by_segments.map(|id| (id, slot_of.get(&id).copied().unwrap_or_default())),
            };
            if target.map(|(id, _)| id) != online_speaker {
                chunks.push(ChunkSpeaker {
                    chunk_index,
                    speaker_id: target.map(|(id, _)| id),
                    speaker_slot: target.map(|(_, slot)| slot),
                });
            }
        }

        Some(Reassignment {
            created,
            retired,
            segments,
            chunks,
        })
    }
}

/// Run the diarizer's end-of-session refine on the blocking pool with a
/// `timeout` deadline. Returns the outcome and the diarizer's speakers,
/// or `None` on error, panic or timeout: the online result then stands.
/// Logs carry counts, timings and the error, never vectors.
pub(super) async fn refine_at_stop(
    diarizer: Box<dyn Diarizer>,
    timeout: Duration,
    session_id: StreamingSessionId,
) -> Option<(RefineOutcome, Vec<Speaker>)> {
    let started = Instant::now();
    let budget = RefineBudget {
        deadline: deadline_after(started, timeout),
    };
    let task = tokio::task::spawn_blocking(move || {
        let speakers = diarizer.speakers();
        diarizer
            .refine_plan(budget)
            .map(|outcome| (outcome, speakers))
    });
    let elapsed_ms = || started.elapsed().as_millis() as u64;
    match tokio::time::timeout(timeout.saturating_add(JOIN_GRACE), task).await {
        Ok(Ok(Ok(result))) => {
            info!(
                %session_id,
                elapsed_ms = elapsed_ms(),
                outcome = outcome_kind(&result.0),
                "offline diarization refine finished"
            );
            Some(result)
        }
        Ok(Ok(Err(e))) => {
            warn!(%session_id, elapsed_ms = elapsed_ms(), error = %e, "offline refine failed; keeping the online result");
            None
        }
        Ok(Err(join)) => {
            warn!(%session_id, elapsed_ms = elapsed_ms(), panicked = join.is_panic(), "offline refine task died; keeping the online result");
            None
        }
        Err(_) => {
            warn!(%session_id, elapsed_ms = elapsed_ms(), "offline refine ignored its deadline; keeping the online result");
            None
        }
    }
}

/// `start + timeout`. When that overflows (an effectively unbounded
/// timeout) the deadline is about 136 years away instead; if even that
/// overflows, `start` itself, so the refine fails fast and the online
/// result stands.
fn deadline_after(start: Instant, timeout: Duration) -> Instant {
    start
        .checked_add(timeout)
        .or_else(|| start.checked_add(Duration::from_secs(u64::from(u32::MAX))))
        .unwrap_or(start)
}

fn outcome_kind(outcome: &RefineOutcome) -> &'static str {
    match outcome {
        RefineOutcome::Unchanged => "unchanged",
        RefineOutcome::Merges(_) => "merges",
        RefineOutcome::Relabel(_) => "relabel",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn span(id: SpeakerId, slot: u32, start: usize, end: usize) -> SpeakerSpan {
        SpeakerSpan {
            speaker_id: id,
            slot,
            start_sample: start,
            end_sample: end,
        }
    }

    fn at(chunk_index: u32, span_index: u16) -> SpanRef {
        SpanRef {
            chunk_index,
            span_index,
        }
    }

    #[test]
    fn plan_moves_segments_through_the_same_attribution_rules() {
        let (a, b) = (Speaker::anonymous(0), Speaker::anonymous(1));
        let (s0, s1, s2) = (SegmentId::new(), SegmentId::new(), SegmentId::new());
        let mut ledger = AttributionLedger::default();
        // Chunk 0: a then b; segments follow their spans.
        ledger.record(
            0,
            &[span(a.id, 0, 0, 100), span(b.id, 1, 100, 300)],
            [(s0, 0, 100, Some(a.id)), (s1, 100, 300, Some(b.id))],
            Some(b.id),
        );
        // Chunk 1: b only.
        ledger.record(
            1,
            &[span(b.id, 1, 0, 200)],
            [(s2, 0, 200, Some(b.id))],
            Some(b.id),
        );

        // The refine decides chunk 0's second span was really a.
        let plan = SpanRelabelPlan {
            spans: vec![(at(0, 0), a.id), (at(0, 1), a.id), (at(1, 0), b.id)],
            created: vec![],
            retired: vec![],
        };
        let r = ledger.reassign(&plan, &[a.clone(), b.clone()]).unwrap();

        assert_eq!(
            r.segments,
            vec![SegmentSpeaker {
                segment_id: s1,
                speaker_id: Some(a.id)
            }]
        );
        assert_eq!(
            r.chunks,
            vec![ChunkSpeaker {
                chunk_index: 0,
                speaker_id: Some(a.id),
                speaker_slot: Some(0)
            }]
        );
        assert!(r.retired.is_empty(), "b still owns s2");
        assert!(r.created.is_empty());
    }

    #[test]
    fn speaker_losing_every_segment_is_retired_to_where_they_went() {
        let (a, b) = (Speaker::anonymous(0), Speaker::anonymous(1));
        let fresh = Speaker::anonymous(2);
        let (s0, s1) = (SegmentId::new(), SegmentId::new());
        let mut ledger = AttributionLedger::default();
        ledger.record(
            0,
            &[span(a.id, 0, 0, 100)],
            [(s0, 0, 100, Some(a.id))],
            Some(a.id),
        );
        ledger.record(
            1,
            &[span(b.id, 1, 0, 100)],
            [(s1, 0, 100, Some(b.id))],
            Some(b.id),
        );

        let plan = SpanRelabelPlan {
            spans: vec![(at(0, 0), a.id), (at(1, 0), fresh.id)],
            created: vec![fresh.clone()],
            retired: vec![SpeakerRetirement {
                speaker: b.id,
                majority_recipient: Some(fresh.id),
            }],
        };
        let r = ledger.reassign(&plan, &[a, b.clone()]).unwrap();
        assert_eq!(
            r.created,
            vec![SpeakerRef {
                id: fresh.id,
                slot: 2
            }]
        );
        assert_eq!(
            r.retired,
            vec![SpeakerRetirement {
                speaker: b.id,
                majority_recipient: Some(fresh.id)
            }]
        );
        assert_eq!(r.chunks[0].speaker_slot, Some(2));
    }

    #[test]
    fn plan_that_moves_no_segment_yields_nothing() {
        let a = Speaker::anonymous(0);
        let mut ledger = AttributionLedger::default();
        ledger.record(
            0,
            &[span(a.id, 0, 0, 100)],
            [(SegmentId::new(), 0, 100, Some(a.id))],
            Some(a.id),
        );
        let plan = SpanRelabelPlan {
            spans: vec![(at(0, 0), a.id)],
            ..SpanRelabelPlan::default()
        };
        assert_eq!(ledger.reassign(&plan, &[a]), None);
    }

    #[test]
    fn retirements_come_in_speaker_id_order() {
        let speakers: Vec<Speaker> = (0..8).map(Speaker::anonymous).collect();
        let keeper = Speaker::anonymous(8);
        let mut ledger = AttributionLedger::default();
        for (i, s) in speakers.iter().enumerate() {
            ledger.record(
                i as u32,
                &[span(s.id, s.slot, 0, 100)],
                [(SegmentId::new(), 0, 100, Some(s.id))],
                Some(s.id),
            );
        }
        let plan = SpanRelabelPlan {
            spans: (0..8).map(|i| (at(i, 0), keeper.id)).collect(),
            ..SpanRelabelPlan::default()
        };
        let mut online = speakers.clone();
        online.push(keeper);
        let r = ledger.reassign(&plan, &online).unwrap();
        let got: Vec<SpeakerId> = r.retired.iter().map(|x| x.speaker).collect();
        let mut sorted = got.clone();
        sorted.sort_by_key(|id| id.0);
        assert_eq!(got.len(), 8);
        assert_eq!(got, sorted);
    }

    #[test]
    fn chunk_rows_never_name_a_speaker_the_store_is_not_told_to_create() {
        let (a, b) = (Speaker::anonymous(0), Speaker::anonymous(1));
        let fresh = Speaker::anonymous(2);
        let s0 = SegmentId::new();
        let mut ledger = AttributionLedger::default();
        // Chunk 0: a short span with a segment, then a long span with
        // none (ASR found no words in it).
        ledger.record(
            0,
            &[span(a.id, 0, 0, 100), span(a.id, 0, 100, 400)],
            [(s0, 0, 100, Some(a.id))],
            Some(a.id),
        );
        ledger.record(
            1,
            &[span(b.id, 1, 0, 100)],
            [(SegmentId::new(), 0, 100, Some(b.id))],
            Some(b.id),
        );
        // The segment moves to b; the long span becomes a new speaker
        // that owns no segment.
        let plan = SpanRelabelPlan {
            spans: vec![(at(0, 0), b.id), (at(0, 1), fresh.id), (at(1, 0), b.id)],
            created: vec![fresh.clone()],
            retired: vec![],
        };
        let r = ledger.reassign(&plan, &[a, b]).unwrap();
        assert!(r.created.is_empty(), "fresh owns no segment");
        assert!(
            r.chunks.iter().all(|c| c.speaker_id != Some(fresh.id)),
            "{:?}",
            r.chunks
        );
    }

    #[test]
    fn live_pane_lines_never_keep_a_retired_speaker() {
        let (a, b) = (Speaker::anonymous(0), Speaker::anonymous(1));
        let fresh = Speaker::anonymous(2);
        let mut ledger = AttributionLedger::default();
        // Chunk 0 is a's only chunk: a short span with a segment, then a
        // long span with none.
        ledger.record(
            0,
            &[span(a.id, 0, 0, 100), span(a.id, 0, 100, 400)],
            [(SegmentId::new(), 0, 100, Some(a.id))],
            Some(a.id),
        );
        ledger.record(
            1,
            &[span(b.id, 1, 0, 100)],
            [(SegmentId::new(), 0, 100, Some(b.id))],
            Some(b.id),
        );
        // a's segment moves to b and a is retired; the long span's new
        // speaker owns no segment and is not sent, so chunk 0's line
        // takes the speaker that owns its segments instead of keeping a.
        let plan = SpanRelabelPlan {
            spans: vec![(at(0, 0), b.id), (at(0, 1), fresh.id), (at(1, 0), b.id)],
            created: vec![fresh.clone()],
            retired: vec![],
        };
        let r = ledger.reassign(&plan, &[a.clone(), b.clone()]).unwrap();
        assert_eq!(r.retired.len(), 1);
        assert_eq!(r.retired[0].speaker, a.id);
        assert_eq!(
            r.chunks,
            vec![ChunkSpeaker {
                chunk_index: 0,
                speaker_id: Some(b.id),
                speaker_slot: Some(1)
            }]
        );
    }

    #[test]
    fn chunk_only_changes_are_discarded() {
        let a = Speaker::anonymous(0);
        let b = Speaker::anonymous(1);
        let mut ledger = AttributionLedger::default();
        // The segment sits on the first span; only the second, longer
        // span (and so the chunk's dominant speaker) changes.
        ledger.record(
            0,
            &[span(a.id, 0, 0, 100), span(a.id, 0, 100, 400)],
            [(SegmentId::new(), 0, 100, Some(a.id))],
            Some(a.id),
        );
        let plan = SpanRelabelPlan {
            spans: vec![(at(0, 0), a.id), (at(0, 1), b.id)],
            ..SpanRelabelPlan::default()
        };
        assert_eq!(ledger.reassign(&plan, &[a, b]), None);
    }

    #[test]
    fn chunks_without_spans_are_not_recorded() {
        let mut ledger = AttributionLedger::default();
        ledger.record(0, &[], [(SegmentId::new(), 0, 100, None)], None);
        assert!(ledger.chunks.is_empty());
    }
}
