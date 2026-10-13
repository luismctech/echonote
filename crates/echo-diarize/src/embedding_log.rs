//! In-memory, timestamped log of the embeddings computed in one session.
//!
//! [`OnlineDiarizer`](crate::OnlineDiarizer) appends one
//! [`EmbeddingLogEntry`] per embedded span so the end-of-session refine
//! can re-cluster the whole sequence (ADR 0001, D1).
//!
//! Privacy: speaker embeddings can re-identify a voice. The log lives in
//! RAM for one session only. It is never serialised, persisted, put in an
//! event or logged: the entry type has no `Serialize` impl and its
//! `Debug` impl prints the vector's dimension, never its values.

use echo_domain::{SpanRef, SpeakerId};

use crate::embedding::l2_normalize;

/// Hard cap on the number of entries (ADR 0001, D1: about 14 MB at 192
/// dims). Reaching it compacts the log instead of dropping entries.
pub const EMBEDDING_LOG_CAP: usize = 16_384;

/// One embedded span (or, after compaction, several adjacent spans of
/// the same online speaker).
#[derive(Clone, PartialEq)]
pub struct EmbeddingLogEntry {
    /// Spans this entry covers, in time order. More than one only after
    /// compaction.
    pub spans: Vec<SpanRef>,
    /// Session time of the first sample, in milliseconds.
    pub start_ms: u32,
    /// Session time just past the last sample, in milliseconds.
    pub end_ms: u32,
    /// Capture run: 0 at start, +1 after every resume.
    pub epoch: u16,
    /// Window-local segmenter speaker hint; `None` on the whole-chunk
    /// path. Never an identity across windows.
    pub local_speaker: Option<u8>,
    /// Speaker the online clusterer assigned at the time.
    pub online_speaker: SpeakerId,
    /// The embedding as produced by the embedder.
    pub embedding: Box<[f32]>,
}

impl EmbeddingLogEntry {
    /// Logged speech duration in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u32 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

impl std::fmt::Debug for EmbeddingLogEntry {
    // Manual impl: the vector must never reach a log line.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingLogEntry")
            .field("spans", &self.spans)
            .field("start_ms", &self.start_ms)
            .field("end_ms", &self.end_ms)
            .field("epoch", &self.epoch)
            .field("local_speaker", &self.local_speaker)
            .field("online_speaker", &self.online_speaker)
            .field("embedding_dim", &self.embedding.len())
            .finish()
    }
}

/// Bounded, append-only (until compaction) embedding log.
#[derive(Debug)]
pub struct EmbeddingLog {
    entries: Vec<EmbeddingLogEntry>,
    cap: usize,
}

impl Default for EmbeddingLog {
    fn default() -> Self {
        Self::with_cap(EMBEDDING_LOG_CAP)
    }
}

impl EmbeddingLog {
    /// Empty log that compacts when it holds `cap` entries (`cap >= 2`).
    #[must_use]
    pub fn with_cap(cap: usize) -> Self {
        Self {
            entries: Vec::new(),
            cap: cap.max(2),
        }
    }

    /// Entries in insertion (time) order.
    #[must_use]
    pub fn entries(&self) -> &[EmbeddingLogEntry] {
        &self.entries
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` when nothing was logged.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Drop every entry (session reset).
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Append an entry, compacting first when the log is full.
    pub fn push(&mut self, entry: EmbeddingLogEntry) {
        if self.entries.len() >= self.cap {
            self.compact();
        }
        self.entries.push(entry);
    }

    /// Coalesce entries so the log has room again; no span is dropped.
    ///
    /// First pass: each pair of time-adjacent entries with the same
    /// online speaker and epoch becomes one entry. When no such pair
    /// exists (speakers alternate on every span), each entry is paired
    /// with the next entry of the same speaker and epoch instead, so the
    /// log still shrinks. A coalesced entry has the duration-weighted
    /// mean embedding (renormalised), the union of the time ranges and
    /// the concatenated spans.
    fn compact(&mut self) {
        let before = self.entries.len();
        self.entries = coalesce_adjacent(std::mem::take(&mut self.entries));
        if self.entries.len() == before {
            self.entries = coalesce_same_speaker(std::mem::take(&mut self.entries));
        }
        tracing::debug!(
            before,
            after = self.entries.len(),
            "embedding log compacted"
        );
    }
}

fn same_run(a: &EmbeddingLogEntry, b: &EmbeddingLogEntry) -> bool {
    a.online_speaker == b.online_speaker && a.epoch == b.epoch
}

fn coalesce_adjacent(entries: Vec<EmbeddingLogEntry>) -> Vec<EmbeddingLogEntry> {
    let mut out = Vec::with_capacity(entries.len());
    let mut iter = entries.into_iter().peekable();
    while let Some(a) = iter.next() {
        match iter.next_if(|b| same_run(&a, b)) {
            Some(b) => out.push(coalesce(a, b)),
            None => out.push(a),
        }
    }
    out
}

fn coalesce_same_speaker(entries: Vec<EmbeddingLogEntry>) -> Vec<EmbeddingLogEntry> {
    let mut slots: Vec<Option<EmbeddingLogEntry>> = entries.into_iter().map(Some).collect();
    let mut out = Vec::with_capacity(slots.len());
    for i in 0..slots.len() {
        let Some(a) = slots[i].take() else { continue };
        let partner =
            (i + 1..slots.len()).find(|&j| slots[j].as_ref().is_some_and(|b| same_run(&a, b)));
        match partner.and_then(|j| slots[j].take()) {
            Some(b) => out.push(coalesce(a, b)),
            None => out.push(a),
        }
    }
    out
}

fn coalesce(a: EmbeddingLogEntry, b: EmbeddingLogEntry) -> EmbeddingLogEntry {
    let (wa, wb) = (a.duration_ms().max(1) as f32, b.duration_ms().max(1) as f32);
    let mut embedding: Vec<f32> = a
        .embedding
        .iter()
        .zip(b.embedding.iter())
        .map(|(x, y)| (x * wa + y * wb) / (wa + wb))
        .collect();
    l2_normalize(&mut embedding);
    let mut spans = a.spans;
    spans.extend(b.spans);
    EmbeddingLogEntry {
        spans,
        start_ms: a.start_ms.min(b.start_ms),
        end_ms: a.end_ms.max(b.end_ms),
        epoch: a.epoch,
        local_speaker: if a.local_speaker == b.local_speaker {
            a.local_speaker
        } else {
            None
        },
        online_speaker: a.online_speaker,
        embedding: embedding.into_boxed_slice(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn entry(
        chunk: u32,
        start_ms: u32,
        end_ms: u32,
        speaker: SpeakerId,
        v: [f32; 2],
    ) -> EmbeddingLogEntry {
        EmbeddingLogEntry {
            spans: vec![SpanRef {
                chunk_index: chunk,
                span_index: 0,
            }],
            start_ms,
            end_ms,
            epoch: 0,
            local_speaker: None,
            online_speaker: speaker,
            embedding: Box::new(v),
        }
    }

    fn chunks(log: &EmbeddingLog) -> Vec<Vec<u32>> {
        log.entries()
            .iter()
            .map(|e| e.spans.iter().map(|s| s.chunk_index).collect())
            .collect()
    }

    #[test]
    fn debug_output_never_prints_the_vector() {
        let e = entry(0, 0, 1_000, SpeakerId::new(), [0.123_456, 0.654_321]);
        let text = format!("{e:?}");
        assert!(text.contains("embedding_dim: 2"), "{text}");
        assert!(!text.contains("0.123"), "vector leaked: {text}");
    }

    #[test]
    fn full_log_coalesces_adjacent_same_speaker_entries_without_dropping_spans() {
        let (a, b) = (SpeakerId::new(), SpeakerId::new());
        let mut log = EmbeddingLog::with_cap(4);
        log.push(entry(0, 0, 1_000, a, [1.0, 0.0]));
        log.push(entry(1, 1_000, 4_000, a, [0.0, 1.0]));
        log.push(entry(2, 4_000, 5_000, b, [1.0, 0.0]));
        log.push(entry(3, 5_000, 6_000, b, [1.0, 0.0]));
        log.push(entry(4, 6_000, 7_000, a, [1.0, 0.0]));

        assert_eq!(chunks(&log), vec![vec![0, 1], vec![2, 3], vec![4]]);
        let first = &log.entries()[0];
        assert_eq!((first.start_ms, first.end_ms), (0, 4_000));
        assert_eq!(first.online_speaker, a);
        // Duration-weighted (1:3) mean, renormalised.
        let norm = (1.0_f32 + 9.0).sqrt();
        assert!((first.embedding[0] - 1.0 / norm).abs() < 1e-6);
        assert!((first.embedding[1] - 3.0 / norm).abs() < 1e-6);
    }

    #[test]
    fn entries_of_different_epochs_are_never_coalesced() {
        let a = SpeakerId::new();
        let mut log = EmbeddingLog::with_cap(2);
        log.push(entry(0, 0, 1_000, a, [1.0, 0.0]));
        let mut resumed = entry(1, 1_000, 2_000, a, [1.0, 0.0]);
        resumed.epoch = 1;
        log.push(resumed);
        log.push(entry(2, 2_000, 3_000, a, [1.0, 0.0]));
        assert_eq!(log.len(), 3, "different epochs stay apart");
    }

    #[test]
    fn alternating_speakers_still_shrink_the_log() {
        let (a, b) = (SpeakerId::new(), SpeakerId::new());
        let mut log = EmbeddingLog::with_cap(4);
        for (i, s) in [a, b, a, b].into_iter().enumerate() {
            let t = i as u32 * 1_000;
            log.push(entry(i as u32, t, t + 1_000, s, [1.0, 0.0]));
        }
        log.push(entry(4, 4_000, 5_000, a, [1.0, 0.0]));
        assert_eq!(chunks(&log), vec![vec![0, 2], vec![1, 3], vec![4]]);
    }

    #[test]
    fn clear_empties_the_log() {
        let mut log = EmbeddingLog::default();
        log.push(entry(0, 0, 1_000, SpeakerId::new(), [1.0, 0.0]));
        log.clear();
        assert!(log.is_empty());
    }
}
