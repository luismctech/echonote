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

use std::collections::HashMap;

use echo_domain::{SpanRef, SpeakerId};

use crate::embedding::l2_normalize;

/// Hard cap on the number of entries (ADR 0001, D1: about 14 MB at 192
/// dims). The log never holds more; reaching it compacts the log.
pub const EMBEDDING_LOG_CAP: usize = 16_384;

/// Hard cap on the extra window embeddings across all entries (ADR 0001,
/// Q3: about 13 MB at 192 dims, so the whole log stays near 35 MB at the
/// worst case of 512 dims for the entries alone). Reaching it thins the
/// windows; entries are never dropped for windows.
pub const EMBEDDING_LOG_WINDOW_CAP: usize = 16_384;

/// One embedded span (or, after compaction, several spans of the same
/// online speaker and epoch).
#[derive(Clone, PartialEq)]
pub struct EmbeddingLogEntry {
    /// Spans this entry covers, in time order. More than one only after
    /// compaction.
    pub spans: Vec<SpanRef>,
    /// Session time of the first sample, in milliseconds.
    pub start_ms: u32,
    /// Session time just past the last sample, in milliseconds.
    pub end_ms: u32,
    /// Speech the entry's spans hold, in milliseconds. Equal to
    /// `end_ms - start_ms` for one span; after coalescing entries that
    /// were not adjacent, the time range also covers other speakers'
    /// turns, so this sum is what weights the entry. Private so it only
    /// comes from [`Self::new`] (the range length) or coalescing (the
    /// sum); read it with [`Self::duration_ms`].
    speech_ms: u32,
    /// Capture run: 0 at start, +1 after every resume.
    pub epoch: u16,
    /// Window-local segmenter speaker hint; `None` on the whole-chunk
    /// path. Never an identity across windows.
    pub local_speaker: Option<u8>,
    /// Speaker the online clusterer assigned at the time.
    pub online_speaker: SpeakerId,
    /// The embedding as produced by the embedder.
    pub embedding: Box<[f32]>,
    /// Extra embeddings of fixed-length windows inside a long span, in
    /// time order (ADR 0001, Q3). Only the re-segmentation reads them;
    /// empty for short spans and when dense windows are off.
    pub windows: Vec<Box<[f32]>>,
}

impl EmbeddingLogEntry {
    /// Entry for spans covering the session-time range `time` (ms); its
    /// speech duration is the length of that range.
    #[must_use]
    pub fn new(
        spans: Vec<SpanRef>,
        time: std::ops::Range<u32>,
        epoch: u16,
        local_speaker: Option<u8>,
        online_speaker: SpeakerId,
        embedding: Box<[f32]>,
    ) -> Self {
        Self {
            spans,
            start_ms: time.start,
            end_ms: time.end,
            speech_ms: time.end.saturating_sub(time.start),
            epoch,
            local_speaker,
            online_speaker,
            embedding,
            windows: Vec::new(),
        }
    }

    /// The same entry with these window embeddings.
    #[must_use]
    pub fn with_windows(mut self, windows: Vec<Box<[f32]>>) -> Self {
        self.windows = windows;
        self
    }

    /// Logged speech duration in milliseconds (the summed speech of the
    /// entry's spans), not the time range: the weight of this entry in every duration-based
    /// decision of the refine.
    #[must_use]
    pub fn duration_ms(&self) -> u32 {
        self.speech_ms
    }
}

impl std::fmt::Debug for EmbeddingLogEntry {
    // Manual impl: the vector must never reach a log line.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingLogEntry")
            .field("spans", &self.spans)
            .field("start_ms", &self.start_ms)
            .field("end_ms", &self.end_ms)
            .field("speech_ms", &self.speech_ms)
            .field("epoch", &self.epoch)
            .field("local_speaker", &self.local_speaker)
            .field("online_speaker", &self.online_speaker)
            .field("embedding_dim", &self.embedding.len())
            .field("windows", &self.windows.len())
            .finish()
    }
}

/// Bounded, append-only (until compaction) embedding log.
#[derive(Debug)]
pub struct EmbeddingLog {
    entries: Vec<EmbeddingLogEntry>,
    cap: usize,
    windows: usize,
    window_cap: usize,
}

impl Default for EmbeddingLog {
    fn default() -> Self {
        Self::with_caps(EMBEDDING_LOG_CAP, EMBEDDING_LOG_WINDOW_CAP)
    }
}

impl EmbeddingLog {
    /// Empty log that never holds more than `cap` entries (`cap >= 2`)
    /// or more than [`EMBEDDING_LOG_WINDOW_CAP`] window embeddings.
    #[must_use]
    pub fn with_cap(cap: usize) -> Self {
        Self::with_caps(cap, EMBEDDING_LOG_WINDOW_CAP)
    }

    /// Empty log that never holds more than `cap` entries (`cap >= 2`)
    /// or more than `window_cap` window embeddings.
    #[must_use]
    pub fn with_caps(cap: usize, window_cap: usize) -> Self {
        Self {
            entries: Vec::new(),
            cap: cap.max(2),
            windows: 0,
            window_cap,
        }
    }

    /// Window embeddings held across all entries.
    #[must_use]
    pub fn window_count(&self) -> usize {
        self.windows
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
        self.windows = 0;
    }

    /// Append an entry, compacting first when the log is full and
    /// thinning the windows when they exceed their cap.
    pub fn push(&mut self, mut entry: EmbeddingLogEntry) {
        if self.entries.len() >= self.cap {
            self.compact();
            self.windows = self.entries.iter().map(|e| e.windows.len()).sum();
        }
        entry.windows.truncate(self.window_cap);
        self.windows += entry.windows.len();
        self.entries.push(entry);
        if self.windows > self.window_cap {
            self.thin_windows();
        }
    }

    /// Bring the window count back under its cap, O(n) per pass: first
    /// keep every other window of each entry that has several (the first
    /// one stays), so the remaining windows still cover each long span;
    /// once every entry has at most one, drop windows oldest first.
    fn thin_windows(&mut self) {
        let before = self.windows;
        while self.windows > self.window_cap && self.entries.iter().any(|e| e.windows.len() > 1) {
            for e in &mut self.entries {
                if e.windows.len() > 1 {
                    let mut i = 0;
                    e.windows.retain(|_| {
                        i += 1;
                        i % 2 == 1
                    });
                }
            }
            self.windows = self.entries.iter().map(|e| e.windows.len()).sum();
        }
        for e in &mut self.entries {
            if self.windows <= self.window_cap {
                break;
            }
            self.windows -= e.windows.len();
            e.windows.clear();
        }
        tracing::debug!(
            before,
            after = self.windows,
            "embedding log windows thinned"
        );
    }

    /// Make room for at least one entry. Runs on the streaming assign
    /// path, so every pass is O(n):
    ///
    /// 1. Coalesce each pair of time-adjacent entries with the same
    ///    online speaker and epoch.
    /// 2. If the log is still above three quarters of the cap (speakers
    ///    alternate), pair each remaining entry with the next unpaired
    ///    entry of the same speaker and epoch, in one pass.
    /// 3. If nothing could be coalesced (every entry has its own speaker
    ///    and epoch), drop the oldest entry. Its spans keep their online
    ///    speaker.
    ///
    /// Passes 1 and 2 usually shrink the log by a large fraction, so the
    /// compaction cost is amortised over many pushes. A coalesced entry
    /// has the speech-weighted mean embedding (renormalised), the union
    /// of the time ranges, the summed speech duration and the
    /// concatenated spans.
    fn compact(&mut self) {
        let before = self.entries.len();
        self.entries = coalesce_adjacent(std::mem::take(&mut self.entries));
        if self.entries.len() > self.cap / 4 * 3 {
            self.entries = coalesce_same_run(std::mem::take(&mut self.entries));
        }
        let mut dropped = 0_usize;
        if self.entries.len() >= self.cap {
            dropped = self.entries.len() + 1 - self.cap;
            self.entries.drain(..dropped);
        }
        if dropped > 0 {
            // Counts only: the dropped spans keep their online speaker,
            // so the refine silently covers less of the session.
            tracing::warn!(
                before,
                after = self.entries.len(),
                dropped,
                "embedding log full after compaction; oldest entries dropped"
            );
        } else {
            tracing::debug!(
                before,
                after = self.entries.len(),
                "embedding log compacted"
            );
        }
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

/// Pair every entry with the next unpaired entry of the same speaker and
/// epoch. One pass with a map from run to the output position of its
/// pending (unpaired) entry; a coalesced entry stays at the earlier
/// entry's position, so the output keeps the order of the first spans.
fn coalesce_same_run(entries: Vec<EmbeddingLogEntry>) -> Vec<EmbeddingLogEntry> {
    let mut out: Vec<EmbeddingLogEntry> = Vec::with_capacity(entries.len());
    let mut pending: HashMap<(SpeakerId, u16), usize> = HashMap::new();
    for entry in entries {
        let key = (entry.online_speaker, entry.epoch);
        match pending.remove(&key) {
            Some(at) => absorb(&mut out[at], entry),
            None => {
                pending.insert(key, out.len());
                out.push(entry);
            }
        }
    }
    out
}

fn coalesce(mut a: EmbeddingLogEntry, b: EmbeddingLogEntry) -> EmbeddingLogEntry {
    absorb(&mut a, b);
    a
}

/// Fold `b` into `a` (same speaker and epoch, `b` later in time).
fn absorb(a: &mut EmbeddingLogEntry, b: EmbeddingLogEntry) {
    let (wa, wb) = (a.speech_ms.max(1) as f32, b.speech_ms.max(1) as f32);
    let mut embedding: Vec<f32> = a
        .embedding
        .iter()
        .zip(b.embedding.iter())
        .map(|(x, y)| (x * wa + y * wb) / (wa + wb))
        .collect();
    l2_normalize(&mut embedding);
    a.embedding = embedding.into_boxed_slice();
    a.spans.extend(b.spans);
    a.windows.extend(b.windows);
    a.start_ms = a.start_ms.min(b.start_ms);
    a.end_ms = a.end_ms.max(b.end_ms);
    a.speech_ms = a.speech_ms.saturating_add(b.speech_ms);
    if a.local_speaker != b.local_speaker {
        a.local_speaker = None;
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
        EmbeddingLogEntry::new(
            vec![SpanRef {
                chunk_index: chunk,
                span_index: 0,
            }],
            start_ms..end_ms,
            0,
            None,
            speaker,
            Box::new(v),
        )
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
    fn entries_of_different_epochs_are_never_coalesced_and_the_cap_holds() {
        let a = SpeakerId::new();
        let mut log = EmbeddingLog::with_cap(2);
        for epoch in 0..3_u16 {
            let t = u32::from(epoch) * 1_000;
            let mut e = entry(u32::from(epoch), t, t + 1_000, a, [1.0, 0.0]);
            e.epoch = epoch;
            log.push(e);
        }
        // Nothing can be coalesced: the oldest entry makes room.
        let epochs: Vec<_> = log.entries().iter().map(|e| e.epoch).collect();
        assert_eq!(epochs, vec![1, 2]);
    }

    #[test]
    fn the_log_never_exceeds_its_cap() {
        let speakers: Vec<_> = (0..5).map(|_| SpeakerId::new()).collect();
        let mut log = EmbeddingLog::with_cap(8);
        for i in 0..200_u32 {
            let t = i * 1_000;
            log.push(entry(i, t, t + 1_000, speakers[i as usize % 5], [1.0, 0.0]));
            assert!(log.len() <= 8, "len {} after push {i}", log.len());
        }
        let spans: usize = log.entries().iter().map(|e| e.spans.len()).sum();
        assert!(spans > 8, "compaction coalesced instead of dropping");
    }

    #[test]
    fn coalescing_non_adjacent_entries_keeps_the_speech_duration() {
        let (a, b) = (SpeakerId::new(), SpeakerId::new());
        let mut log = EmbeddingLog::with_cap(4);
        for (i, s) in [a, b, a, b].into_iter().enumerate() {
            let t = i as u32 * 1_000;
            log.push(entry(i as u32, t, t + 1_000, s, [1.0, 0.0]));
        }
        log.push(entry(4, 4_000, 5_000, a, [1.0, 0.0]));
        let first = &log.entries()[0];
        assert_eq!((first.start_ms, first.end_ms), (0, 3_000), "time span");
        assert_eq!(first.duration_ms(), 2_000, "speech excludes b's turn");
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

    fn windowed(chunk: u32, speaker: SpeakerId, n: usize) -> EmbeddingLogEntry {
        let t = chunk * 6_000;
        entry(chunk, t, t + 6_000, speaker, [1.0, 0.0]).with_windows(
            (0..n)
                .map(|i| vec![i as f32, 1.0].into_boxed_slice())
                .collect(),
        )
    }

    #[test]
    fn debug_output_never_prints_the_window_vectors() {
        let e = windowed(0, SpeakerId::new(), 2);
        let text = format!("{e:?}");
        assert!(text.contains("windows: 2"), "{text}");
    }

    #[test]
    fn windows_beyond_their_cap_are_thinned_and_the_cap_holds() {
        let a = SpeakerId::new();
        let mut log = EmbeddingLog::with_caps(100, 8);
        for i in 0..5 {
            log.push(windowed(i, a, 4));
            assert!(log.window_count() <= 8, "{} windows", log.window_count());
        }
        assert_eq!(log.len(), 5, "thinning windows never drops entries");
        let counted: usize = log.entries().iter().map(|e| e.windows.len()).sum();
        assert_eq!(counted, log.window_count());
        assert!(
            log.entries()
                .iter()
                .all(|e| e.windows.first().is_some_and(|w| w[0] == 0.0)),
            "every entry keeps its first window"
        );
    }

    #[test]
    fn single_windows_beyond_the_cap_go_oldest_first() {
        let a = SpeakerId::new();
        let mut log = EmbeddingLog::with_caps(100, 2);
        for i in 0..4 {
            log.push(windowed(i, a, 1));
        }
        assert!(log.window_count() <= 2);
        assert!(
            log.entries()[3].windows.len() == 1,
            "newest keeps its window"
        );
        assert!(
            log.entries()[0].windows.is_empty(),
            "oldest lost its window"
        );
    }

    #[test]
    fn coalescing_keeps_both_window_lists_and_the_count() {
        let a = SpeakerId::new();
        let mut log = EmbeddingLog::with_caps(2, 100);
        log.push(windowed(0, a, 2));
        log.push(windowed(1, a, 3));
        log.push(windowed(2, a, 1));
        assert_eq!(log.entries()[0].windows.len(), 5);
        assert_eq!(log.window_count(), 6);
    }

    #[test]
    fn clear_empties_the_log() {
        let mut log = EmbeddingLog::default();
        log.push(entry(0, 0, 1_000, SpeakerId::new(), [1.0, 0.0]));
        log.clear();
        assert!(log.is_empty());
    }
}
