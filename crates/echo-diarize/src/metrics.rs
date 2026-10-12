//! Diarization error rate (DER) scoring over RTTM speaker segments.
//!
//! Pure, model-free helpers used by the `echo-proto bench der` harness:
//!
//! - [`parse_rttm`] / [`write_rttm`] read and write the NIST RTTM
//!   `SPEAKER` lines (`SPEAKER <file> 1 <start> <dur> <NA> <NA> <spk> <NA> <NA>`).
//! - [`score_der`] computes DER = (missed + false alarm + speaker
//!   confusion) / scored reference speech, md-eval style: the timeline
//!   is cut into elementary intervals at every segment boundary, each
//!   interval is scored with the counts of active reference and
//!   hypothesis speakers, and hypothesis labels are mapped one-to-one
//!   onto reference labels so that the mapped co-speaking time is
//!   maximal (Hungarian assignment). Label names never matter.
//!
//! The collar follows md-eval semantics: `collar` seconds on *each* side
//! of every reference segment boundary are excluded from scoring.
//! `skip_overlap` additionally excludes regions where the reference has
//! two or more simultaneous speakers.

use std::ops::AddAssign;

/// One `SPEAKER` line of an RTTM file.
#[derive(Debug, Clone, PartialEq)]
pub struct RttmSegment {
    /// Recording identifier (second RTTM field).
    pub file_id: String,
    /// Speaker label (eighth RTTM field).
    pub speaker: String,
    /// Onset in seconds.
    pub start: f64,
    /// Duration in seconds.
    pub duration: f64,
}

impl RttmSegment {
    /// Offset (end time) in seconds.
    #[must_use]
    pub fn end(&self) -> f64 {
        self.start + self.duration
    }
}

/// Malformed RTTM input.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("RTTM line {line}: {reason}")]
pub struct RttmError {
    /// 1-based line number of the offending line.
    pub line: usize,
    /// What is wrong with it.
    pub reason: String,
}

/// Parse every `SPEAKER` line of an RTTM document.
///
/// Blank lines, `;;` comments and non-`SPEAKER` records are ignored.
///
/// # Errors
///
/// Returns [`RttmError`] when a `SPEAKER` line has fewer than eight
/// fields, a non-numeric or negative onset/duration.
pub fn parse_rttm(text: &str) -> Result<Vec<RttmSegment>, RttmError> {
    let mut segments = Vec::new();
    for (idx, raw) in text.lines().enumerate() {
        let line = idx + 1;
        let fields: Vec<&str> = raw.split_whitespace().collect();
        if fields.first() != Some(&"SPEAKER") {
            continue;
        }
        let err = |reason: &str| RttmError {
            line,
            reason: reason.to_string(),
        };
        if fields.len() < 8 {
            return Err(err("expected at least 8 fields"));
        }
        let number = |s: &str, what: &str| -> Result<f64, RttmError> {
            let v: f64 = s
                .parse()
                .map_err(|_| err(&format!("{what} `{s}` is not a number")))?;
            if v.is_finite() && v >= 0.0 {
                Ok(v)
            } else {
                Err(err(&format!("{what} `{s}` must be finite and >= 0")))
            }
        };
        segments.push(RttmSegment {
            file_id: fields[1].to_string(),
            speaker: fields[7].to_string(),
            start: number(fields[3], "onset")?,
            duration: number(fields[4], "duration")?,
        });
    }
    Ok(segments)
}

/// Render segments as RTTM `SPEAKER` lines (millisecond precision).
#[must_use]
pub fn write_rttm(segments: &[RttmSegment]) -> String {
    segments
        .iter()
        .map(|s| {
            format!(
                "SPEAKER {} 1 {:.3} {:.3} <NA> <NA> {} <NA> <NA>\n",
                s.file_id, s.start, s.duration, s.speaker
            )
        })
        .collect()
}

/// Scoring options for [`score_der`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DerConfig {
    /// Seconds excluded on each side of every reference boundary.
    pub collar: f64,
    /// Exclude regions where the reference has 2+ speakers.
    pub skip_overlap: bool,
}

impl Default for DerConfig {
    /// Collar 0.25 s, overlap scored (the VoxConverse / md-eval
    /// convention).
    fn default() -> Self {
        Self {
            collar: 0.25,
            skip_overlap: false,
        }
    }
}

/// Error components of one scored recording (or a pool of them), in
/// seconds of speaker time.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DerReport {
    /// Scored reference speaker time (the DER denominator).
    pub total: f64,
    /// Reference speaker time with no hypothesis speaker to cover it.
    pub missed: f64,
    /// Hypothesis speaker time with no reference speaker under it.
    pub false_alarm: f64,
    /// Speaker time attributed to the wrong (unmapped) speaker.
    pub confusion: f64,
    /// Distinct reference speakers (summed when pooling).
    pub ref_speakers: usize,
    /// Distinct hypothesis speakers (summed when pooling).
    pub hyp_speakers: usize,
}

impl DerReport {
    /// `(missed + false_alarm + confusion) / total`. Zero when nothing
    /// was scored and nothing went wrong; infinite when only false
    /// alarms exist over an empty reference.
    #[must_use]
    pub fn der(&self) -> f64 {
        let errors = self.missed + self.false_alarm + self.confusion;
        if self.total > 0.0 {
            errors / self.total
        } else if errors > 0.0 {
            f64::INFINITY
        } else {
            0.0
        }
    }
}

impl AddAssign for DerReport {
    /// Pool two reports: every component is summed, so the pooled DER
    /// is time-weighted across recordings.
    fn add_assign(&mut self, rhs: Self) {
        self.total += rhs.total;
        self.missed += rhs.missed;
        self.false_alarm += rhs.false_alarm;
        self.confusion += rhs.confusion;
        self.ref_speakers += rhs.ref_speakers;
        self.hyp_speakers += rhs.hyp_speakers;
    }
}

/// Score `hypothesis` against `reference` for one recording.
///
/// `file_id`s are ignored; filter both slices to the same recording
/// first. Overlapping segments of the same speaker are merged.
#[must_use]
pub fn score_der(
    reference: &[RttmSegment],
    hypothesis: &[RttmSegment],
    config: &DerConfig,
) -> DerReport {
    let refs = speaker_timelines(reference);
    let hyps = speaker_timelines(hypothesis);

    let mut excluded: Vec<(f64, f64)> = Vec::new();
    if config.collar > 0.0 {
        for seg in reference {
            for t in [seg.start, seg.end()] {
                excluded.push((t - config.collar, t + config.collar));
            }
        }
    }
    let excluded = merge_intervals(excluded);

    let mut points: Vec<f64> = refs
        .iter()
        .chain(&hyps)
        .flat_map(|(_, ivs)| ivs.iter().flat_map(|&(a, b)| [a, b]))
        .chain(excluded.iter().flat_map(|&(a, b)| [a, b]))
        .collect();
    points.sort_by(f64::total_cmp);
    points.dedup();

    // Elementary intervals: (duration, active ref indices, active hyp indices).
    let mut intervals: Vec<(f64, Vec<usize>, Vec<usize>)> = Vec::new();
    for w in points.windows(2) {
        let (a, b) = (w[0], w[1]);
        let mid = 0.5 * (a + b);
        if b <= a || covers(&excluded, mid) {
            continue;
        }
        let r = active_at(&refs, mid);
        if config.skip_overlap && r.len() >= 2 {
            continue;
        }
        let h = active_at(&hyps, mid);
        if r.is_empty() && h.is_empty() {
            continue;
        }
        intervals.push((b - a, r, h));
    }

    let mut weights = vec![vec![0.0_f64; hyps.len()]; refs.len()];
    for (d, r, h) in &intervals {
        for &ri in r {
            for &hi in h {
                weights[ri][hi] += d;
            }
        }
    }
    let mapping = max_weight_assignment(&weights);

    let mut report = DerReport {
        ref_speakers: refs.len(),
        hyp_speakers: hyps.len(),
        ..DerReport::default()
    };
    for (d, r, h) in &intervals {
        let (nr, nh) = (r.len() as f64, h.len() as f64);
        let correct = r
            .iter()
            .filter(|&&ri| mapping[ri].is_some_and(|hi| h.contains(&hi)))
            .count() as f64;
        report.total += nr * d;
        report.missed += (nr - nh).max(0.0) * d;
        report.false_alarm += (nh - nr).max(0.0) * d;
        report.confusion += (nr.min(nh) - correct) * d;
    }
    report
}

/// Group segments by speaker into sorted, merged `(start, end)` lists.
/// Zero-length segments are dropped.
fn speaker_timelines(segments: &[RttmSegment]) -> Vec<(String, Vec<(f64, f64)>)> {
    let mut by_speaker: Vec<(String, Vec<(f64, f64)>)> = Vec::new();
    for seg in segments.iter().filter(|s| s.duration > 0.0) {
        match by_speaker.iter_mut().find(|(name, _)| *name == seg.speaker) {
            Some((_, ivs)) => ivs.push((seg.start, seg.end())),
            None => by_speaker.push((seg.speaker.clone(), vec![(seg.start, seg.end())])),
        }
    }
    by_speaker
        .into_iter()
        .map(|(name, ivs)| (name, merge_intervals(ivs)))
        .collect()
}

/// Sort and merge overlapping or touching intervals.
fn merge_intervals(mut ivs: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    ivs.sort_by(|x, y| x.0.total_cmp(&y.0));
    let mut out: Vec<(f64, f64)> = Vec::with_capacity(ivs.len());
    for (a, b) in ivs {
        match out.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(b),
            _ => out.push((a, b)),
        }
    }
    out
}

/// Whether `t` falls inside one of the sorted, disjoint `ivs`.
fn covers(ivs: &[(f64, f64)], t: f64) -> bool {
    let i = ivs.partition_point(|iv| iv.0 <= t);
    i > 0 && t < ivs[i - 1].1
}

/// Indices of the speakers whose timeline covers `t`.
fn active_at(timelines: &[(String, Vec<(f64, f64)>)], t: f64) -> Vec<usize> {
    timelines
        .iter()
        .enumerate()
        .filter(|(_, (_, ivs))| covers(ivs, t))
        .map(|(i, _)| i)
        .collect()
}

/// Maximum-weight one-to-one assignment of rows to columns.
///
/// Returns, for each row, the assigned column or `None` when the row is
/// left unmatched (more rows than columns).
///
/// Hungarian algorithm (Kuhn-Munkres, O(n^3)) on the square cost matrix
/// `max_weight - weight`, padded with zero-weight dummy rows/columns.
pub(crate) fn max_weight_assignment(weights: &[Vec<f64>]) -> Vec<Option<usize>> {
    let rows = weights.len();
    let cols = weights.first().map_or(0, Vec::len);
    let n = rows.max(cols);
    if n == 0 {
        return Vec::new();
    }
    let max_w = weights.iter().flatten().copied().fold(0.0_f64, f64::max);
    let cost = |i: usize, j: usize| -> f64 {
        let w = if i < rows && j < cols {
            weights[i][j]
        } else {
            0.0
        };
        max_w - w
    };

    // 1-based potentials and matching, as in the classic formulation.
    let mut u = vec![0.0_f64; n + 1];
    let mut v = vec![0.0_f64; n + 1];
    let mut matched_row = vec![0_usize; n + 1]; // column -> row
    let mut way = vec![0_usize; n + 1];
    for i in 1..=n {
        matched_row[0] = i;
        let mut j0 = 0;
        let mut minv = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = matched_row[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0;
            for j in 1..=n {
                if used[j] {
                    continue;
                }
                let cur = cost(i0 - 1, j - 1) - u[i0] - v[j];
                if cur < minv[j] {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if minv[j] < delta {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[matched_row[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if matched_row[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            matched_row[j0] = matched_row[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut assignment = vec![None; rows];
    for (j, &i) in matched_row.iter().enumerate().skip(1).take(cols) {
        if (1..=rows).contains(&i) {
            assignment[i - 1] = Some(j - 1);
        }
    }
    assignment
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    fn seg(speaker: &str, start: f64, end: f64) -> RttmSegment {
        RttmSegment {
            file_id: "f".to_string(),
            speaker: speaker.to_string(),
            start,
            duration: end - start,
        }
    }

    fn no_collar() -> DerConfig {
        DerConfig {
            collar: 0.0,
            skip_overlap: false,
        }
    }

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-6, "{a} != {b}");
    }

    fn two_turns() -> Vec<RttmSegment> {
        vec![seg("A", 0.0, 10.0), seg("B", 10.0, 20.0)]
    }

    #[test]
    fn parses_speaker_lines_and_skips_the_rest() {
        let text = ";; comment\n\nSPEAKER rec1 1 1.500 2.250 <NA> <NA> alice <NA> <NA>\n\
                    SPKR-INFO rec1 1 <NA> <NA> <NA> unknown alice <NA> <NA>\n\
                    SPEAKER rec1 1 4.0 1.0 <NA> <NA> bob <NA> <NA>\n";
        let segs = parse_rttm(text).unwrap();
        assert_eq!(
            segs,
            vec![
                RttmSegment {
                    file_id: "rec1".into(),
                    speaker: "alice".into(),
                    start: 1.5,
                    duration: 2.25
                },
                RttmSegment {
                    file_id: "rec1".into(),
                    speaker: "bob".into(),
                    start: 4.0,
                    duration: 1.0
                },
            ]
        );
    }

    #[test]
    fn malformed_lines_report_their_line_number() {
        let err = parse_rttm(
            "SPEAKER rec1 1 0.0 1.0 <NA> <NA> a <NA> <NA>\nSPEAKER rec1 1 x 1.0 <NA> <NA> a\n",
        )
        .unwrap_err();
        assert_eq!(err.line, 2);
        assert_eq!(parse_rttm("SPEAKER rec1 1 0.0\n").unwrap_err().line, 1);
        assert_eq!(
            parse_rttm("SPEAKER rec1 1 0.0 -1.0 <NA> <NA> a <NA> <NA>\n")
                .unwrap_err()
                .line,
            1
        );
    }

    #[test]
    fn write_then_parse_round_trips() {
        let segs = vec![seg("s0", 0.25, 1.5), seg("s1", 2.0, 3.125)];
        let text = write_rttm(&segs);
        assert!(text.starts_with("SPEAKER f 1 0.250 1.250 <NA> <NA> s0 <NA> <NA>\n"));
        assert_eq!(parse_rttm(&text).unwrap(), segs);
    }

    #[test]
    fn perfect_hypothesis_scores_zero() {
        let r = score_der(&two_turns(), &two_turns(), &no_collar());
        approx(r.total, 20.0);
        approx(r.der(), 0.0);
        assert_eq!((r.ref_speakers, r.hyp_speakers), (2, 2));
    }

    #[test]
    fn empty_hypothesis_is_all_missed() {
        let r = score_der(&two_turns(), &[], &no_collar());
        approx(r.missed, 20.0);
        approx(r.false_alarm + r.confusion, 0.0);
        approx(r.der(), 1.0);
    }

    #[test]
    fn swapped_labels_score_zero_after_mapping() {
        let hyp = vec![seg("spk1", 0.0, 10.0), seg("spk0", 10.0, 20.0)];
        approx(score_der(&two_turns(), &hyp, &no_collar()).der(), 0.0);
    }

    #[test]
    fn partial_confusion_counts_only_the_wrong_span() {
        // A→X (10 s), B→Y (5 s); B's [10, 15) is labelled X.
        let hyp = vec![seg("X", 0.0, 15.0), seg("Y", 15.0, 20.0)];
        let r = score_der(&two_turns(), &hyp, &no_collar());
        approx(r.confusion, 5.0);
        approx(r.missed + r.false_alarm, 0.0);
        approx(r.der(), 0.25);
    }

    #[test]
    fn hypothesis_beyond_reference_is_false_alarm() {
        let r = score_der(&[seg("A", 0.0, 10.0)], &[seg("X", 0.0, 12.0)], &no_collar());
        approx(r.false_alarm, 2.0);
        approx(r.der(), 0.2);
    }

    #[test]
    fn collar_excludes_errors_near_reference_boundaries() {
        let hyp = vec![seg("X", 0.0, 10.2), seg("Y", 10.2, 20.0)];
        let with = score_der(&two_turns(), &hyp, &DerConfig::default());
        // 20 s minus 0.25 (start) + 0.5 (turn change) + 0.25 (end).
        approx(with.total, 19.0);
        approx(with.der(), 0.0);
        let without = score_der(&two_turns(), &hyp, &no_collar());
        approx(without.confusion, 0.2);
        approx(without.der(), 0.01);
    }

    #[test]
    fn overlap_is_scored_unless_skipped() {
        let reference = vec![seg("A", 0.0, 10.0), seg("B", 5.0, 15.0)];
        let hyp = vec![seg("X", 0.0, 10.0), seg("Y", 10.0, 15.0)];
        let scored = score_der(&reference, &hyp, &no_collar());
        approx(scored.total, 20.0);
        approx(scored.missed, 5.0);
        approx(scored.der(), 0.25);
        let skipped = score_der(
            &reference,
            &hyp,
            &DerConfig {
                collar: 0.0,
                skip_overlap: true,
            },
        );
        approx(skipped.total, 10.0);
        approx(skipped.der(), 0.0);
    }

    #[test]
    fn same_speaker_overlapping_segments_count_once() {
        let hyp = vec![
            seg("X", 0.0, 6.0),
            seg("X", 4.0, 10.0),
            seg("Y", 10.0, 20.0),
        ];
        approx(score_der(&two_turns(), &hyp, &no_collar()).der(), 0.0);
    }

    #[test]
    fn extra_hypothesis_speaker_is_confusion_not_mapped() {
        // Three hypothesis speakers for two reference speakers: the
        // smallest one cannot be mapped.
        let hyp = vec![
            seg("X", 0.0, 10.0),
            seg("Y", 10.0, 17.0),
            seg("Z", 17.0, 20.0),
        ];
        let r = score_der(&two_turns(), &hyp, &no_collar());
        approx(r.confusion, 3.0);
        assert_eq!(r.hyp_speakers, 3);
    }

    #[test]
    fn pooling_is_time_weighted() {
        let mut pooled = score_der(&two_turns(), &[], &no_collar());
        pooled += score_der(&two_turns(), &two_turns(), &no_collar());
        approx(pooled.total, 40.0);
        approx(pooled.der(), 0.5);
        assert_eq!(pooled.ref_speakers, 4);
    }

    #[test]
    fn empty_reference_and_hypothesis_score_zero() {
        approx(score_der(&[], &[], &no_collar()).der(), 0.0);
        assert!(score_der(&[], &[seg("X", 0.0, 1.0)], &no_collar())
            .der()
            .is_infinite());
    }

    #[test]
    fn assignment_maximises_total_weight() {
        let w = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 6.0],
            vec![3.0, 6.0, 9.0],
        ];
        let a = max_weight_assignment(&w);
        let total: f64 = a.iter().enumerate().map(|(r, c)| w[r][c.unwrap()]).sum();
        assert!((total - 14.0).abs() < EPS, "total {total}");
        let rect = max_weight_assignment(&[vec![0.0, 5.0], vec![1.0, 0.0], vec![0.0, 9.0]]);
        assert_eq!(rect, vec![None, Some(0), Some(1)]);
    }
}
