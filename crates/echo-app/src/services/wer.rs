//! Word Error Rate (WER) computation.
//!
//! WER = (S + D + I) / N where:
//!
//! * S = substitutions
//! * D = deletions
//! * I = insertions
//! * N = number of reference words
//!
//! We use a token-level Levenshtein over normalized words. Normalization
//! is intentionally lightweight (lowercase, strip non-alphanumerics,
//! collapse whitespace) so we don't accidentally hide real model errors
//! behind aggressive cleaning. Punctuation differences should still
//! count if they change tokenization meaningfully.
//!
//! The algorithm is the classic O(N·M) DP. For Phase 0 fixtures we are
//! talking about hundreds of words at most, so a smarter algorithm
//! would just add complexity without meaningful wins.

use serde::{Deserialize, Serialize};

/// Aggregate WER stats for a single (reference, hypothesis) pair.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WerStats {
    /// Number of reference words after normalization.
    pub reference_words: u32,
    /// Number of hypothesis words after normalization.
    pub hypothesis_words: u32,
    /// Substitution edits.
    pub substitutions: u32,
    /// Deletion edits.
    pub deletions: u32,
    /// Insertion edits.
    pub insertions: u32,
}

impl WerStats {
    /// `0.0` when reference is empty *and* hypothesis is empty,
    /// `1.0` when reference is empty but hypothesis isn't.
    #[must_use]
    pub fn wer(&self) -> f64 {
        if self.reference_words == 0 {
            return if self.hypothesis_words == 0 { 0.0 } else { 1.0 };
        }
        let edits = u64::from(self.substitutions + self.deletions + self.insertions);
        edits as f64 / f64::from(self.reference_words)
    }

    /// Sum two stats. Useful to roll up multiple fixtures into a single
    /// global WER, weighted by reference length.
    #[must_use]
    pub fn merge(self, other: Self) -> Self {
        Self {
            reference_words: self.reference_words + other.reference_words,
            hypothesis_words: self.hypothesis_words + other.hypothesis_words,
            substitutions: self.substitutions + other.substitutions,
            deletions: self.deletions + other.deletions,
            insertions: self.insertions + other.insertions,
        }
    }
}

/// Lightweight normalization: lowercase, strip non-alphanumeric chars
/// from each token, drop empty tokens. Apostrophes inside words are
/// preserved so `"don't"` stays one token.
#[must_use]
pub fn normalize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '\'')
                .collect::<String>()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// Compute WER stats between a reference and hypothesis transcript.
#[must_use]
pub fn compute(reference: &str, hypothesis: &str) -> WerStats {
    let r = normalize(reference);
    let h = normalize(hypothesis);
    levenshtein(&r, &h)
}

/// Token-level Levenshtein with edit-type breakdown. Returns the
/// counts; `WerStats::wer` derives the actual ratio.
fn levenshtein(reference: &[String], hypothesis: &[String]) -> WerStats {
    let n = reference.len();
    let m = hypothesis.len();

    if n == 0 {
        return WerStats {
            reference_words: 0,
            hypothesis_words: m as u32,
            substitutions: 0,
            deletions: 0,
            insertions: m as u32,
        };
    }
    if m == 0 {
        return WerStats {
            reference_words: n as u32,
            hypothesis_words: 0,
            substitutions: 0,
            deletions: n as u32,
            insertions: 0,
        };
    }

    // dp[i][j] = (cost, sub, del, ins) for ref[..i] vs hyp[..j].
    // We store the operation breakdown alongside the cost so we can
    // report it without backtracking through the full matrix.
    #[derive(Clone, Copy)]
    struct Cell {
        cost: u32,
        sub: u32,
        del: u32,
        ins: u32,
    }

    let mut prev: Vec<Cell> = (0..=m)
        .map(|j| Cell {
            cost: j as u32,
            sub: 0,
            del: 0,
            ins: j as u32,
        })
        .collect();
    let mut curr: Vec<Cell> = vec![
        Cell {
            cost: 0,
            sub: 0,
            del: 0,
            ins: 0,
        };
        m + 1
    ];

    for (i, r_word) in reference.iter().enumerate() {
        curr[0] = Cell {
            cost: (i + 1) as u32,
            sub: 0,
            del: (i + 1) as u32,
            ins: 0,
        };
        for (j, h_word) in hypothesis.iter().enumerate() {
            let match_eq = r_word == h_word;
            // Three candidate origins: substitute, delete (skip ref), insert (skip hyp).
            let sub_cell = prev[j];
            let del_cell = prev[j + 1];
            let ins_cell = curr[j];

            let sub_cost = sub_cell.cost + u32::from(!match_eq);
            let del_cost = del_cell.cost + 1;
            let ins_cost = ins_cell.cost + 1;

            // Pick min; ties prefer match > sub > del > ins so we don't
            // over-count substitutions when an exact match is possible.
            let mut best = sub_cell;
            best.cost = sub_cost;
            if !match_eq {
                best.sub += 1;
            }
            if del_cost < best.cost {
                best = del_cell;
                best.cost = del_cost;
                best.del += 1;
            }
            if ins_cost < best.cost {
                best = ins_cell;
                best.cost = ins_cost;
                best.ins += 1;
            }
            curr[j + 1] = best;
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    let final_cell = prev[m];
    WerStats {
        reference_words: n as u32,
        hypothesis_words: m as u32,
        substitutions: final_cell.sub,
        deletions: final_cell.del,
        insertions: final_cell.ins,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn identical_strings_have_zero_wer() {
        let s = compute("the quick brown fox", "the quick brown fox");
        assert_eq!(s.substitutions, 0);
        assert_eq!(s.deletions, 0);
        assert_eq!(s.insertions, 0);
        assert!(approx(s.wer(), 0.0));
    }

    #[test]
    fn case_and_punctuation_normalized() {
        let s = compute("The Quick, Brown FOX!", "the quick brown fox");
        assert!(approx(s.wer(), 0.0));
    }

    #[test]
    fn one_substitution_in_four_is_quarter_wer() {
        let s = compute("the quick brown fox", "the quick brown dog");
        assert_eq!(s.substitutions, 1);
        assert_eq!(s.deletions, 0);
        assert_eq!(s.insertions, 0);
        assert!(approx(s.wer(), 0.25));
    }

    #[test]
    fn deletion_counts_against_wer() {
        let s = compute("the quick brown fox", "the quick fox");
        assert_eq!(s.deletions, 1);
        assert!(approx(s.wer(), 0.25));
    }

    #[test]
    fn insertion_counts_against_wer() {
        let s = compute("the quick brown fox", "the very quick brown fox");
        assert_eq!(s.insertions, 1);
        assert!(approx(s.wer(), 0.25));
    }

    #[test]
    fn empty_reference_with_hypothesis_is_total_error() {
        let s = compute("", "anything goes here");
        assert_eq!(s.reference_words, 0);
        assert_eq!(s.insertions, 3);
        assert!(approx(s.wer(), 1.0));
    }

    #[test]
    fn empty_both_is_zero_wer() {
        let s = compute("", "");
        assert!(approx(s.wer(), 0.0));
    }

    #[test]
    fn merge_aggregates_correctly() {
        let a = compute("hello world", "hello mars");
        let b = compute("rust is great", "rust is great");
        let m = a.clone().merge(b.clone());
        assert_eq!(m.reference_words, 5);
        assert_eq!(m.substitutions, 1);
        // Global WER = 1 / 5 = 0.20, even though file a alone was 0.50.
        assert!(approx(m.wer(), 0.2));
    }

    #[test]
    fn apostrophes_kept_inside_words() {
        let toks = normalize("don't won't can't");
        assert_eq!(toks, vec!["don't", "won't", "can't"]);
    }

    #[test]
    fn matrix_picks_global_minimum_not_greedy() {
        // Greedy left-to-right would pick more subs; the DP must find
        // the alignment with the cheapest total edit cost.
        let s = compute("a b c d e", "x a b c d e");
        assert_eq!(s.insertions, 1);
        assert_eq!(s.substitutions, 0);
        assert_eq!(s.deletions, 0);
    }
}
