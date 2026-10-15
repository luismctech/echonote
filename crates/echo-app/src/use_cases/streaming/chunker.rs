//! Speech-aligned chunk boundaries.
//!
//! The pipeline feeds every analysed stretch of the buffer through the
//! VAD and records one [`TimedFrame`] per VAD window, positioned in
//! buffer coordinates. [`SpeechTimeline::next_cut`] then decides where
//! the next chunk ends:
//!
//! - leading silence (minus a short pre-roll) is cut off on its own and
//!   reported as skipped, so the transcriber never sees it;
//! - once at least `min` samples are buffered, speech is cut at the
//!   first speech → silence transition (the hysteresis end of an
//!   utterance) at or after `min`, or at the last transition when the
//!   buffer already ends in silence (a short utterance followed by a
//!   pause);
//! - when no transition shows up by `max`, the cut is forced at the
//!   quietest window (lowest VAD probability) in the `lookback` before
//!   `max`, which avoids splitting a word whenever a dip exists;
//! - a speech cut holding less than `min_speech` of confident speech
//!   (windows at or above [`CONFIDENT_SPEECH_PROBABILITY`]) is
//!   reported as silence instead, so noise blips never reach the
//!   transcriber.
//!
//! All positions are sample indices into the pipeline buffer (device
//! rate, interleaved), so cuts map 1:1 to buffer drains.

use echo_domain::VoiceState;

/// Chunk length bounds and cut tuning, in buffer samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ChunkBounds {
    /// Speech shorter than this is only cut when it is followed by
    /// silence until the end of the analysed audio.
    pub min: usize,
    /// Hard ceiling: a chunk never exceeds it while VAD frames exist.
    pub max: usize,
    /// Window before `max` searched for the quietest forced cut.
    pub lookback: usize,
    /// Silence kept before a speech onset so the first phoneme is not
    /// clipped by a late VAD decision.
    pub pre_roll: usize,
    /// Least confident speech (windows at or above
    /// [`CONFIDENT_SPEECH_PROBABILITY`]) a chunk needs to be
    /// transcribed; weaker chunks are skipped as silence.
    pub min_speech: usize,
}

/// Speech probability at which a VAD window counts as confident
/// speech. Above the detector's start threshold, so windows kept
/// voiced only by hysteresis or by a hesitant onset do not count.
pub(super) const CONFIDENT_SPEECH_PROBABILITY: f32 = 0.5;

/// One VAD window positioned in buffer coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct TimedFrame {
    /// Exclusive end of the window in the buffer.
    pub end: usize,
    /// Raw speech probability of the window.
    pub probability: f32,
    /// VAD state after the window, hysteresis included.
    pub state: VoiceState,
}

/// Where the next chunk ends and what to do with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Cut {
    /// `[0, n)` holds no speech: report it as skipped.
    Silence(usize),
    /// `[0, n)` holds speech: transcribe it.
    Speech(usize),
}

/// VAD frames covering the analysed prefix of the pipeline buffer.
#[derive(Debug, Default)]
pub(super) struct SpeechTimeline {
    frames: Vec<TimedFrame>,
}

impl SpeechTimeline {
    /// Append a frame. Ends must not go backwards; a frame that does
    /// not advance the timeline is ignored.
    pub fn push(&mut self, frame: TimedFrame) {
        if frame.end > self.analyzed() {
            self.frames.push(frame);
        }
    }

    /// Buffer samples covered by frames so far.
    pub fn analyzed(&self) -> usize {
        self.frames.last().map_or(0, |f| f.end)
    }

    /// Forget the first `n` buffer samples after they were drained.
    pub fn consume(&mut self, n: usize) {
        self.frames.retain(|f| f.end > n);
        for f in &mut self.frames {
            f.end -= n;
        }
    }

    /// Drop every frame (pause, session reset).
    pub fn clear(&mut self) {
        self.frames.clear();
    }

    /// Next cut while audio keeps arriving, or `None` to wait for more.
    pub fn next_cut(&self, bounds: &ChunkBounds) -> Option<Cut> {
        let analyzed = self.analyzed();
        let Some(onset) = self.speech_onset() else {
            if analyzed < bounds.min {
                return None;
            }
            let cut = self.last_end_at_or_before(analyzed.saturating_sub(bounds.pre_roll));
            return (cut > 0).then_some(Cut::Silence(cut));
        };

        let lead = self.last_end_at_or_before(onset.saturating_sub(bounds.pre_roll));
        if lead > 0 {
            return Some(Cut::Silence(lead));
        }
        if analyzed < bounds.min {
            return None;
        }

        let ends: Vec<usize> = self.speech_ends().filter(|&t| t <= bounds.max).collect();
        if let Some(&t) = ends.iter().find(|&&t| t >= bounds.min) {
            return Some(self.speech(t, bounds));
        }
        let ends_in_silence = self
            .frames
            .last()
            .is_some_and(|f| f.state == VoiceState::Silence);
        if ends_in_silence {
            if let Some(&t) = ends.last() {
                return Some(self.speech(t, bounds));
            }
        }
        if analyzed >= bounds.max {
            return Some(self.speech(self.quietest_cut(bounds), bounds));
        }
        None
    }

    /// Next cut when no more audio will arrive (stop, pause): the
    /// whole `len`-sample buffer must be emitted. Call repeatedly,
    /// draining each cut, until it returns `None`.
    pub fn final_cut(&self, bounds: &ChunkBounds, len: usize) -> Option<Cut> {
        if len == 0 {
            return None;
        }
        if let Some(cut) = self.next_cut(bounds) {
            return Some(cut);
        }
        let Some(onset) = self.speech_onset() else {
            return Some(Cut::Silence(len));
        };
        let lead = self.last_end_at_or_before(onset.saturating_sub(bounds.pre_roll));
        if lead > 0 {
            return Some(Cut::Silence(lead));
        }
        let ends_in_silence = self
            .frames
            .last()
            .is_some_and(|f| f.state == VoiceState::Silence);
        match self.speech_ends().last() {
            Some(t) if ends_in_silence && t < len => Some(self.speech(t, bounds)),
            _ => Some(self.speech(len, bounds)),
        }
    }

    /// Classify the speech chunk `[0, n)`: [`Cut::Speech`] when it holds
    /// at least `bounds.min_speech` samples of confident speech,
    /// [`Cut::Silence`] otherwise.
    fn speech(&self, n: usize, bounds: &ChunkBounds) -> Cut {
        if self.confident_speech(n) >= bounds.min_speech {
            Cut::Speech(n)
        } else {
            Cut::Silence(n)
        }
    }

    /// Samples in `[0, n)` covered by windows at or above
    /// [`CONFIDENT_SPEECH_PROBABILITY`].
    fn confident_speech(&self, n: usize) -> usize {
        self.frames
            .iter()
            .enumerate()
            .filter(|(_, f)| f.probability >= CONFIDENT_SPEECH_PROBABILITY)
            .map(|(i, f)| f.end.min(n).saturating_sub(self.start(i)))
            .sum()
    }

    /// Start of the window at index `i`.
    fn start(&self, i: usize) -> usize {
        if i == 0 {
            0
        } else {
            self.frames[i - 1].end
        }
    }

    /// Start of the first voiced window, if any.
    fn speech_onset(&self) -> Option<usize> {
        self.frames
            .iter()
            .position(|f| f.state == VoiceState::Voiced)
            .map(|i| self.start(i))
    }

    /// Start of every silent window that follows a voiced one.
    fn speech_ends(&self) -> impl Iterator<Item = usize> + '_ {
        self.frames.windows(2).filter_map(|w| {
            (w[0].state == VoiceState::Voiced && w[1].state == VoiceState::Silence)
                .then_some(w[0].end)
        })
    }

    /// Largest window end `<= limit`, or 0.
    fn last_end_at_or_before(&self, limit: usize) -> usize {
        self.frames
            .iter()
            .map(|f| f.end)
            .take_while(|&e| e <= limit)
            .last()
            .unwrap_or(0)
    }

    /// End of the lowest-probability window ending in
    /// `[max - lookback, max]`; the latest one wins ties so chunks stay
    /// as long as allowed. Falls back to the last end `<= max`, then to
    /// `max` itself.
    fn quietest_cut(&self, bounds: &ChunkBounds) -> usize {
        let from = bounds.max.saturating_sub(bounds.lookback);
        self.frames
            .iter()
            .filter(|f| f.end >= from && f.end <= bounds.max && f.end > 0)
            .fold(None::<&TimedFrame>, |best, f| match best {
                Some(b) if b.probability < f.probability => Some(b),
                _ => Some(f),
            })
            .map(|f| f.end)
            .or_else(|| {
                let end = self.last_end_at_or_before(bounds.max);
                (end > 0).then_some(end)
            })
            .unwrap_or(bounds.max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use VoiceState::{Silence as S, Voiced as V};

    /// 1 "second" = 10 samples keeps the arithmetic readable.
    const SEC: usize = 10;

    fn bounds() -> ChunkBounds {
        ChunkBounds {
            min: 3 * SEC,
            max: 8 * SEC,
            lookback: 2 * SEC,
            pre_roll: 3,
            min_speech: SEC / 2,
        }
    }

    /// One frame per second, all voiced, with the given probabilities.
    fn voiced_with(probabilities: &[f32]) -> SpeechTimeline {
        let mut t = SpeechTimeline::default();
        for (k, &probability) in probabilities.iter().enumerate() {
            t.push(TimedFrame {
                end: (k + 1) * SEC,
                probability,
                state: V,
            });
        }
        t
    }

    /// Append `n` one-second silent frames.
    fn then_silence(mut t: SpeechTimeline, n: usize) -> SpeechTimeline {
        for _ in 0..n {
            let end = t.analyzed() + SEC;
            t.push(TimedFrame {
                end,
                probability: 0.1,
                state: S,
            });
        }
        t
    }

    /// One frame per second with the given states, probability 0.9
    /// for voiced and 0.1 for silent windows.
    fn timeline(states: &[VoiceState]) -> SpeechTimeline {
        let mut t = SpeechTimeline::default();
        for (k, &state) in states.iter().enumerate() {
            let probability = if state == V { 0.9 } else { 0.1 };
            t.push(TimedFrame {
                end: (k + 1) * SEC,
                probability,
                state,
            });
        }
        t
    }

    #[test]
    fn waits_until_min_before_cutting_speech() {
        assert_eq!(timeline(&[V, V]).next_cut(&bounds()), None);
        // Transition at 2 s but speech resumed: wait for a later end.
        assert_eq!(timeline(&[V, V, S, V]).next_cut(&bounds()), None);
    }

    #[test]
    fn cuts_at_the_first_speech_end_at_or_after_min() {
        let t = timeline(&[V, V, S, V, V, S, V, S]);
        assert_eq!(t.next_cut(&bounds()), Some(Cut::Speech(5 * SEC)));
    }

    #[test]
    fn short_utterance_followed_by_silence_is_cut_at_its_end() {
        let t = timeline(&[V, S, S]);
        assert_eq!(t.next_cut(&bounds()), Some(Cut::Speech(SEC)));
    }

    #[test]
    fn leading_silence_is_cut_off_keeping_the_pre_roll() {
        let t = timeline(&[S, S, V]);
        // Onset at 20, pre-roll 3 ⇒ largest end ≤ 17 is 10.
        assert_eq!(t.next_cut(&bounds()), Some(Cut::Silence(SEC)));
    }

    #[test]
    fn pure_silence_is_released_once_min_is_buffered() {
        assert_eq!(timeline(&[S, S]).next_cut(&bounds()), None);
        assert_eq!(
            timeline(&[S, S, S]).next_cut(&bounds()),
            Some(Cut::Silence(2 * SEC))
        );
    }

    #[test]
    fn forced_cut_lands_on_the_quietest_window_before_max() {
        let mut t = SpeechTimeline::default();
        let probs = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.4, 0.8, 0.9];
        for (k, p) in probs.iter().enumerate() {
            t.push(TimedFrame {
                end: (k + 1) * SEC,
                probability: *p,
                state: V,
            });
        }
        // Lookback [60, 80]: ends 60 (0.9), 70 (0.4), 80 (0.8).
        assert_eq!(t.next_cut(&bounds()), Some(Cut::Speech(7 * SEC)));
    }

    #[test]
    fn forced_cut_prefers_the_latest_window_on_ties() {
        let t = timeline(&[V; 9]);
        assert_eq!(t.next_cut(&bounds()), Some(Cut::Speech(8 * SEC)));
    }

    #[test]
    fn speech_end_beyond_max_falls_back_to_a_forced_cut() {
        let mut states = vec![V; 9];
        states.push(S);
        // The only transition (90) lies past max (80).
        assert_eq!(
            timeline(&states).next_cut(&bounds()),
            Some(Cut::Speech(8 * SEC))
        );
    }

    #[test]
    fn consume_rebases_the_remaining_frames() {
        let mut t = timeline(&[V, V, S, S]);
        t.consume(2 * SEC);
        assert_eq!(t.analyzed(), 2 * SEC);
        assert_eq!(t.next_cut(&bounds()), None, "only silence < min remains");
    }

    #[test]
    fn final_cut_emits_short_speech_and_trims_trailing_silence() {
        let b = bounds();
        let t = timeline(&[V, V]);
        assert_eq!(t.final_cut(&b, 2 * SEC + 4), Some(Cut::Speech(2 * SEC + 4)));

        let t = timeline(&[V, S]);
        assert_eq!(t.final_cut(&b, 2 * SEC), Some(Cut::Speech(SEC)));

        let t = timeline(&[S]);
        assert_eq!(t.final_cut(&b, SEC + 3), Some(Cut::Silence(SEC + 3)));
        assert_eq!(SpeechTimeline::default().final_cut(&b, 0), None);
        assert_eq!(
            SpeechTimeline::default().final_cut(&b, 5),
            Some(Cut::Silence(5))
        );
    }

    #[test]
    fn hesitant_blip_followed_by_silence_is_skipped() {
        // Voiced by hysteresis only: no window reaches the confidence floor.
        let t = then_silence(voiced_with(&[0.4]), 2);
        assert_eq!(t.next_cut(&bounds()), Some(Cut::Silence(SEC)));
    }

    #[test]
    fn hesitant_forced_cut_is_skipped() {
        let t = voiced_with(&[0.4; 9]);
        assert_eq!(t.next_cut(&bounds()), Some(Cut::Silence(8 * SEC)));
    }

    #[test]
    fn hesitant_final_chunk_is_skipped() {
        let t = voiced_with(&[0.4, 0.4]);
        assert_eq!(t.final_cut(&bounds(), 2 * SEC), Some(Cut::Silence(2 * SEC)));
    }

    #[test]
    fn confident_speech_at_the_floor_is_transcribed() {
        // One confident second clears the half-second floor.
        let t = then_silence(voiced_with(&[0.4, 0.5]), 1);
        assert_eq!(t.next_cut(&bounds()), Some(Cut::Speech(2 * SEC)));
    }
}
