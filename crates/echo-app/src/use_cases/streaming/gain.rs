//! Automatic gain stage applied to captured audio before the pipeline
//! analyses it.
//!
//! A live session captured at about -54 dBFS made the Silero VAD mark
//! ~25 s of real speech as silence, while the same audio raised by
//! +28 dB lost nothing. [`AutoGain`] therefore normalises the speech
//! level of every captured frame before the chunker's VAD analysis,
//! the RMS gate, Whisper, the diarizer and the saved audio see it.
//!
//! The stage is pure and deterministic (no I/O, no clocks) and works on
//! interleaved device-format frames of any rate and channel count, so
//! it behaves identically on every OS.
//!
//! # Algorithm
//!
//! - The input is measured in [`BLOCK_MS`] blocks (mean square over
//!   every channel of the block's frames).
//! - A noise floor is the minimum block level over the last
//!   [`FLOOR_WINDOW_MS`]. A block is *voiced* when it is at least
//!   [`VOICED_ABOVE_FLOOR_DB`] above that floor and above
//!   [`MIN_VOICED_RMS`]; only voiced blocks inform the speech level, so
//!   steady noise or silence never drives the gain up.
//! - The speech level is the [`SPEECH_PERCENTILE`] of voiced block
//!   levels over the last [`WINDOW_MS`]. It is trusted once at least
//!   [`MIN_VOICED_BLOCKS`] voiced blocks are in the window; until then,
//!   and whenever too few remain (long silence), the current gain is
//!   held. A fresh stage starts at 1.0.
//! - The wanted gain is `TARGET_RMS / level`, clamped to
//!   `[1.0, MAX_GAIN]`: quiet input is raised, loud input is never
//!   attenuated.
//! - The applied gain moves towards the wanted gain by at most
//!   [`MAX_RISE_DB`] (up) or [`MAX_FALL_DB`] (down) per block, ramped
//!   linearly sample by sample within the block, so there are no steps.
//! - While the gain is above 1.0 a soft limiter bends samples above
//!   [`LIMITER_KNEE`] smoothly towards full scale, so output stays in
//!   `[-1, 1]` on transients. At exactly 1.0 the input is passed through
//!   bit-identical.

use std::collections::VecDeque;

use echo_domain::{AudioFormat, Sample};

/// Length of one level-measurement block.
const BLOCK_MS: u32 = 20;
/// Sliding window the speech level is estimated over.
const WINDOW_MS: u32 = 10_000;
/// Window of the noise-floor minimum tracker. Shorter than
/// [`WINDOW_MS`] so the floor follows a changing room within seconds.
const FLOOR_WINDOW_MS: u32 = 5_000;
/// Percentile of voiced block levels taken as the speech level: high
/// enough to track the voice rather than its quiet tails.
const SPEECH_PERCENTILE: f32 = 0.9;
/// Speech RMS the gain aims for (≈ -26 dBFS), a normal close-mic level.
pub(super) const TARGET_RMS: f32 = 0.05;
/// Largest gain applied (+30 dB), enough for a -56 dBFS speaker while
/// keeping a near-silent input from being raised into loud noise.
pub(super) const MAX_GAIN: f32 = 31.622_776;
/// A block counts as voiced at this many dB above the noise floor
/// (2× its RMS).
const VOICED_ABOVE_FLOOR_DB: f32 = 6.0;
/// Blocks quieter than this (-80 dBFS) never count as voiced, so
/// digital dither is not mistaken for speech.
const MIN_VOICED_RMS: f32 = 1e-4;
/// Voiced blocks (≈ 300 ms of speech) required before the level is
/// trusted.
const MIN_VOICED_BLOCKS: usize = 15;
/// Largest gain increase per block.
const MAX_RISE_DB: f32 = 1.0;
/// Largest gain decrease per block: faster than the rise so a louder
/// talker is not over-amplified for long.
const MAX_FALL_DB: f32 = 3.0;
/// Level above which the soft limiter starts bending samples.
const LIMITER_KNEE: f32 = 0.9;
/// Level reported for an all-zero block, keeping the dB maths finite.
const SILENCE_DB: f32 = -120.0;

/// Automatic gain control for one capture stream.
///
/// See the [module docs](self) for the algorithm. Feed every captured
/// frame through [`AutoGain::process`] in capture order.
pub(super) struct AutoGain {
    channels: usize,
    block_frames: usize,
    window_blocks: usize,
    floor_blocks: usize,
    /// Level (dB) of each recent block and whether it was voiced,
    /// newest last; at most `window_blocks` entries.
    history: VecDeque<(f32, bool)>,
    /// Sum of squares and frame count of the block being measured.
    block_sum_sq: f64,
    block_len: usize,
    /// Gain at the start of the current block and at its end.
    gain_from: f32,
    gain_to: f32,
}

impl AutoGain {
    /// Creates a stage for frames in `format`, starting at gain 1.0.
    pub(super) fn new(format: AudioFormat) -> Self {
        let blocks_in = |ms: u32| (ms / BLOCK_MS) as usize;
        let block_frames =
            (u64::from(format.sample_rate_hz) * u64::from(BLOCK_MS) / 1_000) as usize;
        Self {
            channels: usize::from(format.channels.max(1)),
            block_frames: block_frames.max(1),
            window_blocks: blocks_in(WINDOW_MS),
            floor_blocks: blocks_in(FLOOR_WINDOW_MS),
            history: VecDeque::with_capacity(blocks_in(WINDOW_MS)),
            block_sum_sq: 0.0,
            block_len: 0,
            gain_from: 1.0,
            gain_to: 1.0,
        }
    }

    /// Gain the stage is currently heading to (applied at the end of the
    /// block being processed).
    pub(super) fn gain(&self) -> f32 {
        self.gain_to
    }

    /// Forgets the level history after a gap (pause/resume): the noise
    /// floor and speech level may have changed. The gain itself is kept
    /// as the starting point because the device and speaker are the
    /// same, so speech right after the resume is not left quiet while
    /// the history refills.
    pub(super) fn reset(&mut self) {
        self.history.clear();
        self.block_sum_sq = 0.0;
        self.block_len = 0;
        self.gain_from = self.gain_to;
    }

    /// Applies the gain in place to interleaved `samples` (whole frames;
    /// a trailing partial frame is processed as if complete) and updates
    /// the level estimate from the unmodified input.
    pub(super) fn process(&mut self, samples: &mut [Sample]) {
        for frame in samples.chunks_mut(self.channels) {
            self.block_sum_sq += frame
                .iter()
                .map(|&s| f64::from(s) * f64::from(s))
                .sum::<f64>()
                / frame.len() as f64;
            self.block_len += 1;

            if self.gain_from != 1.0 || self.gain_to != 1.0 {
                let progress = self.block_len as f32 / self.block_frames as f32;
                let gain = self.gain_from + (self.gain_to - self.gain_from) * progress;
                for s in frame.iter_mut() {
                    *s = soft_limit(*s * gain);
                }
            }

            if self.block_len == self.block_frames {
                self.end_block();
            }
        }
    }

    /// Closes the measured block: records its level and sets the gain
    /// the next block ramps to.
    fn end_block(&mut self) {
        let mean_sq = self.block_sum_sq / self.block_len as f64;
        self.block_sum_sq = 0.0;
        self.block_len = 0;
        self.gain_from = self.gain_to;
        if !mean_sq.is_finite() {
            // Corrupt input: keep the history and the gain unchanged.
            return;
        }

        let rms = mean_sq.sqrt() as f32;
        let level = if rms > 0.0 {
            to_db(rms).max(SILENCE_DB)
        } else {
            SILENCE_DB
        };
        let floor = self
            .history
            .iter()
            .rev()
            .take(self.floor_blocks.saturating_sub(1))
            .map(|&(db, _)| db)
            .fold(level, f32::min);
        let voiced = rms >= MIN_VOICED_RMS && level - floor >= VOICED_ABOVE_FLOOR_DB;
        if self.history.len() == self.window_blocks {
            self.history.pop_front();
        }
        self.history.push_back((level, voiced));

        let Some(speech_db) = self.speech_level() else {
            return; // Not enough speech: hold the current gain.
        };
        let wanted = from_db(to_db(TARGET_RMS) - speech_db).clamp(1.0, MAX_GAIN);
        let current = to_db(self.gain_to);
        let step = (to_db(wanted) - current).clamp(-MAX_FALL_DB, MAX_RISE_DB);
        self.gain_to = from_db(current + step).clamp(1.0, MAX_GAIN);
        if (self.gain_to - wanted).abs() <= wanted * 1e-4 {
            // Land exactly on the wanted gain (in particular on 1.0).
            self.gain_to = wanted;
        }
    }

    /// High percentile of voiced block levels in the window, in dB, or
    /// `None` while fewer than [`MIN_VOICED_BLOCKS`] are voiced.
    fn speech_level(&self) -> Option<f32> {
        let mut voiced: Vec<f32> = self
            .history
            .iter()
            .filter(|&&(_, v)| v)
            .map(|&(db, _)| db)
            .collect();
        if voiced.len() < MIN_VOICED_BLOCKS {
            return None;
        }
        let rank = ((voiced.len() - 1) as f32 * SPEECH_PERCENTILE) as usize;
        let (_, level, _) = voiced.select_nth_unstable_by(rank, f32::total_cmp);
        Some(*level)
    }
}

fn to_db(linear: f32) -> f32 {
    20.0 * linear.log10()
}

fn from_db(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

/// Passes samples up to [`LIMITER_KNEE`] unchanged and bends louder
/// ones smoothly (tanh) towards, but never past, full scale.
fn soft_limit(x: Sample) -> Sample {
    let magnitude = x.abs();
    if magnitude <= LIMITER_KNEE {
        return x;
    }
    let headroom = 1.0 - LIMITER_KNEE;
    let bent = LIMITER_KNEE + headroom * ((magnitude - LIMITER_KNEE) / headroom).tanh();
    // tanh saturates to exactly 1.0 in f32 for huge inputs.
    bent.min(1.0).copysign(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 16_000;
    const MONO: AudioFormat = AudioFormat {
        sample_rate_hz: SR,
        channels: 1,
    };

    fn rms(samples: &[Sample]) -> f32 {
        let sum: f32 = samples.iter().map(|s| s * s).sum();
        (sum / samples.len().max(1) as f32).sqrt()
    }

    fn db(x: f32) -> f32 {
        20.0 * x.log10()
    }

    /// 440 Hz tone at `voiced_rms`, 200 ms on / 100 ms off, with the
    /// gaps at `gap_amplitude` (deterministic, near silent).
    fn speech_like(frames: usize, rate: u32, voiced_rms: f32) -> Vec<Sample> {
        let amplitude = voiced_rms * std::f32::consts::SQRT_2;
        let period = rate as usize * 300 / 1_000;
        let on = rate as usize * 200 / 1_000;
        (0..frames)
            .map(|i| {
                if i % period < on {
                    let t = i as f32 / rate as f32;
                    amplitude * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Deterministic uniform noise (LCG) scaled to `target_rms`.
    fn noise(len: usize, target_rms: f32) -> Vec<Sample> {
        let mut state: u32 = 0x1234_5678;
        // Uniform in [-a, a] has RMS a / sqrt(3).
        let a = target_rms * 3.0_f32.sqrt();
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let u = (state >> 8) as f32 / (1u32 << 24) as f32;
                (2.0 * u - 1.0) * a
            })
            .collect()
    }

    /// Feeds `input` in 10 ms frames, like a capture callback would.
    fn run(gain: &mut AutoGain, input: &[Sample], frame_len: usize) -> Vec<Sample> {
        let mut out = input.to_vec();
        for frame in out.chunks_mut(frame_len) {
            gain.process(frame);
        }
        out
    }

    #[test]
    fn quiet_speech_is_raised_to_the_target_level() {
        let input = speech_like(20 * SR as usize, SR, 0.003);
        let mut agc = AutoGain::new(MONO);
        let out = run(&mut agc, &input, 160);

        // After 5 s every voiced 200 ms burst sits near the target.
        let period = SR as usize * 300 / 1_000;
        let on = SR as usize * 200 / 1_000;
        let start = 5 * SR as usize / period * period;
        let mut bursts = 0;
        for burst in (start..out.len() - period).step_by(period) {
            let level = rms(&out[burst + 16..burst + on - 16]);
            let off_by = (db(level) - db(TARGET_RMS)).abs();
            assert!(
                off_by <= 6.0,
                "burst at {burst}: rms {level} is {off_by} dB off"
            );
            bursts += 1;
        }
        assert!(bursts > 40);
        assert!(agc.gain() > 1.0 && agc.gain() <= MAX_GAIN);
    }

    #[test]
    fn loud_input_passes_through_bit_identical() {
        let input: Vec<Sample> = (0..10 * SR as usize)
            .map(|i| 0.3 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SR as f32).sin())
            .collect();
        let mut agc = AutoGain::new(MONO);
        let out = run(&mut agc, &input, 160);
        assert_eq!(agc.gain(), 1.0);
        assert!(out
            .iter()
            .zip(&input)
            .all(|(o, i)| o.to_bits() == i.to_bits()));
    }

    /// Steady low noise has no speech-like dynamics, so the gain stays
    /// exactly 1.0 for the whole run: silence alone never ramps it up.
    #[test]
    fn steady_noise_alone_never_raises_the_gain() {
        let input = noise(30 * SR as usize, 0.0005);
        let mut agc = AutoGain::new(MONO);
        let mut out = input.clone();
        for frame in out.chunks_mut(160) {
            agc.process(frame);
            assert_eq!(agc.gain(), 1.0);
        }
        assert!(out
            .iter()
            .zip(&input)
            .all(|(o, i)| o.to_bits() == i.to_bits()));
    }

    #[test]
    fn output_stays_within_full_scale_on_a_loud_transient() {
        let mut input = speech_like(10 * SR as usize, SR, 0.003);
        // A 50 ms near-full-scale click after the gain has risen.
        for (i, s) in input[8 * SR as usize..][..800].iter_mut().enumerate() {
            *s = if i % 2 == 0 { 0.95 } else { -0.95 };
        }
        let mut agc = AutoGain::new(MONO);
        let out = run(&mut agc, &input, 160);
        assert!(agc.gain() > 2.0, "the gain must have risen: {}", agc.gain());
        assert!(out.iter().all(|s| s.is_finite() && s.abs() <= 1.0));
    }

    #[test]
    fn stereo_channels_share_one_gain() {
        let rate = 48_000;
        let left = speech_like(10 * rate as usize, rate, 0.003);
        let input: Vec<Sample> = left.iter().flat_map(|&l| [l, 0.5 * l]).collect();
        let mut agc = AutoGain::new(AudioFormat {
            sample_rate_hz: rate,
            channels: 2,
        });
        let out = run(&mut agc, &input, 960);
        assert!(agc.gain() > 2.0, "the gain must have risen: {}", agc.gain());
        for (o, i) in out.chunks(2).zip(input.chunks(2)) {
            if i[0].abs() > 1e-4 {
                let gain_l = o[0] / i[0];
                let gain_r = o[1] / i[1];
                assert!(
                    (gain_l - gain_r).abs() <= gain_l * 1e-5,
                    "{gain_l} vs {gain_r}"
                );
            }
        }
    }
}
