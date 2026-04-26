//! Lightweight energy-based Voice Activity Detection.
//!
//! Strategy:
//!
//! 1. Slice the input into fixed-size frames (default 30 ms at 16 kHz).
//! 2. Compute RMS per frame.
//! 3. Apply hysteresis: switch from `Silence` to `Voiced` only when N
//!    consecutive frames exceed `start_threshold`; switch back when M
//!    consecutive frames fall below `end_threshold`. This kills
//!    flapping at thresholds.
//!
//! Good enough as a first-pass gate to skip pure-silence chunks before
//! invoking Whisper. A neural VAD (Silero) lands in Sprint 2 when
//! diarization needs sharper boundaries.

use async_trait::async_trait;
use echo_domain::{DomainError, Sample, Vad, VoiceState};

/// Tunable knobs for [`EnergyVad`]. Defaults are tuned for a desk mic
/// in a quiet-ish room (laptop coffeeshop / home office).
#[derive(Debug, Clone, Copy)]
pub struct VadConfig {
    /// Frame length, in milliseconds. 20–40 ms is the standard range.
    pub frame_ms: u32,
    /// RMS above which a frame counts as voiced. ~ -34 dBFS.
    pub start_threshold: f32,
    /// RMS below which a voiced run ends. Lower than `start_threshold`
    /// to add hysteresis. ~ -40 dBFS.
    pub end_threshold: f32,
    /// Consecutive voiced frames required to flip Silence → Voiced.
    pub start_frames: u8,
    /// Consecutive silent frames required to flip Voiced → Silence.
    pub end_frames: u8,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            frame_ms: 30,
            start_threshold: 0.02,
            end_threshold: 0.01,
            start_frames: 3, // ~90 ms — kills clicks and short bumps
            end_frames: 10,  // ~300 ms — keeps natural pauses inside an utterance
        }
    }
}

/// Hysteresis-based VAD. Stateful: feed it samples chronologically.
#[derive(Debug, Clone)]
pub struct EnergyVad {
    config: VadConfig,
    sample_rate_hz: u32,
    frame_len: usize,
    state: VoiceState,
    voiced_run: u8,
    silent_run: u8,
    /// Carry-over samples that did not fill a full frame on the last call.
    carry: Vec<Sample>,
}

impl EnergyVad {
    /// Build a VAD bound to `sample_rate_hz` (mono).
    #[must_use]
    pub fn new(sample_rate_hz: u32, config: VadConfig) -> Self {
        let frame_len = (sample_rate_hz as usize * config.frame_ms as usize) / 1_000;
        Self {
            config,
            sample_rate_hz,
            frame_len: frame_len.max(1),
            state: VoiceState::Silence,
            voiced_run: 0,
            silent_run: 0,
            carry: Vec::with_capacity(frame_len),
        }
    }

    /// Convenience constructor for the Whisper-canonical 16 kHz mono stream.
    #[must_use]
    pub fn for_whisper() -> Self {
        Self::new(16_000, VadConfig::default())
    }

    /// Current state. Updates only when [`Self::push`] is called.
    #[must_use]
    pub fn state(&self) -> VoiceState {
        self.state
    }

    /// Sample rate this instance was built for.
    #[must_use]
    pub fn sample_rate_hz(&self) -> u32 {
        self.sample_rate_hz
    }

    /// Frame size in samples used internally.
    #[must_use]
    pub fn frame_len(&self) -> usize {
        self.frame_len
    }

    /// Reset to silence; discards any carry-over.
    pub fn reset(&mut self) {
        self.state = VoiceState::Silence;
        self.voiced_run = 0;
        self.silent_run = 0;
        self.carry.clear();
    }

    /// Feed `samples` into the VAD; returns the new state.
    pub fn push(&mut self, samples: &[Sample]) -> VoiceState {
        let mut buf: Vec<Sample> = if self.carry.is_empty() {
            samples.to_vec()
        } else {
            let mut b = Vec::with_capacity(self.carry.len() + samples.len());
            b.append(&mut self.carry);
            b.extend_from_slice(samples);
            b
        };

        let mut cursor = 0;
        while cursor + self.frame_len <= buf.len() {
            let frame = &buf[cursor..cursor + self.frame_len];
            self.classify_frame(frame);
            cursor += self.frame_len;
        }

        // Carry the tail for next call.
        if cursor < buf.len() {
            self.carry = buf.split_off(cursor);
        }

        self.state
    }

    fn classify_frame(&mut self, frame: &[Sample]) {
        let rms = rms(frame);
        match self.state {
            VoiceState::Silence => {
                if rms >= self.config.start_threshold {
                    self.voiced_run = self.voiced_run.saturating_add(1);
                    if self.voiced_run >= self.config.start_frames {
                        self.state = VoiceState::Voiced;
                        self.silent_run = 0;
                    }
                } else {
                    self.voiced_run = 0;
                }
            }
            VoiceState::Voiced => {
                if rms <= self.config.end_threshold {
                    self.silent_run = self.silent_run.saturating_add(1);
                    if self.silent_run >= self.config.end_frames {
                        self.state = VoiceState::Silence;
                        self.voiced_run = 0;
                    }
                } else {
                    self.silent_run = 0;
                }
            }
        }
    }
}

#[async_trait]
impl Vad for EnergyVad {
    fn sample_rate_hz(&self) -> u32 {
        EnergyVad::sample_rate_hz(self)
    }

    async fn push(&mut self, samples: &[Sample]) -> Result<VoiceState, DomainError> {
        // RMS classification is infallible — there is no I/O, no model
        // to load, no shape mismatch. We surface the inherent state
        // change directly. Wrap in `Ok` to satisfy the port contract.
        Ok(EnergyVad::push(self, samples))
    }

    fn reset(&mut self) {
        EnergyVad::reset(self);
    }
}

/// Root mean square of a sample buffer. Returns `0.0` for an empty slice.
#[must_use]
pub fn rms(samples: &[Sample]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn sine(freq_hz: f32, duration_ms: u32, amplitude: f32) -> Vec<f32> {
        let n = (16_000 * duration_ms as usize) / 1_000;
        let mut v = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / 16_000.0;
            v.push(amplitude * (2.0 * std::f32::consts::PI * freq_hz * t).sin());
        }
        v
    }

    #[test]
    fn rms_of_silence_is_zero() {
        assert_eq!(rms(&[0.0_f32; 1_000]), 0.0);
    }

    #[test]
    fn rms_of_unit_sine_is_one_over_sqrt_two() {
        let s = sine(440.0, 1_000, 1.0);
        let r = rms(&s);
        let expected = std::f32::consts::FRAC_1_SQRT_2;
        assert!(
            (r - expected).abs() < 0.005,
            "expected ~{expected}, got {r}"
        );
    }

    #[test]
    fn pure_silence_stays_silent() {
        let mut vad = EnergyVad::for_whisper();
        let chunk = vec![0.0_f32; 16_000]; // 1 s of silence
        assert_eq!(vad.push(&chunk), VoiceState::Silence);
    }

    #[test]
    fn loud_tone_eventually_flips_to_voiced() {
        let mut vad = EnergyVad::for_whisper();
        let s = sine(440.0, 500, 0.5); // half-amplitude → RMS ~ 0.35, well above 0.02
        let final_state = vad.push(&s);
        assert_eq!(final_state, VoiceState::Voiced);
    }

    #[test]
    fn hysteresis_keeps_voiced_through_short_pause() {
        let mut vad = EnergyVad::for_whisper();
        // 500 ms tone → voiced
        vad.push(&sine(440.0, 500, 0.5));
        assert_eq!(vad.state(), VoiceState::Voiced);
        // 200 ms silence < end_frames * 30 ms = 300 ms → still voiced
        vad.push(&vec![0.0_f32; (16_000 * 200) / 1_000]);
        assert_eq!(vad.state(), VoiceState::Voiced);
        // Another 500 ms silence → now silent
        vad.push(&vec![0.0_f32; (16_000 * 500) / 1_000]);
        assert_eq!(vad.state(), VoiceState::Silence);
    }

    #[test]
    fn very_quiet_tone_below_start_threshold_does_not_flip() {
        let mut vad = EnergyVad::for_whisper();
        // amplitude 0.005 → RMS ~ 0.0035, below start_threshold 0.02
        let s = sine(440.0, 1_000, 0.005);
        assert_eq!(vad.push(&s), VoiceState::Silence);
    }

    #[test]
    fn small_pushes_accumulate_via_carry_buffer() {
        let mut vad = EnergyVad::for_whisper();
        let s = sine(440.0, 500, 0.5);
        // Push in tiny 50-sample chunks.
        let mut state = VoiceState::Silence;
        for chunk in s.chunks(50) {
            state = vad.push(chunk);
        }
        assert_eq!(state, VoiceState::Voiced);
    }
}
