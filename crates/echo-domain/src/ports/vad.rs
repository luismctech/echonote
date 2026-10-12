//! Voice Activity Detection port.
//!
//! A `Vad` answers "is the next slice of audio voiced?" so the
//! transcription pipeline can skip silent chunks before paying the
//! Whisper cost, and so the diarizer can build per-utterance segments
//! instead of clustering background noise.
//!
//! ## Stateful contract
//!
//! Implementations are *stateful by design*. Two reasons:
//!
//! 1. Energy-based VADs need hysteresis (consecutive-frame counters)
//!    to avoid flapping at threshold boundaries.
//! 2. Neural VADs (Silero) carry an LSTM hidden state across frames;
//!    the per-frame inference depends on what was seen before.
//!
//! Callers must therefore feed samples *chronologically* through a
//! single instance, and call [`Vad::reset`] before reusing one across
//! independent sessions.
//!
//! ## Format
//!
//! Every adapter declares the sample-rate it expects via
//! [`Vad::sample_rate_hz`]. Mixing rates is a bug — convert with
//! [`crate::Resampler`] first. All adapters consume mono `f32` PCM
//! in the `[-1.0, 1.0]` range, matching [`crate::Sample`].

use async_trait::async_trait;

use crate::{DomainError, Sample};

/// Voice / non-voice classification of an audio span.
///
/// Re-exported from the domain so `echo-app` use cases can branch on
/// the result without depending on the concrete VAD implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceState {
    /// Speech is likely present.
    Voiced,
    /// Below the speech threshold for long enough to count as silence.
    Silence,
}

/// Verdict of one analysis window, as reported by [`Vad::push_frames`].
///
/// Lets callers locate speech/silence transitions *inside* a push
/// (e.g. to cut transcription chunks at pauses) instead of only
/// learning one verdict per push.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VadFrame {
    /// Exclusive end of the window, as a sample offset into the slice
    /// passed to the push that produced it. Only the end is reported:
    /// a window may start in audio an adapter carried over from the
    /// previous push.
    pub end_sample: usize,
    /// Raw speech probability of the window in `[0.0, 1.0]`. Adapters
    /// without a probability report `1.0` for voiced, `0.0` otherwise.
    pub probability: f32,
    /// Detector state after this window, hysteresis included.
    pub state: VoiceState,
}

/// Voice Activity Detector.
///
/// Implementations: [`echo_audio::vad::EnergyVad`] (cheap, RMS-based,
/// good as a chunk-level gate) and [`echo_audio::vad::SileroVad`]
/// (neural, sharp boundaries, used by the diarizer).
///
/// Methods are `async` even when the underlying inference is sync so
/// that GPU-backed adapters or batching strategies can plug in later
/// without breaking the trait surface. CPU adapters can ignore the
/// async-ness — they return `Poll::Ready` immediately.
#[async_trait]
pub trait Vad: Send + Sync {
    /// Sample rate this instance expects on the input stream.
    fn sample_rate_hz(&self) -> u32;

    /// Push more samples through the detector and return the current
    /// classification. Internal state advances on every call.
    async fn push(&mut self, samples: &[Sample]) -> Result<VoiceState, DomainError>;

    /// Push more samples and return one [`VadFrame`] per analysis
    /// window completed by this call, in chronological order. Advances
    /// the same internal state as [`Vad::push`]; use one or the other
    /// for a given stream, not both on the same samples.
    ///
    /// The default implementation calls [`Vad::push`] and reports a
    /// single frame spanning the whole slice (no frame for an empty
    /// slice), so adapters without per-window resolution keep working.
    async fn push_frames(&mut self, samples: &[Sample]) -> Result<Vec<VadFrame>, DomainError> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        let state = self.push(samples).await?;
        let probability = if state == VoiceState::Voiced {
            1.0
        } else {
            0.0
        };
        Ok(vec![VadFrame {
            end_sample: samples.len(),
            probability,
            state,
        }])
    }

    /// Reset the internal state to the silence baseline. Use this
    /// between independent sessions instead of allocating a fresh
    /// detector — adapters with heavy model state (Silero) will
    /// reuse loaded weights.
    fn reset(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Adapter that only implements `push`, to exercise the default
    /// `push_frames`.
    struct FixedVad(VoiceState);

    #[async_trait]
    impl Vad for FixedVad {
        fn sample_rate_hz(&self) -> u32 {
            16_000
        }
        async fn push(&mut self, _samples: &[Sample]) -> Result<VoiceState, DomainError> {
            Ok(self.0)
        }
        fn reset(&mut self) {}
    }

    #[tokio::test]
    async fn default_push_frames_reports_one_frame_spanning_the_push() {
        let mut voiced = FixedVad(VoiceState::Voiced);
        let frames = voiced.push_frames(&[0.0; 1_600]).await.unwrap();
        assert_eq!(
            frames,
            vec![VadFrame {
                end_sample: 1_600,
                probability: 1.0,
                state: VoiceState::Voiced,
            }]
        );

        let mut silent = FixedVad(VoiceState::Silence);
        let frames = silent.push_frames(&[0.0; 320]).await.unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].end_sample, 320);
        assert_eq!(frames[0].probability, 0.0);
        assert_eq!(frames[0].state, VoiceState::Silence);
    }

    #[tokio::test]
    async fn default_push_frames_reports_nothing_for_an_empty_push() {
        let mut vad = FixedVad(VoiceState::Voiced);
        assert!(vad.push_frames(&[]).await.unwrap().is_empty());
    }
}
