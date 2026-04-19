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

    /// Reset the internal state to the silence baseline. Use this
    /// between independent sessions instead of allocating a fresh
    /// detector — adapters with heavy model state (Silero) will
    /// reuse loaded weights.
    fn reset(&mut self);
}
