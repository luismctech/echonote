//! Sample-rate conversion port.
//!
//! Application-layer use cases need to feed the [`Transcriber`] with
//! samples in [`AudioFormat::WHISPER`] regardless of what the
//! microphone delivered. Doing the conversion through a port keeps the
//! application crate free of dependencies on `rubato` (or any other
//! resampling library).
//!
//! [`Transcriber`]: crate::ports::transcriber::Transcriber

use crate::ports::audio::{AudioFormat, Sample};
use crate::DomainError;

/// Anything that can convert a buffer of PCM samples to a target
/// [`AudioFormat`]. Implementations must accept multi-channel
/// interleaved input and downmix to mono when the target requests it.
pub trait Resampler: Send + Sync {
    /// Convert `samples` from `input` format to the
    /// [`AudioFormat::WHISPER`] canonical format (16 kHz mono `f32`).
    ///
    /// Implementations MAY short-circuit and return the original
    /// samples (or a clone) when `input == AudioFormat::WHISPER`.
    fn to_whisper(
        &self,
        samples: &[Sample],
        input: AudioFormat,
    ) -> Result<Vec<Sample>, DomainError>;
}
