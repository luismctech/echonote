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

    /// Start a stateful conversion of one continuous capture stream to
    /// [`AudioFormat::WHISPER`]. Use it when audio arrives in blocks: the
    /// filter keeps its history across [`ResampleStream::push`] calls, so
    /// block edges leave no artefacts and every sample is converted once.
    fn stream(&self) -> Box<dyn ResampleStream>;
}

/// Stateful conversion of a continuous stream of PCM blocks to
/// [`AudioFormat::WHISPER`], created by [`Resampler::stream`].
pub trait ResampleStream: Send {
    /// Convert the next block of interleaved `samples` in `input` format
    /// and append the 16 kHz mono result to `out`. Output may lag input
    /// by an internal block plus the filter's look-ahead (a few tens of
    /// milliseconds); [`Self::flush`] releases it.
    ///
    /// `input` may change between calls (a device reconnected with
    /// another configuration): the audio of the previous format is
    /// flushed first, so the output stays continuous and in time.
    fn push(
        &mut self,
        samples: &[Sample],
        input: AudioFormat,
        out: &mut Vec<Sample>,
    ) -> Result<(), DomainError>;

    /// Append the audio still held in the filter to `out` and start
    /// afresh. Call it at every discontinuity (pause, stop): after a
    /// flush, the output of a run holds exactly its input duration.
    fn flush(&mut self, out: &mut Vec<Sample>) -> Result<(), DomainError>;
}
