//! Sample-rate conversion to the Whisper-canonical 16 kHz mono `f32`.
//!
//! Wraps [`rubato::SincFixedIn`] with a small façade that:
//!
//! 1. Downmixes any number of channels to mono before resampling
//!    (simple equal-weight average).
//! 2. Hides the chunked nature of `rubato` behind a single
//!    `resample_to_whisper(samples, format) -> Vec<f32>` call suitable
//!    for offline use (CLI `transcribe` subcommand, tests).
//! 3. Skips the resampler entirely when input is already 16 kHz mono.
//!
//! For real-time streaming during recording the resampler should be
//! reused frame-by-frame; that path lands alongside the streaming ASR
//! pipeline in Sprint 1.

use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

use echo_domain::{AudioFormat, DomainError, Sample};

/// Whisper expects 16 kHz mono `f32`.
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Errors specific to the resampler. Mapped to [`DomainError`] when
/// crossing the application boundary.
#[derive(Debug, thiserror::Error)]
pub enum ResampleError {
    /// Input format is invalid (zero channels or zero sample rate).
    #[error("invalid input format: {0:?}")]
    InvalidFormat(AudioFormat),
    /// Underlying rubato error.
    #[error("rubato: {0}")]
    Rubato(String),
}

impl From<ResampleError> for DomainError {
    fn from(value: ResampleError) -> Self {
        match value {
            ResampleError::InvalidFormat(f) => {
                DomainError::AudioFormatUnsupported(format!("invalid input format {f:?}"))
            }
            ResampleError::Rubato(msg) => {
                DomainError::AudioFormatUnsupported(format!("resample failed: {msg}"))
            }
        }
    }
}

/// One-shot resample of `samples` (interleaved if multi-channel) into
/// 16 kHz mono `f32`.
///
/// Returns the original buffer unchanged when the input is already in
/// the target format.
pub fn resample_to_whisper(
    samples: &[Sample],
    format: AudioFormat,
) -> Result<Vec<Sample>, ResampleError> {
    if format.channels == 0 || format.sample_rate_hz == 0 {
        return Err(ResampleError::InvalidFormat(format));
    }

    let mono = downmix_to_mono(samples, format.channels);

    if format.sample_rate_hz == WHISPER_SAMPLE_RATE {
        return Ok(mono);
    }

    resample_mono(&mono, format.sample_rate_hz, WHISPER_SAMPLE_RATE)
}

/// Equal-weight downmix. Stereo -> mean(L, R); 5.1 -> mean of all six
/// channels. Good enough for ASR; high-fidelity downmix lands later if
/// the LLM benchmark demands it.
fn downmix_to_mono(samples: &[Sample], channels: u16) -> Vec<Sample> {
    if channels <= 1 {
        return samples.to_vec();
    }
    let ch = channels as usize;
    let frames = samples.len() / ch;
    let mut out = Vec::with_capacity(frames);
    let inv = 1.0 / channels as f32;
    for i in 0..frames {
        let base = i * ch;
        let mut acc = 0.0f32;
        for c in 0..ch {
            acc += samples[base + c];
        }
        out.push(acc * inv);
    }
    out
}

/// Resample a mono buffer from `from_hz` to `to_hz` using rubato's
/// asynchronous sinc resampler with a balanced quality/speed preset.
fn resample_mono(
    samples: &[Sample],
    from_hz: u32,
    to_hz: u32,
) -> Result<Vec<Sample>, ResampleError> {
    if samples.is_empty() {
        return Ok(Vec::new());
    }

    // Balanced preset: 256-tap sinc, oversampling 256, Blackman-Harris
    // window. This gets ~96 dB SNR — well below Whisper's noise floor.
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let chunk_size: usize = 1024;
    let ratio = f64::from(to_hz) / f64::from(from_hz);

    let mut resampler = SincFixedIn::<f32>::new(ratio, 2.0, params, chunk_size, 1)
        .map_err(|e| ResampleError::Rubato(e.to_string()))?;

    // Pad the input with zeros so rubato can produce the very last
    // chunk. Output is trimmed to the exact expected length below.
    let total_in = samples.len();
    let expected_out = ((total_in as f64) * ratio).round() as usize;

    let mut input = Vec::with_capacity(total_in + chunk_size);
    input.extend_from_slice(samples);
    let pad = (chunk_size - (total_in % chunk_size)) % chunk_size;
    input.resize(total_in + pad, 0.0);

    let mut output: Vec<f32> = Vec::with_capacity(expected_out + chunk_size);
    let mut cursor = 0;
    while cursor + chunk_size <= input.len() {
        let in_frame = vec![&input[cursor..cursor + chunk_size]];
        let out_frame = resampler
            .process(&in_frame, None)
            .map_err(|e| ResampleError::Rubato(e.to_string()))?;
        output.extend_from_slice(&out_frame[0]);
        cursor += chunk_size;
    }

    output.truncate(expected_out);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn passthrough_when_already_in_target_format() {
        let samples = vec![0.1, -0.2, 0.3];
        let out = resample_to_whisper(&samples, AudioFormat::WHISPER).unwrap();
        assert_eq!(out, samples);
    }

    #[test]
    fn downmix_stereo_to_mono_equal_weight() {
        // 1 frame stereo: L=1.0, R=-1.0 -> mono=0.0
        let interleaved = vec![1.0_f32, -1.0, 0.5, 0.5, 0.2, 0.4];
        let mono = downmix_to_mono(&interleaved, 2);
        assert_eq!(mono, vec![0.0, 0.5, 0.3]);
    }

    #[test]
    fn downsamples_44100_to_16000_within_5_percent_of_expected_length() {
        // 1 second of 440 Hz at 44.1 kHz mono.
        let n = 44_100;
        let mut samples = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / 44_100.0;
            samples.push(0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin());
        }
        let fmt = AudioFormat {
            sample_rate_hz: 44_100,
            channels: 1,
        };
        let out = resample_to_whisper(&samples, fmt).unwrap();
        // Expected ~16000 samples; allow 5 % slack for the trim heuristic.
        let diff = (out.len() as i64 - 16_000_i64).unsigned_abs();
        assert!(
            diff <= 800,
            "resampled length {} too far from 16000",
            out.len()
        );
        // Energy should survive the resample (no DC, no clipping).
        let rms = (out.iter().map(|s| s * s).sum::<f32>() / out.len() as f32).sqrt();
        assert!(
            rms > 0.2 && rms < 0.5,
            "unexpected rms after resample: {rms}"
        );
    }

    #[test]
    fn downmix_then_resample_combination_works() {
        // 1 second of stereo @ 48 kHz: L = sine, R = silence. Should
        // come out as a halved-amplitude sine at 16 kHz mono.
        let frames = 48_000usize;
        let mut interleaved = Vec::with_capacity(frames * 2);
        for i in 0..frames {
            let t = i as f32 / 48_000.0;
            let s = 0.8 * (2.0 * std::f32::consts::PI * 220.0 * t).sin();
            interleaved.push(s);
            interleaved.push(0.0);
        }
        let fmt = AudioFormat {
            sample_rate_hz: 48_000,
            channels: 2,
        };
        let out = resample_to_whisper(&interleaved, fmt).unwrap();
        let diff = (out.len() as i64 - 16_000_i64).unsigned_abs();
        assert!(diff <= 800, "len={}", out.len());
    }

    #[test]
    fn invalid_format_returns_error() {
        let fmt = AudioFormat {
            sample_rate_hz: 0,
            channels: 1,
        };
        let err = resample_to_whisper(&[0.0; 100], fmt).unwrap_err();
        assert!(matches!(err, ResampleError::InvalidFormat(_)));
    }
}
