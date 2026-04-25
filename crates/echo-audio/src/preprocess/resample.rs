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

use std::sync::Mutex;

use rubato::{
    Resampler as RubatoResampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};

use echo_domain::{AudioFormat, DomainError, Resampler, Sample};

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

/// Internal chunk size for rubato processing (input frames per call).
const RESAMPLE_CHUNK_SIZE: usize = 1024;

/// Build a fresh [`SincFixedIn`] resampler for the given rate pair.
fn make_sinc_resampler(from_hz: u32, to_hz: u32) -> Result<SincFixedIn<f32>, ResampleError> {
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let ratio = f64::from(to_hz) / f64::from(from_hz);
    SincFixedIn::<f32>::new(ratio, 2.0, params, RESAMPLE_CHUNK_SIZE, 1)
        .map_err(|e| ResampleError::Rubato(e.to_string()))
}

/// Push `samples` through an already-constructed resampler.
fn process_mono(
    resampler: &mut SincFixedIn<f32>,
    samples: &[Sample],
    ratio: f64,
) -> Result<Vec<Sample>, ResampleError> {
    if samples.is_empty() {
        return Ok(Vec::new());
    }

    let total_in = samples.len();
    let expected_out = ((total_in as f64) * ratio).round() as usize;

    let mut input = Vec::with_capacity(total_in + RESAMPLE_CHUNK_SIZE);
    input.extend_from_slice(samples);
    let pad = (RESAMPLE_CHUNK_SIZE - (total_in % RESAMPLE_CHUNK_SIZE)) % RESAMPLE_CHUNK_SIZE;
    input.resize(total_in + pad, 0.0);

    let mut output: Vec<f32> = Vec::with_capacity(expected_out + RESAMPLE_CHUNK_SIZE);
    let mut cursor = 0;
    while cursor + RESAMPLE_CHUNK_SIZE <= input.len() {
        let in_frame = vec![&input[cursor..cursor + RESAMPLE_CHUNK_SIZE]];
        let out_frame = resampler
            .process(&in_frame, None)
            .map_err(|e| ResampleError::Rubato(e.to_string()))?;
        output.extend_from_slice(&out_frame[0]);
        cursor += RESAMPLE_CHUNK_SIZE;
    }

    output.truncate(expected_out);
    Ok(output)
}

/// Resample a mono buffer from `from_hz` to `to_hz` using rubato's
/// asynchronous sinc resampler with a balanced quality/speed preset.
///
/// Creates a fresh resampler each call — use [`RubatoResamplerAdapter`]
/// for the streaming path where the sinc kernel is cached.
fn resample_mono(
    samples: &[Sample],
    from_hz: u32,
    to_hz: u32,
) -> Result<Vec<Sample>, ResampleError> {
    let mut resampler = make_sinc_resampler(from_hz, to_hz)?;
    let ratio = f64::from(to_hz) / f64::from(from_hz);
    process_mono(&mut resampler, samples, ratio)
}

// ── Cached adapter for the streaming pipeline ───────────────────────

/// Cached sinc kernel + the rate pair it was built for.
struct CachedSinc {
    resampler: SincFixedIn<f32>,
    from_hz: u32,
    to_hz: u32,
}

/// Adapter that satisfies the [`echo_domain::Resampler`] port using
/// [`rubato::SincFixedIn`].  Caches the 256-tap sinc kernel across
/// calls so the expensive filter construction only happens once per
/// sample-rate pair (typically once per recording session).
pub struct RubatoResamplerAdapter {
    cached: Mutex<Option<CachedSinc>>,
}

impl RubatoResamplerAdapter {
    pub fn new() -> Self {
        Self {
            cached: Mutex::new(None),
        }
    }
}

impl Default for RubatoResamplerAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for RubatoResamplerAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RubatoResamplerAdapter")
            .finish_non_exhaustive()
    }
}

impl Resampler for RubatoResamplerAdapter {
    fn to_whisper(
        &self,
        samples: &[Sample],
        input: AudioFormat,
    ) -> Result<Vec<Sample>, DomainError> {
        if input.channels == 0 || input.sample_rate_hz == 0 {
            return Err(ResampleError::InvalidFormat(input).into());
        }

        let mono = downmix_to_mono(samples, input.channels);

        if input.sample_rate_hz == WHISPER_SAMPLE_RATE {
            return Ok(mono);
        }

        let from_hz = input.sample_rate_hz;
        let to_hz = WHISPER_SAMPLE_RATE;
        let ratio = f64::from(to_hz) / f64::from(from_hz);

        let mut guard = self.cached.lock().unwrap_or_else(|e| e.into_inner());

        // Reuse the cached kernel when the rate pair matches; otherwise
        // build a fresh one and stash it.
        let sinc = match guard.as_mut() {
            Some(c) if c.from_hz == from_hz && c.to_hz == to_hz => {
                c.resampler.reset();
                &mut c.resampler
            }
            _ => {
                let rs = make_sinc_resampler(from_hz, to_hz)
                    .map_err(DomainError::from)?;
                *guard = Some(CachedSinc {
                    resampler: rs,
                    from_hz,
                    to_hz,
                });
                &mut guard.as_mut().unwrap().resampler
            }
        };

        Ok(process_mono(sinc, &mono, ratio)?)
    }
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
