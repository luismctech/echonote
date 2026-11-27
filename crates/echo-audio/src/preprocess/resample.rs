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
//! For audio arriving in blocks, [`StreamResampler`] keeps the filter
//! state across blocks so every sample is converted once and block
//! edges leave no artefacts.

use rubato::{
    Resampler as RubatoResampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};

use echo_domain::{AudioFormat, DomainError, ResampleStream, Resampler, Sample};

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
/// 16 kHz mono `f32`, time-aligned with the input and as long as it.
///
/// Returns the downmixed buffer unchanged when the input already runs
/// at 16 kHz.
pub fn resample_to_whisper(
    samples: &[Sample],
    format: AudioFormat,
) -> Result<Vec<Sample>, ResampleError> {
    let mut stream = StreamResampler::for_input(format, WHISPER_SAMPLE_RATE)?;
    let mut out = Vec::with_capacity(samples.len());
    stream.push(samples, format, &mut out)?;
    stream.flush(&mut out)?;
    Ok(out)
}

/// Equal-weight downmix. Stereo -> mean(L, R); 5.1 -> mean of all six
/// channels. Good enough for ASR; high-fidelity downmix lands later if
/// the LLM benchmark demands it.
pub(crate) fn downmix_to_mono(samples: &[Sample], channels: u16) -> Vec<Sample> {
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
pub(crate) const RESAMPLE_CHUNK_SIZE: usize = 1024;

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

/// Stateful conversion of one continuous stream of interleaved samples
/// to mono at a fixed target rate.
///
/// The sinc filter keeps its history between [`push`](Self::push)
/// calls, so a stream cut into blocks resamples exactly like the whole
/// signal at once. Its tail is drained by [`flush`](Self::flush), so
/// output sample `k` lines up with input time `k / target_hz` and a
/// flushed run holds `round(frames · target_hz / input_hz)` samples.
/// (`SincFixedIn` starts centred on the first input sample, so its
/// output is already aligned: dropping `output_delay()` samples would
/// shift it early.)
///
/// The input format may change between calls (a device reconnected with
/// another configuration): the audio of the old format is flushed first
/// and the filter is rebuilt for the new one.
pub struct StreamResampler {
    target_hz: u32,
    input: Option<AudioFormat>,
    /// `None` while the input already runs at `target_hz`.
    sinc: Option<SincFixedIn<Sample>>,
    /// Mono input waiting for a full [`RESAMPLE_CHUNK_SIZE`] block.
    pending: Vec<Sample>,
    /// Input frames and output samples of the current run.
    frames_in: u64,
    frames_out: u64,
}

impl StreamResampler {
    /// A stream converting to mono at `target_hz`; the filter is built
    /// on the first [`push`](Self::push).
    pub fn new(target_hz: u32) -> Self {
        Self {
            target_hz,
            input: None,
            sinc: None,
            pending: Vec::with_capacity(RESAMPLE_CHUNK_SIZE * 2),
            frames_in: 0,
            frames_out: 0,
        }
    }

    /// A stream already configured for `input`, so an unusable format
    /// fails here instead of on the first block.
    pub fn for_input(input: AudioFormat, target_hz: u32) -> Result<Self, ResampleError> {
        let mut stream = Self::new(target_hz);
        stream.configure(input)?;
        Ok(stream)
    }

    /// Convert the next block of interleaved `samples` in `input` format
    /// and append the mono result to `out`.
    pub fn push(
        &mut self,
        samples: &[Sample],
        input: AudioFormat,
        out: &mut Vec<Sample>,
    ) -> Result<(), ResampleError> {
        if self.input != Some(input) {
            self.flush(out)?;
            self.configure(input)?;
        }
        let mono = downmix_to_mono(samples, input.channels);
        self.frames_in += mono.len() as u64;
        if self.sinc.is_none() {
            self.frames_out += mono.len() as u64;
            out.extend_from_slice(&mono);
            return Ok(());
        }
        self.pending.extend_from_slice(&mono);
        while self.pending.len() >= RESAMPLE_CHUNK_SIZE {
            self.process_block(out, u64::MAX)?;
        }
        Ok(())
    }

    /// Append the audio still held in the filter to `out` and start a
    /// new run, so the run's output matches its input duration.
    pub fn flush(&mut self, out: &mut Vec<Sample>) -> Result<(), ResampleError> {
        if let (Some(input), true) = (self.input, self.sinc.is_some()) {
            let ratio = f64::from(self.target_hz) / f64::from(input.sample_rate_hz);
            let total = (self.frames_in as f64 * ratio).round() as u64;
            while self.frames_out < total {
                let progress = self.frames_out;
                self.pending.resize(RESAMPLE_CHUNK_SIZE, 0.0);
                self.process_block(out, total)?;
                if self.frames_out == progress {
                    break; // defensive: a block that yields nothing never will
                }
            }
        }
        self.restart();
        Ok(())
    }

    fn configure(&mut self, input: AudioFormat) -> Result<(), ResampleError> {
        if input.channels == 0 || input.sample_rate_hz == 0 || self.target_hz == 0 {
            return Err(ResampleError::InvalidFormat(input));
        }
        self.sinc = if input.sample_rate_hz == self.target_hz {
            None
        } else {
            Some(make_sinc_resampler(input.sample_rate_hz, self.target_hz)?)
        };
        self.input = Some(input);
        self.restart();
        Ok(())
    }

    fn restart(&mut self) {
        self.pending.clear();
        self.frames_in = 0;
        self.frames_out = 0;
        if let Some(sinc) = self.sinc.as_mut() {
            sinc.reset();
        }
    }

    /// Run one [`RESAMPLE_CHUNK_SIZE`] block of `pending` through the
    /// filter, stopping at `cap` output samples for the run.
    fn process_block(&mut self, out: &mut Vec<Sample>, cap: u64) -> Result<(), ResampleError> {
        let Some(sinc) = self.sinc.as_mut() else {
            return Ok(());
        };
        let block: Vec<Sample> = self.pending.drain(..RESAMPLE_CHUNK_SIZE).collect();
        let produced = sinc
            .process(&[block.as_slice()], None)
            .map_err(|e| ResampleError::Rubato(e.to_string()))?;
        let fresh = &produced[0];
        let room = usize::try_from(cap.saturating_sub(self.frames_out)).unwrap_or(usize::MAX);
        let fresh = &fresh[..fresh.len().min(room)];
        out.extend_from_slice(fresh);
        self.frames_out += fresh.len() as u64;
        Ok(())
    }
}

// ── Adapter for the domain port ─────────────────────────────────────

/// Satisfies the [`echo_domain::Resampler`] port with [`rubato`]'s sinc
/// filter: one-shot conversions for whole buffers, and a
/// [`StreamResampler`] per capture stream for audio arriving in blocks.
#[derive(Debug, Default, Clone, Copy)]
pub struct RubatoResamplerAdapter;

impl RubatoResamplerAdapter {
    pub fn new() -> Self {
        Self
    }
}

impl Resampler for RubatoResamplerAdapter {
    fn to_whisper(
        &self,
        samples: &[Sample],
        input: AudioFormat,
    ) -> Result<Vec<Sample>, DomainError> {
        Ok(resample_to_whisper(samples, input)?)
    }

    fn stream(&self) -> Box<dyn ResampleStream> {
        Box::new(WhisperStream(StreamResampler::new(WHISPER_SAMPLE_RATE)))
    }
}

/// [`StreamResampler`] fixed to 16 kHz, behind the domain port.
struct WhisperStream(StreamResampler);

impl ResampleStream for WhisperStream {
    fn push(
        &mut self,
        samples: &[Sample],
        input: AudioFormat,
        out: &mut Vec<Sample>,
    ) -> Result<(), DomainError> {
        Ok(self.0.push(samples, input, out)?)
    }

    fn flush(&mut self, out: &mut Vec<Sample>) -> Result<(), DomainError> {
        Ok(self.0.flush(out)?)
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

    /// `seconds` of a `hz` sine at `rate` mono, amplitude 0.5.
    fn tone(rate: u32, hz: f32, seconds: f32) -> Vec<Sample> {
        let n = (rate as f32 * seconds) as usize;
        (0..n)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * hz * i as f32 / rate as f32).sin())
            .collect()
    }

    /// Silence of `len` samples with a unit impulse at `at`.
    fn impulse(len: usize, at: usize) -> Vec<Sample> {
        let mut x = vec![0.0; len];
        x[at] = 1.0;
        x
    }

    /// Index of the strongest sample in `x[from..to]`.
    fn peak(x: &[Sample], from: usize, to: usize) -> usize {
        (from..to)
            .max_by(|&a, &b| x[a].abs().total_cmp(&x[b].abs()))
            .unwrap()
    }

    fn max_abs_diff(a: &[Sample], b: &[Sample]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    const MONO_48K: AudioFormat = AudioFormat {
        sample_rate_hz: 48_000,
        channels: 1,
    };

    #[test]
    fn one_shot_output_is_time_aligned_and_keeps_the_tail() {
        // An impulse half a second in lands half a second in.
        let out = resample_to_whisper(&impulse(48_000, 24_000), MONO_48K).unwrap();
        assert_eq!(out.len(), 16_000);
        let at = peak(&out, 0, out.len());
        assert!(at.abs_diff(8_000) <= 1, "impulse moved to {at}");

        // The last 5 ms of a tone are real audio, not dropped or padding.
        let out = resample_to_whisper(&tone(48_000, 440.0, 1.0), MONO_48K).unwrap();
        assert_eq!(out.len(), 16_000);
        let tail_rms = (out[out.len() - 80..].iter().map(|s| s * s).sum::<f32>() / 80.0).sqrt();
        assert!(tail_rms > 0.2, "tail lost: rms {tail_rms}");
    }

    #[test]
    fn stream_in_device_sized_blocks_matches_one_shot() {
        let input = tone(48_000, 440.0, 1.0);
        let whole = resample_to_whisper(&input, MONO_48K).unwrap();

        // ~10.7 ms cpal blocks, as the live pipeline receives them.
        let mut stream = StreamResampler::new(WHISPER_SAMPLE_RATE);
        let mut blocks = Vec::new();
        for block in input.chunks(512) {
            stream.push(block, MONO_48K, &mut blocks).unwrap();
        }
        stream.flush(&mut blocks).unwrap();

        assert_eq!(blocks.len(), whole.len());
        let diff = max_abs_diff(&blocks, &whole);
        assert!(
            diff < 1e-5,
            "block edges changed the output: max diff {diff}"
        );
    }

    #[test]
    fn stream_follows_a_format_change_without_losing_time() {
        // 0.5 s mono 48 kHz, then the device comes back as stereo 44.1 kHz;
        // each part has an impulse 0.25 s into it.
        let stereo_44k = AudioFormat {
            sample_rate_hz: 44_100,
            channels: 2,
        };
        let second: Vec<Sample> = impulse(22_050, 11_025)
            .into_iter()
            .flat_map(|s| [s, s])
            .collect();
        let mut stream = StreamResampler::new(WHISPER_SAMPLE_RATE);
        let mut out = Vec::new();
        stream
            .push(&impulse(24_000, 12_000), MONO_48K, &mut out)
            .unwrap();
        stream.push(&second, stereo_44k, &mut out).unwrap();
        stream.flush(&mut out).unwrap();

        // 1 s of audio in, 1 s out: neither part is sped up or lost.
        assert_eq!(out.len(), 16_000);
        let first = peak(&out, 0, 8_000);
        let second = peak(&out, 8_000, 16_000);
        assert!(first.abs_diff(4_000) <= 1, "first impulse at {first}");
        assert!(second.abs_diff(12_000) <= 1, "second impulse at {second}");
    }

    #[test]
    fn stream_passes_16k_through_after_downmix() {
        let stereo_16k = AudioFormat {
            sample_rate_hz: 16_000,
            channels: 2,
        };
        let mut stream = StreamResampler::new(WHISPER_SAMPLE_RATE);
        let mut out = Vec::new();
        stream
            .push(&[0.2, 0.4, -1.0, 1.0], stereo_16k, &mut out)
            .unwrap();
        stream.flush(&mut out).unwrap();
        assert_eq!(out, vec![0.3, 0.0]);
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
