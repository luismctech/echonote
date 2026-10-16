//! Conversion from cpal's native sample formats to the pipeline's `f32`.
//!
//! cpal 0.18 ranks default configs F32 > F64 > descending integer widths
//! (I32 > I24 > I16), so `default_input_config()` can now hand us
//! formats the capture adapters used to reject. Every cpal-backed
//! adapter builds its stream through [`build_input_stream`], which
//! accepts all formats in [`SUPPORTED_FORMATS`] and converts them here.
//!
//! Signed integers are divided by their type's positive maximum (so the
//! full-scale positive sample maps to exactly `1.0`); `u16` is
//! re-centred on its midpoint. Both match the conversions the adapters
//! used before cpal 0.18, so existing I16/U16 captures are unchanged.

use std::time::Duration;

use cpal::traits::DeviceTrait;
use cpal::{Device, SampleFormat, SizedSample, StreamConfig, I24};
use echo_domain::{DomainError, Sample};

/// Sample formats [`build_input_stream`] can capture, best first.
pub const SUPPORTED_FORMATS: &[SampleFormat] = &[
    SampleFormat::F32,
    SampleFormat::F64,
    SampleFormat::I32,
    SampleFormat::I24,
    SampleFormat::I16,
    SampleFormat::U16,
];

/// Preference rank of a sample format when choosing among supported
/// configs: lower is better, and formats outside [`SUPPORTED_FORMATS`]
/// sort last.
#[must_use]
pub fn rank_format(format: SampleFormat) -> usize {
    SUPPORTED_FORMATS
        .iter()
        .position(|&f| f == format)
        .unwrap_or(SUPPORTED_FORMATS.len())
}

/// A cpal sample type the pipeline can ingest.
pub trait PipelineSample: SizedSample + Send + 'static {
    /// Convert to the pipeline's normalised `f32` (nominally `-1.0..=1.0`).
    fn to_pipeline(self) -> Sample;
}

impl PipelineSample for f32 {
    fn to_pipeline(self) -> Sample {
        self
    }
}

impl PipelineSample for f64 {
    #[allow(clippy::cast_possible_truncation)]
    fn to_pipeline(self) -> Sample {
        self as f32
    }
}

impl PipelineSample for i16 {
    fn to_pipeline(self) -> Sample {
        f32::from(self) / f32::from(i16::MAX)
    }
}

impl PipelineSample for u16 {
    fn to_pipeline(self) -> Sample {
        (f32::from(self) - 32_768.0) / 32_768.0
    }
}

impl PipelineSample for i32 {
    #[allow(clippy::cast_possible_truncation)]
    fn to_pipeline(self) -> Sample {
        // Divide in f64: f32 cannot represent i32::MAX exactly.
        (f64::from(self) / f64::from(i32::MAX)) as f32
    }
}

impl PipelineSample for I24 {
    #[allow(clippy::cast_precision_loss)]
    fn to_pipeline(self) -> Sample {
        // 24-bit values fit exactly in f32's 24-bit mantissa.
        const I24_MAX: f32 = 8_388_607.0;
        self.inner() as f32 / I24_MAX
    }
}

/// Build an input stream for `sample_format`, converting every callback
/// buffer to pipeline samples before handing it to `on_samples`.
///
/// # Errors
///
/// [`DomainError::AudioFormatUnsupported`] for formats outside
/// [`SUPPORTED_FORMATS`]; [`DomainError::AudioCaptureFailed`] when cpal
/// cannot build the stream.
pub fn build_input_stream<F, E>(
    device: &Device,
    config: StreamConfig,
    sample_format: SampleFormat,
    on_samples: F,
    on_error: E,
    timeout: Option<Duration>,
) -> Result<cpal::Stream, DomainError>
where
    F: FnMut(Vec<Sample>) + Send + 'static,
    E: FnMut(cpal::Error) + Send + 'static,
{
    let built = match sample_format {
        SampleFormat::F32 => build::<f32, _, _>(device, config, on_samples, on_error, timeout),
        SampleFormat::F64 => build::<f64, _, _>(device, config, on_samples, on_error, timeout),
        SampleFormat::I32 => build::<i32, _, _>(device, config, on_samples, on_error, timeout),
        SampleFormat::I24 => build::<I24, _, _>(device, config, on_samples, on_error, timeout),
        SampleFormat::I16 => build::<i16, _, _>(device, config, on_samples, on_error, timeout),
        SampleFormat::U16 => build::<u16, _, _>(device, config, on_samples, on_error, timeout),
        other => {
            return Err(DomainError::AudioFormatUnsupported(format!(
                "unsupported sample format: {other:?}"
            )));
        }
    };
    built.map_err(|e| DomainError::AudioCaptureFailed(format!("build {sample_format} stream: {e}")))
}

fn build<T, F, E>(
    device: &Device,
    config: StreamConfig,
    mut on_samples: F,
    on_error: E,
    timeout: Option<Duration>,
) -> Result<cpal::Stream, cpal::Error>
where
    T: PipelineSample,
    F: FnMut(Vec<Sample>) + Send + 'static,
    E: FnMut(cpal::Error) + Send + 'static,
{
    device.build_input_stream(
        config,
        move |data: &[T], _info: &cpal::InputCallbackInfo| {
            on_samples(convert(data));
        },
        on_error,
        timeout,
    )
}

/// Convert one callback buffer to pipeline samples.
#[must_use]
pub fn convert<T: PipelineSample>(data: &[T]) -> Vec<Sample> {
    data.iter().map(|&s| s.to_pipeline()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: Sample, expected: Sample) {
        assert!(
            (actual - expected).abs() < 1e-4,
            "expected {expected}, got {actual}"
        );
    }

    fn assert_full_scale<T: PipelineSample>(max: T, min: T, zero: T) {
        let out = convert(&[max, min, zero]);
        assert_close(out[0], 1.0);
        assert_close(out[1], -1.0);
        assert_close(out[2], 0.0);
    }

    #[test]
    fn f32_passes_through() {
        assert_eq!(
            convert(&[1.0_f32, -1.0, 0.0, 0.25]),
            vec![1.0, -1.0, 0.0, 0.25]
        );
    }

    #[test]
    fn f64_narrows_to_f32() {
        assert_full_scale(1.0_f64, -1.0, 0.0);
        assert_close(convert(&[0.5_f64])[0], 0.5);
    }

    #[test]
    fn i32_full_scale() {
        assert_full_scale(i32::MAX, i32::MIN, 0);
        assert_eq!(convert(&[i32::MAX])[0], 1.0);
    }

    #[test]
    fn i24_full_scale() {
        let max = I24::new(8_388_607).expect("in range");
        let min = I24::new(-8_388_608).expect("in range");
        let zero = I24::new(0).expect("in range");
        assert_full_scale(max, min, zero);
        assert_eq!(convert(&[max])[0], 1.0);
    }

    #[test]
    fn i16_full_scale_is_unchanged() {
        assert_full_scale(i16::MAX, i16::MIN, 0);
        assert_eq!(convert(&[i16::MAX])[0], 1.0);
    }

    #[test]
    fn u16_recentres_on_midpoint() {
        assert_full_scale(u16::MAX, u16::MIN, 32_768);
        assert_eq!(convert(&[u16::MIN, 32_768])[..], [-1.0, 0.0]);
    }

    #[test]
    fn integer_formats_rank_by_descending_width_after_floats() {
        let mut formats = vec![
            SampleFormat::U16,
            SampleFormat::I16,
            SampleFormat::I24,
            SampleFormat::I32,
            SampleFormat::F64,
            SampleFormat::F32,
            SampleFormat::U8,
        ];
        formats.sort_by_key(|&f| rank_format(f));
        assert_eq!(
            formats,
            vec![
                SampleFormat::F32,
                SampleFormat::F64,
                SampleFormat::I32,
                SampleFormat::I24,
                SampleFormat::I16,
                SampleFormat::U16,
                SampleFormat::U8,
            ]
        );
    }
}
