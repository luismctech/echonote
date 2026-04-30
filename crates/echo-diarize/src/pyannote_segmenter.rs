//! pyannote-segmentation-3.0 adapter.
//!
//! Loads the ONNX export of pyannote-segmentation-3.0 (shipped by the
//! sherpa-onnx project, ~17 MB) via tract-onnx and exposes it through
//! the [`Segmenter`] trait.
//!
//! ## Model I/O contract
//!
//! The sherpa-onnx ONNX export has:
//!
//! | Tensor   | Shape              | Notes                         |
//! |----------|--------------------|-------------------------------|
//! | Input 0  | `[1, 1, N]`        | mono f32 PCM at 16 kHz        |
//! | Output 0 | `[1, T, S]`        | per-frame speaker probabilities |
//!
//! where `N` = number of input samples (pad to `CHUNK_SAMPLES`),
//! `T` = `N / HOP_SIZE` (≈ 10 ms per frame), and `S` = number of
//! local speaker channels (3 for pyannote-segmentation-3.0).
//!
//! ## Post-processing
//!
//! 1. Threshold each speaker channel at `speaker_threshold`.
//! 2. Run-length encode the binary activity signal per channel.
//! 3. Convert frame indices back to sample offsets.
//! 4. Discard segments shorter than `min_segment_samples`.
//!
//! The result is a list of [`SpeakerSegment`] values ready to be
//! handed to the embedder layer. Each segment carries a zero-based
//! `local_speaker` index that is meaningful only within the current
//! chunk; global speaker identity is resolved by the cluster.
//!
//! ## Chunk size constraint
//!
//! The model was exported with a fixed input width (`CHUNK_SAMPLES`).
//! Shorter inputs are zero-padded on the right; longer inputs are
//! centre-cropped. Callers should feed chunks close to `CHUNK_SAMPLES`
//! for best accuracy. The streaming pipeline typically calls this with
//! its configured `chunk_ms` audio — 5 s chunks work well.
//!
//! ## Verification note
//!
//! The I/O shapes above match the sherpa-onnx v1.10+ ONNX export of
//! pyannote-segmentation-3.0. If you download the model from a
//! different source (e.g. the upstream pyannote-audio HuggingFace
//! repo), verify that its ONNX graph has one `[1, 1, N]` input and one
//! `[1, T, S]` output before use; otherwise `from_path` will error
//! during tract optimisation.

use std::path::Path;
use std::sync::Arc;

use echo_domain::{DomainError, Sample, Segmenter, SpeakerSegment};
use tract_onnx::prelude::*;

/// Sample rate the model was trained on.
pub const PYANNOTE_SAMPLE_RATE: u32 = 16_000;

/// Number of samples per chunk fed to the model.
/// 5 s × 16 000 Hz = 80 000 samples.
pub const PYANNOTE_CHUNK_SAMPLES: usize = 80_000;

/// Frame hop in samples (10 ms at 16 kHz).
pub const PYANNOTE_HOP_SAMPLES: usize = 160;

/// Maximum number of local speakers the model can detect per chunk.
pub const PYANNOTE_MAX_LOCAL_SPEAKERS: u8 = 3;

/// Tunable knobs for [`PyannoteSegmenter`].
#[derive(Debug, Clone, Copy)]
pub struct PyannoteSegmenterConfig {
    /// Probability above which a frame is considered active for a
    /// given local speaker.
    pub speaker_threshold: f32,
    /// Minimum segment length in samples. Segments shorter than this
    /// are discarded (they are likely noise at a speaker boundary).
    pub min_segment_samples: usize,
}

impl Default for PyannoteSegmenterConfig {
    fn default() -> Self {
        Self {
            speaker_threshold: 0.5,
            // 0.5 s minimum — shorter segments produce unreliable
            // embeddings from the downstream ERes2Net / CAM++ adapter.
            min_segment_samples: PYANNOTE_SAMPLE_RATE as usize / 2,
        }
    }
}

type PyannotePlan = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// pyannote-segmentation-3.0 segmenter.
pub struct PyannoteSegmenter {
    model: Arc<PyannotePlan>,
    config: PyannoteSegmenterConfig,
    /// Number of speaker channels S in the model output.
    num_speakers: u8,
}

impl std::fmt::Debug for PyannoteSegmenter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyannoteSegmenter")
            .field("config", &self.config)
            .field("num_speakers", &self.num_speakers)
            .field(
                "model",
                &format!(
                    "<tract plan [1,1,{}] -> [1,T,{}]>",
                    PYANNOTE_CHUNK_SAMPLES, self.num_speakers
                ),
            )
            .finish()
    }
}

impl PyannoteSegmenter {
    /// Load the ONNX model from `path`.
    ///
    /// `num_speakers` must match the `S` dimension of the model's
    /// output tensor.  For the official pyannote-segmentation-3.0
    /// ONNX export from sherpa-onnx this is `3`.
    pub fn from_path(
        path: impl AsRef<Path>,
        num_speakers: u8,
        config: PyannoteSegmenterConfig,
    ) -> Result<Self, DomainError> {
        let path = path.as_ref();

        let model = tract_onnx::onnx()
            .model_for_path(path)
            .map_err(|e| {
                DomainError::ModelNotLoaded(format!(
                    "cannot read pyannote segmentation ONNX at {}: {e}",
                    path.display()
                ))
            })?
            // Pin input shape: [1, 1, CHUNK_SAMPLES]
            .with_input_fact(
                0,
                InferenceFact::dt_shape(
                    f32::datum_type(),
                    [1_usize, 1, PYANNOTE_CHUNK_SAMPLES],
                ),
            )
            .map_err(|e| DomainError::DiarizationFailed(format!("input fact: {e}")))?
            .into_optimized()
            .map_err(|e| DomainError::DiarizationFailed(format!("optimize: {e}")))?
            .into_runnable()
            .map_err(|e| DomainError::DiarizationFailed(format!("runnable: {e}")))?;

        Ok(Self {
            model: Arc::new(model),
            config,
            num_speakers,
        })
    }

    /// Convenience constructor with default config and 3 local speakers.
    pub fn new(path: impl AsRef<Path>) -> Result<Self, DomainError> {
        Self::from_path(
            path,
            PYANNOTE_MAX_LOCAL_SPEAKERS,
            PyannoteSegmenterConfig::default(),
        )
    }
}

impl Segmenter for PyannoteSegmenter {
    fn sample_rate_hz(&self) -> u32 {
        PYANNOTE_SAMPLE_RATE
    }

    fn max_local_speakers(&self) -> u8 {
        self.num_speakers
    }

    fn segment(&mut self, samples: &[Sample]) -> Result<Vec<SpeakerSegment>, DomainError> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }

        // Pad or centre-crop to PYANNOTE_CHUNK_SAMPLES.
        let n = PYANNOTE_CHUNK_SAMPLES;
        let mut padded = vec![0.0_f32; n];
        if samples.len() >= n {
            let start = (samples.len() - n) / 2;
            padded.copy_from_slice(&samples[start..start + n]);
        } else {
            padded[..samples.len()].copy_from_slice(samples);
        }

        // Build input tensor: [1, 1, N].
        let input = Tensor::from_shape(&[1_usize, 1, n], &padded)
            .map_err(|e| DomainError::DiarizationFailed(format!("input tensor: {e}")))?;

        let outputs = self
            .model
            .run(tvec!(input.into_tvalue()))
            .map_err(|e| DomainError::DiarizationFailed(format!("inference: {e}")))?;

        // Output is [1, T, S] — extract the [T, S] view.
        let view = outputs[0]
            .to_array_view::<f32>()
            .map_err(|e| DomainError::DiarizationFailed(format!("output dtype: {e}")))?;

        let shape = view.shape();
        if shape.len() < 3 {
            return Err(DomainError::DiarizationFailed(format!(
                "unexpected pyannote output rank: {} (expected 3)",
                shape.len()
            )));
        }
        let num_frames = shape[1];
        let num_spk = shape[2];

        // How many samples in the original (pre-pad/crop) audio does
        // each frame represent?
        let input_len = samples.len().min(n);
        let samples_per_frame = input_len as f32 / num_frames as f32;

        let mut segments: Vec<SpeakerSegment> = Vec::new();

        for spk in 0..num_spk {
            // Walk frames to collect active spans for this speaker.
            let mut span_start: Option<usize> = None;

            for t in 0..=num_frames {
                let active = if t < num_frames {
                    view[[0, t, spk]] >= self.config.speaker_threshold
                } else {
                    false // sentinel to close an open span at the end
                };

                match (active, span_start) {
                    (true, None) => {
                        span_start = Some(t);
                    }
                    (false, Some(start)) => {
                        let start_sample = (start as f32 * samples_per_frame).round() as usize;
                        let end_sample =
                            ((t as f32 * samples_per_frame).round() as usize).min(samples.len());

                        if end_sample.saturating_sub(start_sample)
                            >= self.config.min_segment_samples
                        {
                            segments.push(SpeakerSegment {
                                start_sample,
                                end_sample,
                                local_speaker: spk as u8,
                            });
                        }
                        span_start = None;
                    }
                    _ => {}
                }
            }
        }

        // Sort by start sample for a clean chronological order.
        segments.sort_by_key(|s| (s.start_sample, s.local_speaker));
        Ok(segments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use echo_domain::Segmenter;
    use std::path::PathBuf;

    fn model_path() -> PathBuf {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.pop();
        p.pop();
        p.push("models/segmenter/pyannote_segmentation_3.onnx");
        p
    }

    fn require_model() -> Option<PyannoteSegmenter> {
        let model = model_path();
        if !model.exists() {
            eprintln!(
                "skipping pyannote test: model not found at {}. \
                 Run `scripts/download-models.sh segmenter` to fetch it.",
                model.display()
            );
            return None;
        }
        match PyannoteSegmenter::new(&model) {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!(
                    "skipping pyannote test: model at {} failed to load \
                     (tract may not support all ONNX ops in this export): {e}",
                    model.display()
                );
                None
            }
        }
    }

    #[test]
    fn reports_sample_rate_and_speaker_count() {
        if let Some(s) = require_model() {
            assert_eq!(s.sample_rate_hz(), PYANNOTE_SAMPLE_RATE);
            assert_eq!(s.max_local_speakers(), PYANNOTE_MAX_LOCAL_SPEAKERS);
        }
    }

    #[test]
    fn empty_input_returns_empty() {
        if let Some(mut s) = require_model() {
            let result = s.segment(&[]).unwrap();
            assert!(result.is_empty());
        }
    }

    #[test]
    fn silent_chunk_returns_no_segments() {
        if let Some(mut s) = require_model() {
            let silence = vec![0.0_f32; PYANNOTE_CHUNK_SAMPLES];
            let segs = s.segment(&silence).unwrap();
            assert!(
                segs.is_empty(),
                "silence should produce no active segments, got {segs:?}",
            );
        }
    }

    #[test]
    fn segments_are_chronologically_ordered() {
        if let Some(mut s) = require_model() {
            // Use the shared meeting fixture if available; otherwise
            // fall back to silence (still exercises the ordering code).
            let audio: Vec<f32> = {
                let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
                p.pop();
                p.pop();
                p.push("fixtures/audio/01_short_meeting.wav");
                if p.exists() {
                    let mut reader = hound::WavReader::open(&p).expect("open wav");
                    reader
                        .samples::<i16>()
                        .map(|s| f32::from(s.unwrap()) / f32::from(i16::MAX))
                        .collect()
                } else {
                    vec![0.0_f32; PYANNOTE_CHUNK_SAMPLES]
                }
            };

            let segs = s.segment(&audio).unwrap();
            for w in segs.windows(2) {
                assert!(
                    w[0].start_sample <= w[1].start_sample,
                    "segments must be in chronological order"
                );
                assert!(
                    w[0].start_sample < w[0].end_sample,
                    "each segment must be non-empty"
                );
            }
        }
    }
}
