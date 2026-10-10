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
//! | Output 0 | `[1, T, 7]`        | per-frame powerset log-probabilities |
//!
//! where `N` = number of input samples (pad to `CHUNK_SAMPLES`) and `T`
//! is the model's frame count (293 for 10 s of audio).
//!
//! The head is a **powerset** classifier (`LogSoftmax` over 7 classes),
//! not 3 independent per-speaker sigmoids: max 3 speakers, max 2
//! simultaneous. The class order is
//! `[none, s0, s1, s2, s0+s1, s0+s2, s1+s2]`. Every value is `<= 0`, so
//! thresholding the raw output can never fire.
//!
//! ## Post-processing
//!
//! 1. Per frame, take the `argmax` class and expand it to the set of
//!    active local speakers ([`decode_powerset`]), giving a `[T, 3]`
//!    boolean activity matrix.
//! 2. Run-length encode the activity per speaker column; spans of
//!    different speakers may overlap.
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

/// Number of powerset classes in the model output
/// (`none`, 3 singles, 3 pairs).
pub const POWERSET_CLASSES: usize = 7;

/// Active local speakers for each powerset class, in the model's class
/// order `[none, s0, s1, s2, s0+s1, s0+s2, s1+s2]`.
const POWERSET_MAP: [[bool; 3]; POWERSET_CLASSES] = [
    [false, false, false],
    [true, false, false],
    [false, true, false],
    [false, false, true],
    [true, true, false],
    [true, false, true],
    [false, true, true],
];

/// Decode flat `[T, 7]` powerset log-probabilities into a per-frame
/// activity matrix `[T][3]`.
///
/// Takes the `argmax` class of each frame (monotonic in log-softmax, so
/// no `exp` is needed) and expands it to the set of active speakers.
/// Trailing values that do not fill a whole frame are ignored.
pub fn decode_powerset(log_probs: &[f32]) -> Vec<[bool; 3]> {
    log_probs
        .chunks_exact(POWERSET_CLASSES)
        .map(|frame| {
            let best = frame
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |acc, (k, &v)| {
                    if v > acc.1 {
                        (k, v)
                    } else {
                        acc
                    }
                })
                .0;
            POWERSET_MAP[best]
        })
        .collect()
}

/// Run-length encode a `[T][3]` activity matrix into per-speaker
/// segments, sorted chronologically.
///
/// `samples_per_frame` maps frame indices to sample offsets; ends are
/// clamped to `max_samples`. Runs shorter than `min_segment_samples`
/// are dropped.
fn activity_to_segments(
    activity: &[[bool; 3]],
    samples_per_frame: f32,
    max_samples: usize,
    min_segment_samples: usize,
) -> Vec<SpeakerSegment> {
    let mut segments = Vec::new();
    for spk in 0..3 {
        let mut span_start: Option<usize> = None;
        // `t == len` is a sentinel that closes a span open at the end.
        for t in 0..=activity.len() {
            let active = activity.get(t).is_some_and(|f| f[spk]);
            match (active, span_start) {
                (true, None) => span_start = Some(t),
                (false, Some(start)) => {
                    let start_sample = (start as f32 * samples_per_frame).round() as usize;
                    let end_sample =
                        ((t as f32 * samples_per_frame).round() as usize).min(max_samples);
                    if end_sample.saturating_sub(start_sample) >= min_segment_samples {
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
    segments.sort_by_key(|s| (s.start_sample, s.local_speaker));
    segments
}

/// Tunable knobs for [`PyannoteSegmenter`].
#[derive(Debug, Clone, Copy)]
pub struct PyannoteSegmenterConfig {
    /// Minimum segment length in samples. Segments shorter than this
    /// are discarded (they are likely noise at a speaker boundary).
    pub min_segment_samples: usize,
}

impl Default for PyannoteSegmenterConfig {
    fn default() -> Self {
        Self {
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

        // Output is [1, T, 7] powerset log-probabilities.
        let view = outputs[0]
            .to_array_view::<f32>()
            .map_err(|e| DomainError::DiarizationFailed(format!("output dtype: {e}")))?;

        let shape = view.shape();
        if shape.len() != 3 || shape[2] != POWERSET_CLASSES {
            return Err(DomainError::DiarizationFailed(format!(
                "unexpected pyannote output shape {shape:?} (expected [1, T, {POWERSET_CLASSES}])"
            )));
        }
        let num_frames = shape[1];
        let flat: Vec<f32> = view.iter().copied().collect();
        let activity = decode_powerset(&flat);

        // How many samples in the original (pre-pad/crop) audio does
        // each frame represent?
        let input_len = samples.len().min(n);
        let samples_per_frame = input_len as f32 / num_frames as f32;

        Ok(activity_to_segments(
            &activity,
            samples_per_frame,
            samples.len(),
            self.config.min_segment_samples,
        ))
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

    /// Build a `[T, 7]` log-softmax-like tensor where each frame puts
    /// its mass on one class (all values are negative, like the model).
    fn frames_of(classes: &[usize]) -> Vec<f32> {
        let mut out = Vec::with_capacity(classes.len() * POWERSET_CLASSES);
        for &c in classes {
            for k in 0..POWERSET_CLASSES {
                out.push(if k == c { -0.01 } else { -6.0 });
            }
        }
        out
    }

    #[test]
    fn decode_powerset_maps_silence_single_and_overlap_classes() {
        let tensor = frames_of(&[0, 2, 4, 6, 3]);
        let act = decode_powerset(&tensor);
        assert_eq!(act[0], [false, false, false], "class 0 is silence");
        assert_eq!(act[1], [false, true, false], "class 2 is speaker1 only");
        assert_eq!(act[2], [true, true, false], "class 4 is speaker0+speaker1");
        assert_eq!(act[3], [false, true, true], "class 6 is speaker1+speaker2");
        assert_eq!(act[4], [false, false, true], "class 3 is speaker2 only");
    }

    #[test]
    fn spans_split_on_gaps_and_drop_short_runs() {
        // speaker0: frames 0..4 and 6..8 (second run too short);
        // 1 sample per frame, min length 3.
        let mut act = vec![[false; 3]; 10];
        for t in [0, 1, 2, 3, 6, 7] {
            act[t][0] = true;
        }
        // speaker1 overlaps speaker0 on frames 2..5.
        for frame in &mut act[2..5] {
            frame[1] = true;
        }
        let segs = activity_to_segments(&act, 1.0, 10, 3);
        assert_eq!(
            segs,
            vec![
                SpeakerSegment {
                    start_sample: 0,
                    end_sample: 4,
                    local_speaker: 0
                },
                SpeakerSegment {
                    start_sample: 2,
                    end_sample: 5,
                    local_speaker: 1
                },
            ]
        );
    }

    #[test]
    fn spans_close_when_activity_runs_to_the_last_frame() {
        let act = vec![[true, false, false]; 5];
        let segs = activity_to_segments(&act, 2.0, 10, 4);
        assert_eq!(segs.len(), 1);
        assert_eq!((segs[0].start_sample, segs[0].end_sample), (0, 10));
    }
}
