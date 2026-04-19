//! 3D-Speaker ERes2Net speaker embedding adapter.
//!
//! Loads the ONNX export of `iic/speech_eres2net_sv_en_voxceleb_16k`
//! (~26 MB, opset 13, [VoxCeleb-trained](https://www.modelscope.cn/models/iic/speech_eres2net_sv_en_voxceleb_16k))
//! and exposes it through the [`SpeakerEmbedder`] trait. The model
//! ingests Kaldi-style 80-bin log-mel filterbank features and emits a
//! 192-dimensional speaker embedding.
//!
//! ## Pipeline
//!
//! ```text
//!   raw 16 kHz f32 PCM
//!         │
//!         ▼
//!   Fbank::compute  ──► (T, 80) log-mel features
//!         │              with per-bin CMN already applied
//!         ▼
//!   pad / centre-crop to T = TARGET_FRAMES (~2 s)
//!         │
//!         ▼
//!   Tensor [1, T, 80] ──► tract ONNX inference ──► [1, 192]
//!         │
//!         ▼
//!   L2-normalise ──► Vec<f32>
//! ```
//!
//! ## Why a fixed time dimension?
//!
//! ERes2Net is a global-pooling network so it accepts arbitrary T. We
//! still pin a constant `target_frames` for two reasons:
//!
//! - **Tract optimisation.** With a concrete shape we can call
//!   `into_optimized()`; with a symbolic dim tract refuses to fold
//!   constants and inference is several × slower.
//! - **Latency budget.** A chunk of ~2 s is the diarisation sweet
//!   spot: long enough for a stable embedding, short enough to keep
//!   latency under one VAD utterance. Calling sites that pass shorter
//!   buffers get them silently zero-padded; longer buffers get
//!   centre-cropped.
//!
//! ## Pre-processing details (matching the exporter)
//!
//! `mel_spec::Fbank` defaults match the Kaldi parameters the model was
//! trained against:
//! `frame_length_ms = 25`, `frame_shift_ms = 10`, `num_mel_bins = 80`,
//! `preemphasis = 0.97`, povey window, `apply_cmn = true`. CMN
//! corresponds to the model's `feature_normalize_type = global-mean`
//! metadata: subtract the per-bin mean across the chunk's frames before
//! feeding the network. Sample-level normalisation
//! (`normalize_samples = 1`) is satisfied implicitly because cpal /
//! whisper hand us f32 PCM already in `[-1, 1]`.

use std::path::Path;
use std::sync::Arc;

use echo_domain::{DomainError, Sample};
use mel_spec::fbank::{Fbank, FbankConfig};
use tract_onnx::prelude::*;

use crate::embedding::{l2_normalize, SpeakerEmbedder};

/// Sample rate the ONNX export was exported with. Mixing rates is a bug.
pub const ERES2NET_SAMPLE_RATE: u32 = 16_000;

/// Output embedding dimensionality. Hard-coded by the model's last `Gemm`.
pub const ERES2NET_EMBED_DIM: usize = 192;

/// Number of mel filterbank bins the model expects per frame.
pub const ERES2NET_FBANK_DIM: usize = 80;

/// Default time dimension we pad / crop chunks to before inference.
/// 300 frames at the 10 ms hop is ~3.0 s of audio, which is the
/// shortest window where ERes2Net still produces stable, well-separated
/// embeddings on the supplied fixtures. Going lower (200 / 2.0 s)
/// roughly halves the same-speaker cosine similarity, going much higher
/// adds latency without measurable gain.
pub const ERES2NET_TARGET_FRAMES: usize = 300;

/// Minimum chunk length for which we'll attempt embedding. Below this
/// the resulting filterbank is too short to produce a meaningful
/// speaker fingerprint and we return `None`.
pub const ERES2NET_MIN_SAMPLES: usize = 8_000; // 0.5 s @ 16 kHz

/// Tunable knobs for [`Eres2NetEmbedder`]. Defaults match the constants
/// above and are calibrated against the supplied two-speaker fixture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Eres2NetConfig {
    /// Time dimension passed to the network. Fixed so tract can
    /// pre-optimise the graph.
    pub target_frames: usize,
    /// Minimum input samples below which `embed` returns `Ok(None)`.
    pub min_samples: usize,
}

impl Default for Eres2NetConfig {
    fn default() -> Self {
        Self {
            target_frames: ERES2NET_TARGET_FRAMES,
            min_samples: ERES2NET_MIN_SAMPLES,
        }
    }
}

/// 3D-Speaker ERes2Net embedder backed by tract.
///
/// The internal `tract` plan is `Send + Sync`, so the embedder can be
/// shared across tasks behind an `Arc`. The surface trait
/// (`SpeakerEmbedder::embed`) takes `&mut self` because future feature
/// extractors (e.g. ones that hold streaming state) will need it; this
/// implementation is internally stateless beyond the loaded model.
pub struct Eres2NetEmbedder {
    model: Arc<TypedRunnableModel<TypedModel>>,
    fbank: Fbank,
    config: Eres2NetConfig,
}

impl std::fmt::Debug for Eres2NetEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Eres2NetEmbedder")
            .field("config", &self.config)
            .field("model", &"<tract plan [1,T,80] -> [1,192]>")
            .finish()
    }
}

impl Eres2NetEmbedder {
    /// Load `path` and prepare the network for inference at the
    /// configured time dimension.
    ///
    /// Errors map to [`DomainError::ModelNotLoaded`] when the file is
    /// missing or unparseable, and to [`DomainError::DiarizationFailed`]
    /// when tract cannot specialise / optimise the graph.
    pub fn from_path(path: impl AsRef<Path>, config: Eres2NetConfig) -> Result<Self, DomainError> {
        let path = path.as_ref();
        let model = tract_onnx::onnx()
            .model_for_path(path)
            .map_err(|e| {
                DomainError::ModelNotLoaded(format!(
                    "cannot read ERes2Net ONNX at {}: {e}",
                    path.display()
                ))
            })?
            .with_input_fact(
                0,
                InferenceFact::dt_shape(
                    f32::datum_type(),
                    [1, config.target_frames, ERES2NET_FBANK_DIM],
                ),
            )
            .map_err(|e| DomainError::DiarizationFailed(format!("input fact: {e}")))?
            .into_optimized()
            .map_err(|e| DomainError::DiarizationFailed(format!("optimize: {e}")))?
            .into_runnable()
            .map_err(|e| DomainError::DiarizationFailed(format!("runnable: {e}")))?;

        let fbank = Fbank::new(FbankConfig {
            sample_rate: f64::from(ERES2NET_SAMPLE_RATE),
            num_mel_bins: ERES2NET_FBANK_DIM,
            ..FbankConfig::default()
        });

        Ok(Self {
            model: Arc::new(model),
            fbank,
            config,
        })
    }

    /// Convenience constructor with default configuration.
    pub fn new(path: impl AsRef<Path>) -> Result<Self, DomainError> {
        Self::from_path(path, Eres2NetConfig::default())
    }

    /// Effective configuration. Useful for tests and observability.
    #[must_use]
    pub fn config(&self) -> Eres2NetConfig {
        self.config
    }
}

impl SpeakerEmbedder for Eres2NetEmbedder {
    fn sample_rate_hz(&self) -> u32 {
        ERES2NET_SAMPLE_RATE
    }

    fn dim(&self) -> usize {
        ERES2NET_EMBED_DIM
    }

    fn embed(&mut self, samples: &[Sample]) -> Result<Option<Vec<f32>>, DomainError> {
        if samples.len() < self.config.min_samples {
            return Ok(None);
        }

        let fbank = self.fbank.compute(samples);
        let actual = fbank.shape()[0];
        if actual == 0 {
            return Ok(None);
        }

        // Pad-with-zero or centre-crop the (T, 80) fbank to the network's
        // fixed-size T = `target_frames` window. Inlined here because the
        // result is consumed once and we'd otherwise have to name the
        // underlying `ndarray::Array2` type in a helper signature.
        let target = self.config.target_frames;
        let mel = ERES2NET_FBANK_DIM;
        let mut flat = vec![0.0_f32; target * mel];
        if actual >= target {
            let start = (actual - target) / 2;
            for t in 0..target {
                for m in 0..mel {
                    flat[t * mel + m] = fbank[[start + t, m]];
                }
            }
        } else {
            for t in 0..actual {
                for m in 0..mel {
                    flat[t * mel + m] = fbank[[t, m]];
                }
            }
        }

        let input = Tensor::from_shape(&[1, self.config.target_frames, ERES2NET_FBANK_DIM], &flat)
            .map_err(|e| DomainError::DiarizationFailed(format!("input tensor: {e}")))?;

        let outputs = self
            .model
            .run(tvec!(input.into_tvalue()))
            .map_err(|e| DomainError::DiarizationFailed(format!("inference: {e}")))?;

        let view = outputs[0]
            .to_array_view::<f32>()
            .map_err(|e| DomainError::DiarizationFailed(format!("output dtype: {e}")))?;

        if view.len() != ERES2NET_EMBED_DIM {
            return Err(DomainError::DiarizationFailed(format!(
                "unexpected embedding length: {} (want {ERES2NET_EMBED_DIM})",
                view.len()
            )));
        }

        let mut embedding: Vec<f32> = view.iter().copied().collect();
        l2_normalize(&mut embedding);
        Ok(Some(embedding))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use crate::embedding::cosine_similarity;
    use echo_domain::Diarizer;

    /// Resolve the workspace-root model path, regardless of which
    /// crate cargo is running tests from.
    fn model_path() -> PathBuf {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        // crates/echo-diarize/.. → workspace root
        p.pop();
        p.pop();
        p.push("models/embedder/eres2net_en_voxceleb.onnx");
        p
    }

    /// `name` is resolved relative to the supplied fixture root, which
    /// is either this crate's `tests/fixtures` or the workspace-root
    /// `fixtures/audio` (TTS fixtures shared with the bench harness).
    fn load_wav(root: &str, name: &str) -> (Vec<f32>, u32) {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        if root == "shared" {
            // Resolve to <workspace>/fixtures/audio/<name>
            p.pop();
            p.pop();
            p.push("fixtures/audio");
        } else {
            p.push("tests/fixtures");
        }
        p.push(name);
        let mut reader = hound::WavReader::open(&p)
            .unwrap_or_else(|e| panic!("open {} failed: {e}", p.display()));
        let spec = reader.spec();
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => {
                let max = (1_i32 << (spec.bits_per_sample - 1)) as f32;
                reader
                    .samples::<i32>()
                    .map(|s| s.unwrap() as f32 / max)
                    .collect()
            }
            hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        };
        // Downmix to mono if the file came in stereo. The supplied
        // fixture is mono but be defensive for future fixtures.
        let mono = if spec.channels == 1 {
            samples
        } else {
            samples
                .chunks_exact(spec.channels as usize)
                .map(|c| c.iter().sum::<f32>() / c.len() as f32)
                .collect()
        };
        (mono, spec.sample_rate)
    }

    /// Resolve the workspace root (regardless of which crate cargo
    /// invoked tests from) and return its absolute path.
    fn workspace_root() -> PathBuf {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        // crates/echo-diarize → ../.. = workspace root
        p.pop();
        p.pop();
        p
    }

    /// Returns `Some(embedder)` when the ERes2Net ONNX model and the
    /// companion two-speaker WAV fixture are both present locally,
    /// `None` otherwise. We skip the test instead of failing it so
    /// that fresh checkouts (and CI environments without the model)
    /// stay green; the caller is expected to print a one-liner so
    /// the skip is visible.
    fn require_model() -> Option<Eres2NetEmbedder> {
        let model = model_path();
        let local_fixture =
            workspace_root().join("crates/echo-diarize/tests/fixtures/two_speakers_en.wav");
        // The TTS fixture comes from `scripts/build-fixtures.sh`. It's
        // shared with the bench harness and only the cross-speaker
        // test references it directly, but we gate on it here so the
        // skip message stays consistent.
        let tts_fixture = workspace_root().join("fixtures/audio/05_long_passage.wav");
        if !model.exists() || !local_fixture.exists() || !tts_fixture.exists() {
            eprintln!(
                "skipping ERes2Net test: missing assets \
                 (model={}, two_speakers_en.wav={}, 05_long_passage.wav={}). \
                 Run `scripts/download-models.sh embed` and \
                 `scripts/build-fixtures.sh` to fetch them.",
                model.exists(),
                local_fixture.exists(),
                tts_fixture.exists()
            );
            return None;
        }
        Some(Eres2NetEmbedder::new(&model).expect("load ERes2Net model"))
    }

    #[test]
    fn embedder_metadata_matches_constants() {
        if let Some(e) = require_model() {
            assert_eq!(e.sample_rate_hz(), ERES2NET_SAMPLE_RATE);
            assert_eq!(e.dim(), ERES2NET_EMBED_DIM);
            assert_eq!(e.config().target_frames, ERES2NET_TARGET_FRAMES);
        }
    }

    #[test]
    fn embed_rejects_too_short_chunks() {
        let Some(mut e) = require_model() else { return };
        let short = vec![0.0_f32; ERES2NET_MIN_SAMPLES - 1];
        let r = e.embed(&short).expect("embed shouldn't error");
        assert!(r.is_none(), "expected None for sub-min chunk");
    }

    #[test]
    fn embed_returns_unit_norm_vector() {
        let Some(mut e) = require_model() else { return };
        // 2 s of low-amplitude white-ish noise so the model has
        // *something* to embed without us shipping a fixture.
        let mut noise = Vec::with_capacity(2 * ERES2NET_SAMPLE_RATE as usize);
        let mut state: u32 = 0x1234_5678;
        for _ in 0..2 * ERES2NET_SAMPLE_RATE as usize {
            // xorshift32 → low-cost deterministic noise
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let v = (state as f32 / u32::MAX as f32) * 0.02 - 0.01;
            noise.push(v);
        }
        let emb = e
            .embed(&noise)
            .unwrap()
            .expect("embedding for 2 s of noise");
        assert_eq!(emb.len(), ERES2NET_EMBED_DIM);
        let norm = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "expected unit norm, got {norm}");
    }

    /// Sanity check the adapter against ground-truth speaker identity.
    ///
    /// We use the bench TTS fixtures (all rendered with the same macOS
    /// `say` voice — see `fixtures/README.md`) as a reliable
    /// "same-speaker" source, and a chunk from the human-recorded
    /// `two_speakers_en.wav` fixture as a "different-speaker" source.
    /// A working embedder must place same-speaker windows visibly
    /// closer in cosine space than cross-speaker windows.
    #[test]
    fn same_speaker_chunks_are_closer_than_cross_speaker_chunks() {
        let Some(mut e) = require_model() else { return };

        // Same-speaker windows: two non-overlapping ~3 s slices of a
        // single TTS recording.
        let (long, sr_long) = load_wav("shared", "05_long_passage.wav");
        assert_eq!(sr_long, 16_000);
        assert!(long.len() >= 9 * 16_000, "long fixture too short");
        let win = 3 * 16_000;
        let a1 = &long[0..win];
        let a2 = &long[(long.len() - win)..]; // last 3 s, same speaker

        // Cross-speaker window: a chunk from a different recording with
        // a human speaker.
        let (other, sr_other) = load_wav("local", "two_speakers_en.wav");
        assert_eq!(sr_other, 16_000);
        let b = &other[0..win];

        let ea1 = e.embed(a1).unwrap().unwrap();
        let ea2 = e.embed(a2).unwrap().unwrap();
        let eb = e.embed(b).unwrap().unwrap();

        let same = cosine_similarity(&ea1, &ea2);
        let cross_1 = cosine_similarity(&ea1, &eb);
        let cross_2 = cosine_similarity(&ea2, &eb);
        let ratio = same / cross_1.max(cross_2).max(1e-3);

        eprintln!("cosine same={same:.3} cross1={cross_1:.3} cross2={cross_2:.3} ratio={ratio:.2}");

        assert!(
            same > cross_1 && same > cross_2,
            "expected same-speaker similarity to dominate (same={same}, cross=({cross_1}, {cross_2}))"
        );
        // The absolute same-speaker score depends on chunk content;
        // the ratio against cross-speaker is the more meaningful
        // signal for downstream clustering.
        assert!(same > 0.45, "same-speaker cosine unexpectedly low: {same}");
        assert!(
            ratio >= 2.0,
            "expected ≥ 2x separation between same- and cross-speaker (got ratio={ratio:.2})"
        );
    }

    /// End-to-end through the [`OnlineDiarizer`]: fed a deterministic
    /// alternation of same- and cross-speaker windows, the diarizer
    /// must converge on exactly two speakers and assign matching
    /// chunks to the same id.
    #[test]
    fn online_diarizer_clusters_two_speakers_correctly() {
        let Some(embedder) = require_model() else {
            return;
        };

        let (long, _) = load_wav("shared", "05_long_passage.wav");
        let (other, _) = load_wav("local", "two_speakers_en.wav");
        let win = 3 * 16_000;
        // Speaker A samples drawn from non-overlapping slices of `long`.
        let a1 = long[0..win].to_vec();
        let a2 = long[win..(2 * win)].to_vec();
        let a3 = long[(long.len() - win)..].to_vec();
        // Speaker B samples drawn from the second half of the human
        // fixture. RMS analysis shows a clear silence at t≈7-8 s with
        // speaker A speaking before and speaker B after, so taking
        // both windows after t=9 s keeps us inside speaker B's turn.
        let sec = 16_000_usize;
        let b1 = other[(9 * sec)..(9 * sec + win)].to_vec();
        let b2 = other[(11 * sec)..(11 * sec + win)].to_vec();

        let mut diarizer = crate::OnlineDiarizer::with_defaults(Box::new(embedder));

        // Build a runtime that drives the async `assign` calls
        // without dragging tokio macros into this otherwise-sync test.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        let id_a1 = rt.block_on(diarizer.assign(&a1)).unwrap().unwrap();
        let id_b1 = rt.block_on(diarizer.assign(&b1)).unwrap().unwrap();
        let id_a2 = rt.block_on(diarizer.assign(&a2)).unwrap().unwrap();
        let id_b2 = rt.block_on(diarizer.assign(&b2)).unwrap().unwrap();
        let id_a3 = rt.block_on(diarizer.assign(&a3)).unwrap().unwrap();

        eprintln!("ids: a1={id_a1} b1={id_b1} a2={id_a2} b2={id_b2} a3={id_a3}");

        assert_eq!(id_a1, id_a2, "speaker A: a1 vs a2 must match");
        assert_eq!(id_a1, id_a3, "speaker A: a1 vs a3 must match");
        assert_eq!(id_b1, id_b2, "speaker B: b1 vs b2 must match");
        assert_ne!(id_a1, id_b1, "speakers A and B must differ");
        assert_eq!(diarizer.speakers().len(), 2);
    }
}
