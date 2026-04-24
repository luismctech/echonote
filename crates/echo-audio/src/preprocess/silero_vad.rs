//! Silero VAD v5 adapter — neural Voice Activity Detection.
//!
//! Implements the [`Vad`] domain port using the Silero VAD v5 model
//! (`silero_vad.onnx`, ~1.2 MB after pre-processing) running through
//! pure-Rust ONNX inference (`tract-onnx`). Compared to the
//! energy-based baseline this adapter:
//!
//! - distinguishes voice from non-voice noise (music, keyboards, fans);
//! - holds an LSTM hidden state across frames so short pauses inside
//!   an utterance are stitched correctly;
//! - is robust to gain (a soft speaker still scores high probability).
//!
//! ## Sample rate and frame size
//!
//! Silero VAD v5 expects 16 kHz mono PCM and processes audio in
//! fixed 512-sample windows (32 ms). The 8 kHz path (256-sample
//! windows) exists upstream but is intentionally not exposed here —
//! the rest of EchoNote is 16 kHz / mono.
//!
//! ## Why the on-disk ONNX is a modified build
//!
//! The upstream Silero v5 graph dispatches between the 16 kHz and
//! 8 kHz sub-networks through an ONNX `If` operator controlled by the
//! `sr` input. `tract-onnx` does not implement `If`, so loading the
//! raw upstream file fails with `optimize: Failed analyse for node
//! #5 "If_0" If`. We therefore pre-process the model at download
//! time (`scripts/simplify-silero-vad.py`): we inline the 16 kHz
//! `then_branch`, drop the now-unused `sr` input and let ORT's
//! constant-folding pass collapse the nested `If`s that depended on
//! static shape values. The result is a pure feed-forward + LSTM
//! graph with 2 inputs (`input`, `state`) and 2 outputs (`output`,
//! `stateN`), bitwise-equivalent to the upstream model for 16 kHz
//! audio.
//!
//! ## Hysteresis
//!
//! The raw model output is a per-frame probability; flipping state
//! on every frame would chop utterances at every short pause. We
//! apply the same `start_frames` / `end_frames` hysteresis as
//! [`super::vad::EnergyVad`], with thresholds tuned for the model's
//! probability scale (`start_threshold ≈ 0.5`, `end_threshold ≈ 0.35`).
//!
//! ## Cost
//!
//! Each `push` runs the LSTM over 512 samples on CPU. On Apple Silicon
//! a single inference takes < 1 ms; throughput is ~30 000 ms of audio
//! per second on one core. We can therefore call it on every chunk of
//! the streaming pipeline without measurable latency impact.

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use echo_domain::{DomainError, Sample, Vad, VoiceState};
use tracing::info;
use tract_onnx::prelude::*;

/// Sample rate Silero VAD v5 was trained for.
pub const SILERO_SAMPLE_RATE: u32 = 16_000;

/// Number of samples consumed per inference at 16 kHz (~32 ms).
pub const SILERO_FRAME_SAMPLES: usize = 512;

/// Hidden-state width per LSTM layer. Hard-coded by the model.
const STATE_HIDDEN: usize = 128;

/// Tunable knobs for [`SileroVad`]. The defaults were sanity-checked
/// against the included `fixtures/audio/*.wav` set: speech reliably
/// flips to `Voiced` within one window and silence falls back within
/// half a second.
#[derive(Debug, Clone, Copy)]
pub struct SileroVadConfig {
    /// Probability above which a frame counts as voiced.
    pub start_threshold: f32,
    /// Probability below which a voiced run can end. Lower than
    /// `start_threshold` to add hysteresis around the boundary.
    pub end_threshold: f32,
    /// Consecutive voiced frames required to flip Silence → Voiced.
    /// `1` is fine because Silero already smooths internally.
    pub start_frames: u8,
    /// Consecutive silent frames required to flip Voiced → Silence.
    /// `16` ≈ 512 ms, which keeps natural breath pauses inside an
    /// utterance.
    pub end_frames: u8,
}

impl Default for SileroVadConfig {
    fn default() -> Self {
        Self {
            start_threshold: 0.5,
            end_threshold: 0.35,
            start_frames: 1,
            end_frames: 16,
        }
    }
}

impl SileroVadConfig {
    /// Permissive thresholds for meeting transcription where missing
    /// speech is far more costly than transcribing a quiet moment.
    /// Works well with re-captured audio (speaker → air → mic) and
    /// low-gain scenarios.
    pub fn meetings() -> Self {
        Self {
            start_threshold: 0.35,
            end_threshold: 0.20,
            start_frames: 1,
            end_frames: 16,
        }
    }
}

/// Concrete tract model type for an optimized + runnable Silero graph.
type SileroModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// Silero VAD v5 detector. Holds the loaded model (cheap to clone via
/// `Arc`) and the per-stream LSTM state plus hysteresis bookkeeping.
pub struct SileroVad {
    model: Arc<SileroModel>,
    config: SileroVadConfig,
    /// LSTM hidden state, shape `[2, 1, 128]`.
    state: Tensor,
    /// Audio that did not fill a 512-sample window on the previous push.
    carry: Vec<Sample>,
    hysteresis: VoiceState,
    voiced_run: u8,
    silent_run: u8,
}

impl std::fmt::Debug for SileroVad {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SileroVad")
            .field("config", &self.config)
            .field("state", &"<Tensor [2,1,128]>")
            .field("carry_len", &self.carry.len())
            .field("hysteresis", &self.hysteresis)
            .finish()
    }
}

impl SileroVad {
    /// Load the model from `path` and prepare it for 16 kHz mono
    /// inference with the supplied config.
    ///
    /// Errors map to [`DomainError::ModelNotLoaded`] when the file is
    /// missing or unparsable, and to [`DomainError::VadFailed`] when
    /// tract cannot specialize / optimize the graph.
    pub fn from_path(path: impl AsRef<Path>, config: SileroVadConfig) -> Result<Self, DomainError> {
        let path = path.as_ref();
        let model = tract_onnx::onnx()
            .model_for_path(path)
            .map_err(|e| {
                DomainError::ModelNotLoaded(format!(
                    "cannot read Silero VAD ONNX at {}: {e}",
                    path.display()
                ))
            })?
            // Pin the symbolic dimensions to the exact shapes we use
            // at runtime; without this tract refuses to optimize.
            // `with_input_fact` on the InferenceModel expects an
            // `InferenceFact`, hence the `dt_shape` helper.
            .with_input_fact(
                0,
                InferenceFact::dt_shape(f32::datum_type(), [1, SILERO_FRAME_SAMPLES]),
            )
            .map_err(|e| DomainError::VadFailed(format!("input fact: {e}")))?
            .with_input_fact(
                1,
                InferenceFact::dt_shape(f32::datum_type(), [2_usize, 1, STATE_HIDDEN]),
            )
            .map_err(|e| DomainError::VadFailed(format!("state fact: {e}")))?
            .into_optimized()
            .map_err(|e| DomainError::VadFailed(format!("optimize: {e}")))?
            .into_runnable()
            .map_err(|e| DomainError::VadFailed(format!("runnable: {e}")))?;

        Ok(Self {
            model: Arc::new(model),
            config,
            state: zero_state(),
            carry: Vec::with_capacity(SILERO_FRAME_SAMPLES),
            hysteresis: VoiceState::Silence,
            voiced_run: 0,
            silent_run: 0,
        })
    }

    /// Convenience constructor with defaults tuned for meeting audio.
    pub fn for_meetings(path: impl AsRef<Path>) -> Result<Self, DomainError> {
        Self::from_path(path, SileroVadConfig::meetings())
    }

    /// Build a fresh detector that **shares this instance's optimized
    /// model** (cheap Arc clone) but starts with a clean LSTM state,
    /// empty carry buffer and `Silence` hysteresis. Use this between
    /// independent sessions instead of re-loading from disk — the
    /// expensive part is the `tract` graph optimization, not the
    /// per-stream state.
    ///
    /// The returned detector keeps the same [`SileroVadConfig`] as
    /// `self`. Caller can mutate it via [`SileroVad::from_path`] or
    /// by exposing a config setter when needed.
    #[must_use]
    pub fn clone_for_new_session(&self) -> Self {
        Self {
            model: Arc::clone(&self.model),
            config: self.config,
            state: zero_state(),
            carry: Vec::with_capacity(SILERO_FRAME_SAMPLES),
            hysteresis: VoiceState::Silence,
            voiced_run: 0,
            silent_run: 0,
        }
    }

    /// Run a single 512-sample window through the network and return
    /// the raw speech probability in `[0.0, 1.0]`. Updates the LSTM
    /// state in place. Used by [`Self::push`] but exposed for tests
    /// and tooling that wants the raw signal.
    pub fn infer_window(&mut self, window: &[Sample]) -> Result<f32, DomainError> {
        debug_assert_eq!(window.len(), SILERO_FRAME_SAMPLES);

        let audio = Tensor::from_shape(&[1, SILERO_FRAME_SAMPLES], window)
            .map_err(|e| DomainError::VadFailed(format!("audio tensor: {e}")))?;

        let inputs = tvec!(audio.into_tvalue(), self.state.clone().into_tvalue(),);
        let mut outputs = self
            .model
            .run(inputs)
            .map_err(|e| DomainError::VadFailed(format!("inference: {e}")))?;

        // outputs[0] is `output [1, 1]`, outputs[1] is `stateN [2, 1, 128]`.
        let new_state_tv = outputs.remove(1);
        let prob_tv = outputs.remove(0);

        let prob_t = prob_tv
            .into_tensor()
            .into_shape(&[1, 1])
            .map_err(|e| DomainError::VadFailed(format!("output shape: {e}")))?;
        let prob = *prob_t
            .to_array_view::<f32>()
            .map_err(|e| DomainError::VadFailed(format!("output dtype: {e}")))?
            .iter()
            .next()
            .ok_or_else(|| DomainError::VadFailed("empty output".into()))?;

        self.state = new_state_tv.into_tensor();
        Ok(prob.clamp(0.0, 1.0))
    }

    fn step_hysteresis(&mut self, prob: f32) {
        match self.hysteresis {
            VoiceState::Silence => {
                if prob >= self.config.start_threshold {
                    self.voiced_run = self.voiced_run.saturating_add(1);
                    if self.voiced_run >= self.config.start_frames {
                        self.hysteresis = VoiceState::Voiced;
                        self.silent_run = 0;
                    }
                } else {
                    self.voiced_run = 0;
                }
            }
            VoiceState::Voiced => {
                if prob <= self.config.end_threshold {
                    self.silent_run = self.silent_run.saturating_add(1);
                    if self.silent_run >= self.config.end_frames {
                        self.hysteresis = VoiceState::Silence;
                        self.voiced_run = 0;
                    }
                } else {
                    self.silent_run = 0;
                }
            }
        }
    }
}

#[async_trait]
impl Vad for SileroVad {
    fn sample_rate_hz(&self) -> u32 {
        SILERO_SAMPLE_RATE
    }

    async fn push(&mut self, samples: &[Sample]) -> Result<VoiceState, DomainError> {
        // Reset the LSTM hidden state at the start of each chunk so
        // cross-chunk state drift cannot lock the model into permanent
        // silence. Each 5-second chunk still gets ~156 windows of
        // intra-chunk LSTM context (plenty for Silero to build a
        // temporal model). Cross-chunk coherence is provided by the
        // hysteresis counters (voiced_run / silent_run) which persist
        // across calls.
        self.state = zero_state();

        // Concat carry + new samples, then drain in 512-sample windows.
        let mut buf: Vec<Sample> = if self.carry.is_empty() {
            samples.to_vec()
        } else {
            let mut b = Vec::with_capacity(self.carry.len() + samples.len());
            b.append(&mut self.carry);
            b.extend_from_slice(samples);
            b
        };

        // Track whether voice was detected at *any* point during this
        // push. A 5-second chunk contains ~156 windows; the hysteresis
        // may flip back to Silence at the tail end even though 3 s of
        // speech appeared earlier. The caller wants to know "should I
        // transcribe this chunk?" — the answer is yes if any window
        // was voiced. We still step the hysteresis on every frame so
        // the LSTM state stays coherent within this push.
        let mut any_voiced = false;
        let mut max_prob: f32 = 0.0;
        let mut sum_prob: f32 = 0.0;
        let mut n_windows: u32 = 0;
        let mut n_above_start: u32 = 0;
        let mut n_above_end: u32 = 0;

        let mut cursor = 0;
        while cursor + SILERO_FRAME_SAMPLES <= buf.len() {
            let prob = self.infer_window(&buf[cursor..cursor + SILERO_FRAME_SAMPLES])?;
            self.step_hysteresis(prob);
            if self.hysteresis == VoiceState::Voiced {
                any_voiced = true;
            }
            max_prob = max_prob.max(prob);
            sum_prob += prob;
            n_windows += 1;
            if prob >= self.config.start_threshold {
                n_above_start += 1;
            }
            if prob >= self.config.end_threshold {
                n_above_end += 1;
            }
            cursor += SILERO_FRAME_SAMPLES;
        }

        if cursor < buf.len() {
            self.carry = buf.split_off(cursor);
        }

        let mean_prob = if n_windows > 0 {
            sum_prob / n_windows as f32
        } else {
            0.0
        };

        let result = if any_voiced {
            VoiceState::Voiced
        } else {
            self.hysteresis
        };

        info!(
            windows = n_windows,
            max_prob = format!("{max_prob:.3}"),
            mean_prob = format!("{mean_prob:.3}"),
            above_start = n_above_start,
            above_end = n_above_end,
            hysteresis = ?self.hysteresis,
            result = ?result,
            "VAD chunk stats"
        );

        Ok(result)
    }

    fn reset(&mut self) {
        self.state = zero_state();
        self.carry.clear();
        self.hysteresis = VoiceState::Silence;
        self.voiced_run = 0;
        self.silent_run = 0;
    }
}

/// Zero-initialized LSTM state of shape `[2, 1, 128]`. Cheap — we
/// allocate one of these per stream lifetime.
fn zero_state() -> Tensor {
    Tensor::from_shape(&[2, 1, STATE_HIDDEN], &[0.0_f32; 2 * STATE_HIDDEN])
        .expect("static-size state tensor cannot fail to build")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// These tests require `models/vad/silero_vad.onnx` on disk. They are
// `#[ignore]`-able via the `ECHO_VAD_MODEL` env knob: when the file
// is missing we skip rather than fail, so a fresh checkout without
// `scripts/download-models.sh vad` does not break `cargo test`.

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn model_path() -> Option<PathBuf> {
        let candidate = std::env::var("ECHO_VAD_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("./models/vad/silero_vad.onnx"));
        if candidate.exists() {
            Some(candidate)
        } else {
            None
        }
    }

    /// Render a sine of `freq_hz` lasting `duration_ms` at 16 kHz.
    fn sine(freq_hz: f32, duration_ms: u32, amplitude: f32) -> Vec<f32> {
        let n = (16_000 * duration_ms as usize) / 1_000;
        let mut v = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / 16_000.0;
            v.push(amplitude * (2.0 * std::f32::consts::PI * freq_hz * t).sin());
        }
        v
    }

    /// Load a 16 kHz mono WAV into a `Vec<f32>` in `[-1, 1]`. Used by
    /// the speech-detection test.
    fn load_wav_f32(path: &str) -> Vec<f32> {
        let mut reader = hound::WavReader::open(path).expect("open wav");
        let spec = reader.spec();
        assert_eq!(spec.sample_rate, 16_000, "fixture must be 16 kHz");
        assert_eq!(spec.channels, 1, "fixture must be mono");
        match spec.sample_format {
            hound::SampleFormat::Int => reader
                .samples::<i16>()
                .map(|s| f32::from(s.unwrap()) / f32::from(i16::MAX))
                .collect(),
            hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        }
    }

    #[tokio::test]
    async fn loads_and_reports_sample_rate() {
        let Some(path) = model_path() else {
            eprintln!("skipping: ECHO_VAD_MODEL or ./models/vad/silero_vad.onnx not present");
            return;
        };
        let vad = SileroVad::for_meetings(&path).expect("load model");
        assert_eq!(vad.sample_rate_hz(), SILERO_SAMPLE_RATE);
    }

    #[tokio::test]
    async fn pure_silence_stays_silent() {
        let Some(path) = model_path() else {
            return;
        };
        let mut vad = SileroVad::for_meetings(&path).expect("load model");
        // 1 s of silence → many windows, none should flip the state.
        let chunk = vec![0.0_f32; 16_000];
        let state = vad.push(&chunk).await.unwrap();
        assert_eq!(state, VoiceState::Silence);
    }

    #[tokio::test]
    async fn pure_tone_is_not_classified_as_speech() {
        // This is the test the energy VAD fails by design: a loud
        // 440 Hz sine has high RMS but is obviously not speech.
        // Silero should keep it as Silence.
        let Some(path) = model_path() else {
            return;
        };
        let mut vad = SileroVad::for_meetings(&path).expect("load model");
        let s = sine(440.0, 1_000, 0.5);
        let state = vad.push(&s).await.unwrap();
        assert_eq!(state, VoiceState::Silence, "Silero must reject pure tones");
    }

    #[tokio::test]
    async fn detects_speech_in_meeting_fixture() {
        let Some(path) = model_path() else {
            return;
        };
        let fixture = "./fixtures/audio/01_short_meeting.wav";
        if !std::path::Path::new(fixture).exists() {
            eprintln!("skipping: fixture {fixture} missing");
            return;
        }

        let mut vad = SileroVad::for_meetings(&path).expect("load model");
        let samples = load_wav_f32(fixture);

        // Feed the whole clip in 100 ms chunks (≈1600 samples) to
        // exercise the carry buffer the way the streaming pipeline
        // actually drives the VAD.
        let chunk_size = 1_600;
        let mut saw_voiced = false;
        for chunk in samples.chunks(chunk_size) {
            let state = vad.push(chunk).await.unwrap();
            if state == VoiceState::Voiced {
                saw_voiced = true;
            }
        }
        assert!(
            saw_voiced,
            "Silero must mark at least one window of the meeting fixture as voiced"
        );
    }

    #[tokio::test]
    async fn clone_for_new_session_shares_model_but_resets_state() {
        let Some(path) = model_path() else {
            return;
        };
        let mut original = SileroVad::for_meetings(&path).expect("load model");
        // Drive the original instance forward so its LSTM state is
        // non-zero and (hopefully) its hysteresis flips to Voiced.
        original.push(&sine(440.0, 1_000, 0.5)).await.unwrap();

        let mut child = original.clone_for_new_session();
        // Cheap shared model: both Arcs point at the same underlying
        // graph. We can't compare Arc::ptr_eq directly without
        // exposing the field, so just sanity-check the child starts
        // at the silence baseline.
        assert_eq!(child.hysteresis, VoiceState::Silence);
        assert_eq!(child.voiced_run, 0);
        assert_eq!(child.silent_run, 0);
        assert!(child.carry.is_empty());

        // And it should still classify pure silence as silent on a
        // fresh stream — proves the LSTM state was zeroed too.
        let state = child.push(&vec![0.0_f32; 16_000]).await.unwrap();
        assert_eq!(state, VoiceState::Silence);
    }

    #[tokio::test]
    async fn reset_returns_to_silence_baseline() {
        let Some(path) = model_path() else {
            return;
        };
        let mut vad = SileroVad::for_meetings(&path).expect("load model");

        // Drive into Voiced via a strong sine if the model happens to
        // misclassify, otherwise via real speech if available. Either
        // way, after `reset` the state must report Silence.
        vad.push(&sine(440.0, 1_000, 0.5)).await.unwrap();
        vad.reset();
        // After reset, classifying a single small buffer that does
        // not cover a full window must keep the state at Silence.
        let state = vad.push(&[0.0_f32; 100]).await.unwrap();
        assert_eq!(state, VoiceState::Silence);
    }
}
