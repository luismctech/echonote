//! whisper.cpp adapter via [`whisper-rs`].

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use tracing::{debug, info, instrument};
use whisper_rs::{
    install_logging_hooks, FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters,
    WhisperError,
};

use echo_domain::{
    DomainError, Sample, Segment, SegmentId, TranscribeOptions, Transcriber, Transcript,
};

/// Loads a `ggml-*.bin` Whisper model from disk and serves
/// transcription requests through the [`Transcriber`] port.
///
/// Construction is cheap once the model is loaded, but loading itself
/// is heavy (memory-maps a multi-hundred-MB file and warms up Metal /
/// CPU kernels). Build one instance per process and share it.
#[derive(Clone)]
pub struct WhisperCppTranscriber {
    inner: Arc<Inner>,
}

struct Inner {
    context: WhisperContext,
    model_path: PathBuf,
}

impl std::fmt::Debug for WhisperCppTranscriber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WhisperCppTranscriber")
            .field("model_path", &self.inner.model_path)
            .finish()
    }
}

impl WhisperCppTranscriber {
    /// Load a Whisper model from disk.
    ///
    /// Returns [`DomainError::ModelNotLoaded`] if the file cannot be
    /// opened or whisper.cpp rejects it.
    #[instrument(skip_all, fields(path = %model_path.as_ref().display()))]
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self, DomainError> {
        let path = model_path.as_ref();
        if !path.exists() {
            return Err(DomainError::ModelNotLoaded(format!(
                "{} does not exist",
                path.display()
            )));
        }
        let path_str = path.to_str().ok_or_else(|| {
            DomainError::ModelNotLoaded(format!("{} contains non-UTF-8 characters", path.display()))
        })?;

        install_logging_hooks_once();

        info!(path = %path.display(), "loading whisper model");
        let params = WhisperContextParameters::default();
        let context = WhisperContext::new_with_params(path_str, params)
            .map_err(|e| DomainError::ModelNotLoaded(format!("whisper.cpp: {e}")))?;

        Ok(Self {
            inner: Arc::new(Inner {
                context,
                model_path: path.to_path_buf(),
            }),
        })
    }

    /// Path the model was loaded from.
    #[must_use]
    pub fn model_path(&self) -> &Path {
        &self.inner.model_path
    }
}

#[async_trait]
impl Transcriber for WhisperCppTranscriber {
    #[instrument(skip(self, samples), fields(samples = samples.len()))]
    async fn transcribe(
        &self,
        samples: &[Sample],
        options: &TranscribeOptions,
    ) -> Result<Transcript, DomainError> {
        if samples.is_empty() {
            return Ok(Transcript {
                segments: Vec::new(),
                language: options.language.clone(),
                duration_ms: 0,
            });
        }

        let inner = Arc::clone(&self.inner);
        let pcm = samples.to_vec();
        let opts = options.clone();

        // whisper.cpp's full() is CPU/GPU-bound and blocking. Keep the
        // async runtime free.
        let result = tokio::task::spawn_blocking(move || run_full(&inner.context, &pcm, &opts))
            .await
            .map_err(|e| DomainError::Invariant(format!("transcribe join: {e}")))?;

        result
    }
}

fn run_full(
    context: &WhisperContext,
    samples: &[f32],
    options: &TranscribeOptions,
) -> Result<Transcript, DomainError> {
    let mut state = context
        .create_state()
        .map_err(|e| DomainError::Invariant(format!("whisper create_state: {e}")))?;

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });
    let threads = options
        .threads
        .map(i32::from)
        .unwrap_or_else(|| (num_cpus_div_two() as i32).max(1));
    params.set_n_threads(threads);

    if let Some(lang) = options.language.as_deref() {
        params.set_language(Some(lang));
    } else {
        params.set_language(Some("auto"));
    }
    params.set_translate(options.translate);
    if let Some(prompt) = options.initial_prompt.as_deref() {
        params.set_initial_prompt(prompt);
    }
    // Suppress whisper.cpp's stdout chatter; we have tracing.
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    debug!(threads, language = ?options.language, "running whisper full");
    state.full(params, samples).map_err(map_whisper_err)?;

    let n = state.full_n_segments();
    let mut segments = Vec::with_capacity(n.max(0) as usize);
    for i in 0..n {
        let seg = state.get_segment(i).ok_or_else(|| {
            DomainError::Invariant(format!("whisper segment {i} reported but missing"))
        })?;
        let text = seg.to_str_lossy().map_err(map_whisper_err)?.into_owned();
        // whisper.cpp returns timestamps in centiseconds (10 ms units).
        let start_ms = (seg.start_timestamp().max(0) as u32).saturating_mul(10);
        let end_ms = (seg.end_timestamp().max(0) as u32).saturating_mul(10);
        segments.push(Segment {
            id: SegmentId::new(),
            start_ms,
            end_ms,
            text,
            speaker_id: None,
            confidence: None,
        });
    }

    let detected_lang_id = state.full_lang_id_from_state();
    let language = if detected_lang_id >= 0 {
        whisper_rs::get_lang_str(detected_lang_id).map(str::to_string)
    } else {
        options.language.clone()
    };

    let duration_ms =
        ((samples.len() as u64 * 1_000) / u64::from(echo_audio_whisper_rate())) as u32;

    Ok(Transcript {
        segments,
        language,
        duration_ms,
    })
}

fn map_whisper_err(err: WhisperError) -> DomainError {
    DomainError::Invariant(format!("whisper.cpp: {err}"))
}

/// Half of the logical CPU count, clamped to `[1, 8]`. Whisper.cpp
/// scales sub-linearly past 8 threads on most laptops; the upper bound
/// avoids burning power for negligible speedup.
fn num_cpus_div_two() -> usize {
    let total = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    (total / 2).clamp(1, 8)
}

/// Whisper-canonical sample rate, mirrored from `echo_audio` to avoid
/// pulling that crate in here just for one constant.
const fn echo_audio_whisper_rate() -> u32 {
    16_000
}

/// Redirect whisper.cpp / GGML logs to the `tracing` subscriber. Idempotent.
fn install_logging_hooks_once() {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        install_logging_hooks();
    });
}
