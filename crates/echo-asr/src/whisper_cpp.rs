//! whisper.cpp adapter via [`whisper-rs`].

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use async_trait::async_trait;
use tracing::{debug, info, instrument};
use whisper_rs::{
    install_logging_hooks, FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters,
    WhisperError, WhisperState,
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
    /// Cached [`WhisperState`] to avoid the ~470 MB GPU buffer
    /// allocation that `create_state()` triggers on Metal per call.
    /// The state is taken out of the mutex for the duration of each
    /// `full()` call and returned afterwards so the next invocation
    /// reuses the same GPU buffers instead of alloc/dealloc cycling.
    cached_state: Mutex<Option<WhisperState>>,
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

        // Eagerly create the first state so Metal GPU buffers are
        // allocated once at load time rather than on the first chunk.
        let initial_state = context
            .create_state()
            .map_err(|e| DomainError::ModelNotLoaded(format!("whisper create_state: {e}")))?;
        info!("whisper state pre-warmed (GPU buffers allocated once)");

        Ok(Self {
            inner: Arc::new(Inner {
                context,
                model_path: path.to_path_buf(),
                cached_state: Mutex::new(Some(initial_state)),
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
        let result = tokio::task::spawn_blocking(move || run_full(&inner, &pcm, &opts))
            .await
            .map_err(|e| DomainError::Invariant(format!("transcribe join: {e}")))?;

        result
    }
}

fn run_full(
    inner: &Inner,
    samples: &[f32],
    options: &TranscribeOptions,
) -> Result<Transcript, DomainError> {
    // Take the cached state (or create a fresh one if somehow missing).
    let mut state = inner
        .cached_state
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .take()
        .or_else(|| {
            debug!("cached whisper state missing; creating a new one");
            inner.context.create_state().ok()
        })
        .ok_or_else(|| DomainError::Invariant("whisper create_state failed".into()))?;

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

    // ── Hallucination mitigation ─────────────────────────────────────
    // Whisper is famous for producing canned YouTube outros ("Gracias
    // por ver el vídeo", "Subscribe to the channel", "Subtitles by the
    // Amara.org community") on silent or near-silent audio, especially
    // in streaming mode. The defaults are tuned for offline batch
    // transcription of clean files, not for our 5-second chunks of
    // mic input. We tighten three knobs:
    //
    // 1. `no_context = true` — never feed the previous chunk's text
    //    back as the decoder prompt. This is THE most important fix:
    //    once the model emits "Gracias", with context enabled the next
    //    chunk sees it in the prompt and self-reinforces into "Gracias
    //    por ver el vídeo" → repeating ad infinitum.
    // 2. `temperature_inc = 0.0` — disable temperature fallback. By
    //    default whisper.cpp retries failed segments with higher
    //    temperatures (0.0, 0.2, 0.4, …). That's exactly when the
    //    sampler invents text. Pure greedy at T=0 is deterministic.
    // 3. Explicit `no_speech_thold` / `logprob_thold` / `entropy_thold`
    //    so segments that the model itself rates as low-confidence get
    //    dropped instead of returned as "Gracias." with high
    //    no-speech probability.
    // 4. `suppress_nst = true` — suppress non-speech tokens like
    //    `(música)`, `[applause]`, etc. that whisper.cpp emits on
    //    silence in some languages.
    params.set_no_context(true);
    params.set_temperature(0.0);
    params.set_temperature_inc(0.0);
    // Tighter than the 0.6 default — empirically Whisper still emits
    // confident single-word hallucinations ("Gracias.", "Thank you.")
    // on near-silent chunks at 0.6, whereas at 0.5 those segments get
    // dropped because the no-speech token wins more often.
    params.set_no_speech_thold(0.5);
    params.set_logprob_thold(-1.0);
    params.set_entropy_thold(2.4);
    params.set_suppress_blank(true);
    params.set_suppress_nst(true);

    debug!(threads, language = ?options.language, "running whisper full");

    // Wrap the FFI call in catch_unwind so a panic in C++ (OOM, Metal
    // driver fault, SIGABRT) doesn't permanently lose the cached
    // WhisperState (~470 MB of GPU buffers on Metal). On caught panic
    // we return the state to the cache and surface the error.
    let full_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        state.full(params, samples).map_err(map_whisper_err)
    }));
    match full_result {
        Ok(Ok(())) => { /* success — continue extracting segments */ }
        Ok(Err(e)) => {
            // Whisper returned an error but didn't panic — state is
            // still usable, return it to the cache.
            let _ = inner
                .cached_state
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .insert(state);
            return Err(e);
        }
        Err(panic_payload) => {
            // C++ panic caught. The state *may* be corrupted, but
            // returning it avoids a guaranteed 470 MB re-allocation.
            // If it truly is corrupt the next call will fail and
            // create_state will be the fallback.
            tracing::warn!("whisper.cpp panicked during full(); returning state to cache");
            let _ = inner
                .cached_state
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .insert(state);
            let msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                format!("whisper.cpp panicked: {s}")
            } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                format!("whisper.cpp panicked: {s}")
            } else {
                "whisper.cpp panicked (unknown payload)".to_string()
            };
            return Err(DomainError::Invariant(msg));
        }
    }

    let n = state.full_n_segments();
    let mut segments = Vec::with_capacity(n.max(0) as usize);
    for i in 0..n {
        let seg = state.get_segment(i).ok_or_else(|| {
            DomainError::Invariant(format!("whisper segment {i} reported but missing"))
        })?;

        // ── Filter 1: no-speech probability ─────────────────────────
        // whisper-rs note: set_no_speech_thold() is documented as "not
        // implemented" in whisper.cpp ≤ v1.3 — the param exists but is
        // not enforced by the engine. We therefore apply the threshold
        // post-hoc on the segment-level probability the model emits.
        // 0.6 catches residual hallucinations that survive the existing
        // entropy / logprob guards without being too aggressive on real
        // speech fragments with low acoustic confidence.
        let no_speech_prob = seg.no_speech_probability();
        if no_speech_prob > NO_SPEECH_POST_THOLD {
            debug!(no_speech_prob, "dropping high-no-speech-prob segment");
            continue;
        }

        let text = seg.to_str_lossy().map_err(map_whisper_err)?.into_owned();

        // ── Filter 2: known-hallucination phrase list ────────────────
        if is_known_hallucination(&text) {
            debug!(text = %text, "dropping known whisper hallucination");
            continue;
        }

        // ── Filter 3: greedy repetition-loop detection ───────────────
        // At temperature = 0 the greedy decoder can get stuck repeating
        // the same word or phrase. `is_repetitive_loop` catches these
        // collapsed outputs (4+ consecutive repetitions of an n-gram)
        // before they reach the transcript.
        if is_repetitive_loop(&text) {
            debug!(text = %text, "dropping repetitive-loop segment");
            continue;
        }

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

    // Return the state to the pool so the next call reuses the GPU
    // buffers instead of re-allocating ~470 MB on Metal.
    let _ = inner
        .cached_state
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .insert(state);

    Ok(Transcript {
        segments,
        language,
        duration_ms,
    })
}

/// Post-hoc no-speech probability threshold applied per segment after
/// `state.full()` returns. Segments whose `no_speech_prob` exceeds this
/// are dropped before they reach the transcript.
///
/// Rationale: `set_no_speech_thold` in whisper.cpp ≤ v1.3 is documented
/// as "not implemented" — the engine does not enforce it, so we must
/// apply the gate ourselves. 0.6 is deliberately conservative: it only
/// drops segments the model itself rates as very likely non-speech.
const NO_SPEECH_POST_THOLD: f32 = 0.6;

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

/// Return `true` when `text` shows signs of a greedy decoder repetition
/// loop: any word-level n-gram (size 1–3) that appears **4 or more
/// consecutive times** in the same segment.
///
/// Example pathological outputs this catches:
/// - "Thank you thank you thank you thank you thank you"
/// - "the the the the the the meeting"
/// - "I think I think I think I think so"
///
/// Minimum segment length: 4 words (shorter segments are never loops).
fn is_repetitive_loop(text: &str) -> bool {
    let words: Vec<&str> = text.split_whitespace().collect();
    let n = words.len();
    if n < 4 {
        return false;
    }
    for gram_size in 1..=3_usize {
        if gram_size * 4 > n {
            break;
        }
        let mut i = 0;
        while i + gram_size * 4 <= n {
            let mut reps = 0usize;
            let mut j = i;
            while j + gram_size <= n && words[j..j + gram_size] == words[i..i + gram_size] {
                reps += 1;
                j += gram_size;
            }
            if reps >= 4 {
                return true;
            }
            i += 1;
        }
    }
    false
}

/// Return `true` when `text` is one of the canonical phrases Whisper
/// hallucinates on silent or near-silent audio. We keep the list
/// **conservative on purpose**: only drop multi-word YouTube/Amara
/// outros plus meta-tokens that are unambiguously not real meeting
/// content. A bare "Gracias." or "Thank you." is left intact because
/// both can be legitimate utterances.
///
/// Matching is case-insensitive and ignores leading/trailing
/// punctuation/whitespace plus accent variants on the Spanish "vídeo"
/// (some Latin-American sub-models drop the accent).
fn is_known_hallucination(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }

    // Generic catch: any segment whose entire content is wrapped in
    // square brackets `[...]` or parentheses `(...)` is a non-speech
    // marker the model emitted (e.g. `[no speech]`, `[BLANK_AUDIO]`,
    // `(música)`, `[silencio]`, `(applause)`). Real meeting speech
    // never starts with `[` or `(`, so this is safe to drop.
    if (trimmed.starts_with('[') && trimmed.ends_with(']'))
        || (trimmed.starts_with('(') && trimmed.ends_with(')'))
    {
        return true;
    }

    // Strip whitespace, leading/trailing punctuation, lowercase, and
    // collapse the two common accent variants of "vídeo".
    let normalized: String = trimmed
        .trim_matches(|c: char| {
            c.is_ascii_punctuation() || c == '¡' || c == '¿' || c == '…' || c == '"' || c == '\''
        })
        .to_lowercase()
        .replace('á', "a")
        .replace('é', "e")
        .replace('í', "i")
        .replace('ó', "o")
        .replace('ú', "u")
        // Normalize underscores → spaces. Whisper.cpp emits markers
        // like `BLANK_AUDIO` that should match the spaced form below.
        // No real Spanish/English word contains underscore.
        .replace('_', " ")
        // collapse multiple spaces produced by the steps above
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");

    const NEEDLES: &[&str] = &[
        // Spanish YouTube outros — by far the most common in our setup.
        "gracias por ver el video",
        "gracias por ver este video",
        "muchas gracias por ver el video",
        "no olviden suscribirse",
        "suscribete al canal",
        "suscribanse al canal",
        "nos vemos en el proximo video",
        // Spanish subtitle credits (community subs).
        "subtitulos por la comunidad de amara.org",
        "subtitulos realizados por la comunidad de amara.org",
        "subtitulado por la comunidad de amara.org",
        "subtitulos en espanol",
        "mas informacion www.alimmenta.com",
        // Bare meta-tokens whisper.cpp emits without the bracket
        // wrappers above (older models, certain quants).
        "no speech",
        "blank audio",
        "silence",
        "silencio",
        "music",
        "musica",
        // English equivalents (in case auto-detect picks en for silent
        // chunks even with --language es).
        "thanks for watching",
        "thank you for watching",
        "subscribe to the channel",
        "subtitles by the amara.org community",
    ];

    NEEDLES.iter().any(|needle| normalized == *needle)
}

#[cfg(test)]
mod tests {
    use super::{is_known_hallucination, is_repetitive_loop};

    #[test]
    fn drops_canonical_spanish_youtube_outros() {
        assert!(is_known_hallucination("Gracias por ver el vídeo."));
        assert!(is_known_hallucination("Gracias por ver el video"));
        assert!(is_known_hallucination(" gracias por ver el vídeo "));
        assert!(is_known_hallucination("¡Gracias por ver el vídeo!"));
        assert!(is_known_hallucination(
            "Subtítulos por la comunidad de Amara.org"
        ));
        assert!(is_known_hallucination("Suscríbete al canal"));
    }

    #[test]
    fn drops_english_outros() {
        assert!(is_known_hallucination("Thanks for watching"));
        assert!(is_known_hallucination(" Thank you for watching. "));
        assert!(is_known_hallucination(
            "Subtitles by the Amara.org community"
        ));
    }

    #[test]
    fn keeps_legitimate_short_utterances() {
        // Bare "Gracias." is a real word; we must not drop it.
        assert!(!is_known_hallucination("Gracias."));
        assert!(!is_known_hallucination("Gracias"));
        assert!(!is_known_hallucination("Muchas gracias."));
        assert!(!is_known_hallucination("Thank you."));
        // Substring matches must not trigger — only exact normalised matches.
        assert!(!is_known_hallucination(
            "Gracias por ver el vídeo que te envié ayer"
        ));
        assert!(!is_known_hallucination("Hola, gracias por ver"));
    }

    #[test]
    fn handles_empty_and_whitespace() {
        assert!(!is_known_hallucination(""));
        assert!(!is_known_hallucination("   "));
        assert!(!is_known_hallucination("..."));
    }

    #[test]
    fn drops_bracketed_meta_tokens() {
        assert!(is_known_hallucination("[no speech]"));
        assert!(is_known_hallucination("[BLANK_AUDIO]"));
        assert!(is_known_hallucination("[silencio]"));
        assert!(is_known_hallucination("[Music]"));
        assert!(is_known_hallucination(" [música] "));
    }

    #[test]
    fn drops_parenthesised_meta_tokens() {
        assert!(is_known_hallucination("(música)"));
        assert!(is_known_hallucination("(silence)"));
        assert!(is_known_hallucination("(applause)"));
    }

    #[test]
    fn drops_bare_meta_tokens_without_brackets() {
        assert!(is_known_hallucination("No speech"));
        assert!(is_known_hallucination(
            "BLANK_AUDIO".to_lowercase().as_str()
        ));
        assert!(is_known_hallucination("silence."));
        assert!(is_known_hallucination("Música"));
    }

    #[test]
    fn detects_single_word_loop() {
        assert!(is_repetitive_loop("the the the the"));
        assert!(is_repetitive_loop(
            "thank you thank you thank you thank you"
        ));
    }

    #[test]
    fn detects_bigram_loop() {
        assert!(is_repetitive_loop(
            "I think I think I think I think so maybe"
        ));
        assert!(is_repetitive_loop(
            "thank you thank you thank you thank you very much"
        ));
    }

    #[test]
    fn keeps_legitimate_speech_with_repetition() {
        // Natural repetition under the threshold.
        assert!(!is_repetitive_loop("yes yes I agree yes"));
        assert!(!is_repetitive_loop("very very good"));
        assert!(!is_repetitive_loop(
            "he said that that was not the right answer"
        ));
    }

    #[test]
    fn short_segments_never_loop() {
        assert!(!is_repetitive_loop(""));
        assert!(!is_repetitive_loop("ok"));
        assert!(!is_repetitive_loop("thank you"));
    }

    #[test]
    fn keeps_text_with_internal_brackets() {
        // Real speech occasionally produces `aside [...]` content; we
        // must only filter when the entire segment is bracketed.
        assert!(!is_known_hallucination(
            "el cliente dijo [textualmente] que no"
        ));
        assert!(!is_known_hallucination("Entonces (más o menos) sí"));
    }
}
