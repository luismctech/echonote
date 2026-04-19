//! Whisper.cpp smoke test gated behind a real model on disk.
//!
//! These tests are NOT run by `cargo test` unless `ECHO_ASR_MODEL`
//! points at a `ggml-*.bin` file, because shipping a model in the
//! repository would balloon the checkout. Locally:
//!
//! ```sh
//! ./scripts/download-models.sh base.en
//! ECHO_ASR_MODEL=$(pwd)/models/asr/ggml-base.en.bin \
//!   cargo test -p echo-asr --test whisper_smoke -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use echo_asr::WhisperCppTranscriber;
use echo_domain::{TranscribeOptions, Transcriber};

fn model_from_env() -> Option<PathBuf> {
    let raw = std::env::var("ECHO_ASR_MODEL").ok()?;
    let path = PathBuf::from(raw);
    if path.exists() {
        Some(path)
    } else {
        eprintln!("ECHO_ASR_MODEL points to {path:?} but it does not exist");
        None
    }
}

#[tokio::test]
#[ignore = "requires ECHO_ASR_MODEL to point at a ggml-*.bin"]
async fn loads_model_and_returns_zero_segments_on_silence() {
    let Some(model) = model_from_env() else {
        eprintln!("skipping: ECHO_ASR_MODEL not set");
        return;
    };

    let transcriber = WhisperCppTranscriber::load(&model).expect("load model");

    // 2 s of digital silence at 16 kHz mono. Whisper should emit zero
    // segments (or a single empty segment).
    let samples = vec![0.0f32; 16_000 * 2];
    let opts = TranscribeOptions {
        language: Some("en".into()),
        ..Default::default()
    };

    let started = std::time::Instant::now();
    let transcript = transcriber.transcribe(&samples, &opts).await.unwrap();
    let elapsed = started.elapsed();

    eprintln!(
        "silence transcript: segments={} text={:?} lang={:?} elapsed={}ms",
        transcript.segments.len(),
        transcript.full_text(),
        transcript.language,
        elapsed.as_millis()
    );

    // Smoke check: we only assert the *shape* of the result. Whisper is
    // known to hallucinate filler tokens like "you" or "(silence)" on
    // pure silence; the goal of this test is to prove the FFI bridge,
    // Metal/CPU backend, model loading and segment iteration all work.
    assert_eq!(transcript.duration_ms, 2_000);
    assert_eq!(
        transcript.language.as_deref(),
        Some("en"),
        "language hint should be echoed back"
    );
    for seg in &transcript.segments {
        assert!(seg.end_ms >= seg.start_ms, "non-monotonic segment: {seg:?}");
        assert!(seg.end_ms <= 2_100, "segment past audio length: {seg:?}");
    }
}

#[tokio::test]
#[ignore = "requires ECHO_ASR_MODEL"]
async fn rejects_missing_model_path() {
    let bogus = PathBuf::from("/tmp/this/path/should/not/exist/ggml.bin");
    let err = WhisperCppTranscriber::load(&bogus).unwrap_err();
    assert!(matches!(err, echo_domain::DomainError::ModelNotLoaded(_)));
}
