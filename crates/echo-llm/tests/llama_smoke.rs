//! llama.cpp smoke test gated behind a real GGUF model on disk.
//!
//! These tests are NOT exercised by `cargo test` unless `ECHO_LLM_MODEL`
//! points at a `.gguf` file, because shipping a multi-GB model in the
//! repository would balloon the checkout. Locally:
//!
//! ```sh
//! ./scripts/download-models.sh llm
//! ECHO_LLM_MODEL=$(pwd)/models/llm/qwen2.5-7b-instruct-q4_k_m.gguf \
//!   cargo test -p echo-llm --test llama_smoke -- --ignored --nocapture
//! ```
//!
//! The point of the smoke test is to prove the FFI bridge, Metal/CPU
//! backend, model loading, sampler chain and stop-sequence handling
//! all work end-to-end — *not* to pin model output (which would be
//! quantization- and seed-sensitive). We only assert shape and timing.

use std::path::PathBuf;
use std::time::Instant;

use echo_domain::{GenerateOptions, LlmModel};
use echo_llm::{LlamaCppLlm, LoadOptions};

fn model_from_env() -> Option<PathBuf> {
    let raw = std::env::var("ECHO_LLM_MODEL").ok()?;
    let path = PathBuf::from(raw);
    if path.exists() {
        Some(path)
    } else {
        eprintln!("ECHO_LLM_MODEL points to {path:?} but it does not exist");
        None
    }
}

#[tokio::test]
#[ignore = "requires ECHO_LLM_MODEL to point at a .gguf file"]
async fn loads_model_and_completes_a_short_greedy_prompt() {
    let Some(model_path) = model_from_env() else {
        eprintln!("skipping: ECHO_LLM_MODEL not set");
        return;
    };

    // Pin a small context + thread count so the smoke test stays
    // reproducible across machines and the timing assertion below
    // doesn't go wild on hosts with 32 cores vs 4.
    let llm = LlamaCppLlm::load_with(
        &model_path,
        LoadOptions::default()
            .with_n_ctx(2_048)
            .with_n_threads(4)
            .with_model_id("smoke"),
    )
    .expect("load gguf");

    assert_eq!(llm.model_id(), "smoke");

    // Use the Qwen chat template — works on Qwen 2 / 2.5 / 3 GGUFs.
    // For other model families this prompt is harmless raw text and
    // the model will continue it; the test only cares that we get a
    // non-empty completion back.
    let prompt = "<|im_start|>system\nYou are a helpful assistant. Answer with a single short sentence.<|im_end|>\n\
                  <|im_start|>user\nReply with the literal word OK and nothing else.<|im_end|>\n\
                  <|im_start|>assistant\n";

    let opts = GenerateOptions {
        max_tokens: 16,
        temperature: 0.0,
        top_p: 1.0,
        seed: Some(7),
        stop: vec!["<|im_end|>".into()],
    };

    let started = Instant::now();
    let out = llm
        .generate(prompt, &opts)
        .await
        .expect("generation succeeds");
    let elapsed = started.elapsed();

    eprintln!(
        "smoke completion: bytes={} elapsed={}ms output={:?}",
        out.len(),
        elapsed.as_millis(),
        out
    );

    assert!(!out.trim().is_empty(), "expected non-empty completion");
    assert!(
        !out.contains("<|im_end|>"),
        "stop sequence must be stripped from the output"
    );
    // Generous ceiling — even on CPU-only laptops a 16-token greedy
    // completion finishes in well under a minute.
    assert!(
        elapsed.as_secs() < 60,
        "smoke completion took {elapsed:?}, something is very wrong"
    );
}

#[tokio::test]
#[ignore = "requires ECHO_LLM_MODEL"]
async fn rejects_missing_model_path() {
    let bogus = PathBuf::from("/tmp/this/path/should/not/exist.gguf");
    let err = LlamaCppLlm::load(&bogus).unwrap_err();
    assert!(matches!(err, echo_domain::DomainError::ModelNotLoaded(_)));
}
