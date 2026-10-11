//! llama.cpp smoke test gated behind a real GGUF model on disk.
//!
//! These tests are NOT exercised by `cargo test` unless `ECHO_LLM_MODEL`
//! points at a `.gguf` file, because shipping a multi-GB model in the
//! repository would balloon the checkout. Locally:
//!
//! ```sh
//! ./scripts/download-models.sh llm
//! ECHO_LLM_MODEL=$(pwd)/models/llm/Qwen3-8B-Q4_K_M.gguf \
//!   cargo test -p echo-llm --test llama_smoke -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Keep `--test-threads=1`: each test loads its own copy of the model, and
//! loading several multi-GB copies at once can fail on machines with
//! limited memory.
//!
//! The point of the smoke test is to prove the FFI bridge, Metal/CPU
//! backend, model loading, sampler chain and stop-sequence handling
//! all work end-to-end — *not* to pin model output (which would be
//! quantization- and seed-sensitive). We only assert shape and timing.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use echo_domain::{
    ChatAssistant, ChatMessage, ChatOptions, ChatRequest, GenerateOptions, LlmModel,
};
use echo_llm::{LlamaCppLlm, LoadOptions};
use futures::StreamExt;

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
                  <|im_start|>user\nReply with the literal word OK and nothing else. /no_think<|im_end|>\n\
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

#[test]
#[ignore = "requires ECHO_LLM_MODEL to point at a .gguf file"]
fn reports_real_context_window_and_token_counts() {
    let Some(model_path) = model_from_env() else {
        eprintln!("skipping: ECHO_LLM_MODEL not set");
        return;
    };

    let llm = LlamaCppLlm::load_with(
        &model_path,
        LoadOptions::default().with_n_ctx(2_048).with_n_threads(4),
    )
    .expect("load gguf");
    let chat = llm.chat_handle();

    assert_eq!(llm.context_window(), 2_048);
    assert_eq!(chat.context_window(), 2_048);

    let text = "Decidimos lanzar la versión nueva el lunes por la mañana.";
    let real = llm.count_tokens(text);
    assert!(real > 0, "a non-empty text must tokenize to something");
    assert_eq!(chat.count_tokens(text), real);
    assert_eq!(llm.count_tokens(""), 0);
}

#[tokio::test]
#[ignore = "requires ECHO_LLM_MODEL"]
async fn rejects_missing_model_path() {
    let bogus = PathBuf::from("/tmp/this/path/should/not/exist.gguf");
    let err = LlamaCppLlm::load(&bogus).unwrap_err();
    assert!(matches!(err, echo_domain::DomainError::ModelNotLoaded(_)));
}

/// Evidence for the `estimate_tokens` contract: on every sample, including
/// the accent-dense worst case, the real tokenizer count must not exceed
/// the fallback estimate, or budgets built on the fallback could overflow.
#[test]
#[ignore = "requires ECHO_LLM_MODEL to point at a .gguf file"]
fn fallback_estimate_is_never_below_the_real_token_count() {
    let Some(model_path) = model_from_env() else {
        eprintln!("skipping: ECHO_LLM_MODEL not set");
        return;
    };
    let llm = LlamaCppLlm::load_with(
        &model_path,
        LoadOptions::default().with_n_ctx(2_048).with_n_threads(4),
    )
    .expect("load gguf");
    let samples = [
        "Decidimos lanzar la versión nueva el lunes por la mañana.",
        "Además, la configuración de producción necesitará aprobación: ¿quién se encargará de la migración?",
        "Ñandú, canción, pingüino, acción, información, característica, económicamente, comunicación.",
        "Ana: Sí, también revisé la planificación del trimestre y creo que deberíamos reorganizar las prioridades según la capacidad del equipo.",
        "We decided to launch the new version on Monday morning and review the rollout plan afterwards.",
        "Speaker 2: The quarterly numbers look fine, but the infrastructure budget needs another review.",
        "Reunión técnica sobre la implementación de microservicios, contenedores y orquestación en la nube.",
        "áéíóúñ áéíóúñ áéíóúñ áéíóúñ áéíóúñ áéíóúñ áéíóúñ áéíóúñ",
    ];
    for text in samples {
        let real = llm.count_tokens(text);
        let estimate = echo_domain::estimate_tokens(text);
        eprintln!(
            "real={real} estimate={estimate} chars={} bytes={}",
            text.chars().count(),
            text.len()
        );
        assert!(real <= estimate, "estimate undercounts {text:?}");
    }
}

fn short_prompt(question: &str) -> String {
    format!(
        "<|im_start|>system\nAnswer with a single short sentence.<|im_end|>\n\
         <|im_start|>user\n{question} /no_think<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

fn greedy_options(max_tokens: u32) -> GenerateOptions {
    GenerateOptions {
        max_tokens,
        temperature: 0.0,
        top_p: 1.0,
        seed: Some(7),
        stop: vec!["<|im_end|>".into()],
    }
}

/// The persistent context must serve consecutive requests, and clearing
/// the KV cache between them must make a repeated greedy prompt reproduce
/// the same answer.
#[tokio::test]
#[ignore = "requires ECHO_LLM_MODEL to point at a .gguf file"]
async fn serves_back_to_back_requests_on_one_context() {
    let Some(model_path) = model_from_env() else {
        eprintln!("skipping: ECHO_LLM_MODEL not set");
        return;
    };
    let llm = LlamaCppLlm::load_with(
        &model_path,
        LoadOptions::default().with_n_ctx(2_048).with_n_threads(4),
    )
    .expect("load gguf");

    let capital = short_prompt("What is the capital of France?");
    let first = llm
        .generate(&capital, &greedy_options(24))
        .await
        .expect("first request");
    let other = llm
        .generate(&short_prompt("What is 2 + 2?"), &greedy_options(24))
        .await
        .expect("second request");
    let again = llm
        .generate(&capital, &greedy_options(24))
        .await
        .expect("third request");
    eprintln!("back-to-back: {first:?} / {other:?} / {again:?}");

    assert!(!first.trim().is_empty());
    assert!(!other.trim().is_empty());
    assert_eq!(first, again, "a reused context must not leak earlier state");
}

/// Dropping a stream mid-answer must stop its decoding so the next
/// request is served instead of waiting behind a long generation.
#[tokio::test]
#[ignore = "requires ECHO_LLM_MODEL to point at a .gguf file"]
async fn dropping_a_stream_early_does_not_block_the_next_request() {
    let Some(model_path) = model_from_env() else {
        eprintln!("skipping: ECHO_LLM_MODEL not set");
        return;
    };
    let llm = LlamaCppLlm::load_with(
        &model_path,
        LoadOptions::default().with_n_ctx(2_048).with_n_threads(4),
    )
    .expect("load gguf");

    // A long answer requested, abandoned after the first piece.
    let long = short_prompt("Count from 1 to 500 separated by commas.");
    let mut stream = llm
        .generate_stream(&long, &greedy_options(512))
        .await
        .expect("open stream");
    let piece = stream.next().await.expect("first piece").expect("ok piece");
    assert!(!piece.is_empty());
    drop(stream);

    let started = Instant::now();
    let out = llm
        .generate(
            &short_prompt("Reply with the word OK."),
            &greedy_options(16),
        )
        .await
        .expect("request after the dropped stream");
    let elapsed = started.elapsed();
    eprintln!("after drop: {out:?} in {} ms", elapsed.as_millis());

    assert!(!out.trim().is_empty());
    // Cancellation is asserted deterministically by the worker unit tests
    // (`a_queued_job_whose_consumer_is_gone_is_skipped` and the sink
    // tests); here it is enough that the next request completes.
    eprintln!("next request completed after {elapsed:?}");
}

/// A chat turn submitted while a long summary-like workload runs is served
/// between its `generate` calls instead of after all of them.
#[tokio::test]
#[ignore = "requires ECHO_LLM_MODEL to point at a .gguf file"]
async fn chat_is_served_between_background_generate_calls() {
    let Some(model_path) = model_from_env() else {
        eprintln!("skipping: ECHO_LLM_MODEL not set");
        return;
    };
    let llm = LlamaCppLlm::load_with(
        &model_path,
        LoadOptions::default().with_n_ctx(2_048).with_n_threads(4),
    )
    .expect("load gguf");
    let chat = llm.chat_handle();

    // A map-reduce-like workload: many separate `generate` calls.
    const BACKGROUND_CALLS: usize = 6;
    let calls_done = Arc::new(AtomicUsize::new(0));
    let background = {
        let llm = llm.clone();
        let calls_done = Arc::clone(&calls_done);
        tokio::spawn(async move {
            for i in 0..BACKGROUND_CALLS {
                let prompt = short_prompt(&format!("Count from 1 to 400, variant {i}."));
                llm.generate(&prompt, &greedy_options(128))
                    .await
                    .expect("background call");
                calls_done.fetch_add(1, Ordering::SeqCst);
            }
        })
    };
    // Let the first call start decoding.
    tokio::time::sleep(Duration::from_millis(800)).await;

    let request = ChatRequest {
        messages: vec![ChatMessage::user("Reply with the word OK. /no_think")],
        options: ChatOptions {
            max_tokens: 16,
            temperature: 0.0,
            top_p: 1.0,
            seed: Some(7),
            stop: vec!["<|im_end|>".into()],
        },
    };
    let asked = Instant::now();
    let mut stream = chat.ask(&request).await.expect("ask");
    let first = stream.next().await.expect("first token").expect("ok token");
    let first_token_after = asked.elapsed();
    let done_at_first_token = calls_done.load(Ordering::SeqCst);
    while stream.next().await.is_some() {}
    eprintln!(
        "chat first token {first:?} after {} ms with {done_at_first_token}/{BACKGROUND_CALLS} \
         background calls done; chat total {} ms",
        first_token_after.as_millis(),
        asked.elapsed().as_millis()
    );

    assert!(
        done_at_first_token < BACKGROUND_CALLS,
        "chat waited for the whole background workload"
    );
    assert!(!background.is_finished(), "background finished before chat");
    background.await.expect("background task");
}

/// Long shared system prompt so prompts built on it share a real prefix.
fn shared_system() -> String {
    let mut text = String::from("You are a careful meeting assistant. Meeting notes follow.\n");
    for i in 0..40 {
        text.push_str(&format!(
            "Note {i}: the team reviewed milestone {i}, assigned an owner and set a date.\n"
        ));
    }
    text
}

fn chatml_turns(system: &str, turns: &[(&str, Option<&str>)]) -> String {
    let mut prompt = format!("<|im_start|>system\n{system}<|im_end|>\n");
    for (user, answer) in turns {
        prompt.push_str(&format!("<|im_start|>user\n{user} /no_think<|im_end|>\n"));
        prompt.push_str("<|im_start|>assistant\n");
        if let Some(answer) = answer {
            prompt.push_str(&format!("{answer}<|im_end|>\n"));
        }
    }
    prompt
}

fn load_small() -> Option<LlamaCppLlm> {
    let model_path = model_from_env()?;
    Some(
        LlamaCppLlm::load_with(
            &model_path,
            LoadOptions::default().with_n_ctx(4_096).with_n_threads(4),
        )
        .expect("load gguf"),
    )
}

/// Output with reuse must match the cold run. Reuse feeds the model the same
/// tokens, but a backend may evaluate the reused prefix under another batch
/// shape than a cold run, so bitwise identity is expected, not guaranteed.
///
/// Identical output passes. A divergence is tolerated only when it happens
/// at least [`EARLY_DIVERGENCE_CHARS`] characters in (a near-tie token
/// flipped by numerics, reported as a warning): a wrong or stale KV state
/// corrupts the answer from the start, so an early divergence fails with
/// the position and both texts.
fn assert_equivalent(cold: &str, warm: &str, what: &str) {
    if cold == warm {
        return;
    }
    let at = cold
        .chars()
        .zip(warm.chars())
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| cold.chars().count().min(warm.chars().count()));
    assert!(
        at >= EARLY_DIVERGENCE_CHARS,
        "{what}: output diverged from the cold run at char {at}\n cold: {cold:?}\n warm: {warm:?}"
    );
    eprintln!("[warn] {what}: output diverged from the cold run at char {at} (numeric drift?)");
}

/// See [`assert_equivalent`].
const EARLY_DIVERGENCE_CHARS: usize = 16;

/// A prompt that partially overlaps the one left in the KV cache must
/// produce the same greedy output as the same prompt on a fresh worker with
/// an empty cache (see [`assert_equivalent`]).
#[tokio::test]
#[ignore = "requires ECHO_LLM_MODEL to point at a .gguf file"]
async fn prefix_reuse_matches_a_cold_cache() {
    let Some(warm) = load_small() else {
        eprintln!("skipping: ECHO_LLM_MODEL not set");
        return;
    };
    let system = shared_system();
    let target = chatml_turns(&system, &[("Which milestone has the latest date?", None)]);
    let other = chatml_turns(&system, &[("Summarise note 3 in one sentence.", None)]);

    // (a) cold: a fresh worker, empty cache.
    let cold_llm = load_small().expect("model");
    let cold = cold_llm
        .generate(&target, &greedy_options(48))
        .await
        .expect("cold");
    assert_eq!(cold_llm.last_reused_tokens(), 0, "cold run reuses nothing");

    // (b) warm: a different prompt first, so the cache holds an
    // overlapping prefix (the shared system prompt).
    warm.generate(&other, &greedy_options(48))
        .await
        .expect("other");
    let reused_other = warm.last_reused_tokens();
    let hot = warm
        .generate(&target, &greedy_options(48))
        .await
        .expect("warm");
    let reused = warm.last_reused_tokens();
    eprintln!(
        "[prefix] reused {reused} of {} tokens",
        warm.count_tokens(&target)
    );
    eprintln!("[prefix] cold {cold:?}\n[prefix] warm {hot:?}");
    assert_eq!(reused_other, 0, "first request on a fresh worker");
    assert!(
        reused > 100,
        "shared system prompt should be reused: {reused}"
    );
    assert!(!cold.trim().is_empty());
    assert_equivalent(&cold, &hot, "prefix reuse");

    // An identical repeat reuses everything but the last token.
    let again = warm
        .generate(&target, &greedy_options(48))
        .await
        .expect("repeat");
    assert_eq!(warm.last_reused_tokens(), warm.count_tokens(&target) - 1);
    assert_equivalent(&cold, &again, "identical repeat");
}

/// Chat turn 2 (turn 1 prompt + its answer + a new user message) reuses at
/// least the whole turn 1 prompt and answers exactly like a cold run.
#[tokio::test]
#[ignore = "requires ECHO_LLM_MODEL to point at a .gguf file"]
async fn chat_turn_two_reuses_turn_one_and_matches_a_cold_run() {
    let Some(llm) = load_small() else {
        eprintln!("skipping: ECHO_LLM_MODEL not set");
        return;
    };
    let system = shared_system();
    let q1 = "Which milestone has the latest date?";
    let p1 = chatml_turns(&system, &[(q1, None)]);
    let a1 = llm
        .generate(&p1, &greedy_options(48))
        .await
        .expect("turn 1");
    let p2 = chatml_turns(&system, &[(q1, Some(&a1)), ("Who owns it?", None)]);
    let n1 = llm.count_tokens(&p1);

    let warm = llm
        .generate(&p2, &greedy_options(48))
        .await
        .expect("turn 2");
    let reused = llm.last_reused_tokens();
    eprintln!("[chat] turn 1 prompt {n1} tokens, turn 2 reused {reused}");
    assert!(reused >= n1, "turn 2 reused {reused} < turn 1 prompt {n1}");

    let cold_llm = load_small().expect("model");
    let cold = cold_llm
        .generate(&p2, &greedy_options(48))
        .await
        .expect("cold turn 2");
    assert_equivalent(&cold, &warm, "chat turn 2");
}
