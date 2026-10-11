//! Real-model acceptance tests for the summary use case.
//!
//! Gated behind a GGUF model on disk (like `echo-llm/tests/llama_smoke.rs`):
//! the tests are `#[ignore]`d and also skip cleanly when `ECHO_LLM_MODEL`
//! is unset. Locally:
//!
//! ```sh
//! ECHO_LLM_MODEL=$(pwd)/models/llm/Qwen3-8B-Q4_K_M.gguf \
//!   cargo test -p echo-app --test summary_real_model -- --ignored --nocapture --test-threads=1
//! ```
//!
//! They prove what the scripted unit tests cannot: that map-reduce over a
//! long meeting fits the production 8,192-token window with the model's own
//! tokenizer, yields the structured General template, and that a fact in the
//! middle of the transcript survives the map step.
//!
//! Accepted limitation (user decision, 2026-10-10): with Qwen3-8B the final
//! General summary of a long, filler-heavy meeting may omit an isolated
//! figure, because the reduce step has to compress many notes into a few
//! template fields. The long test therefore asserts the fact reaches at least
//! one intermediate map note and only reports whether the final summary kept
//! it.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use futures::stream::BoxStream;

use echo_app::SummarizeMeeting;
use echo_domain::{ChatMessage, DomainError, GenerateOptions, LlmModel, SummaryContent};
use echo_llm::LlamaCppLlm;

mod common;
use common::{model_from_env, seed_meeting, synthetic_dialogue, PLANTED_NUMBER};

/// Decorator counting how many completions the use case asks the model for.
struct CountingLlm {
    inner: Arc<LlamaCppLlm>,
    calls: AtomicUsize,
    /// Responses to map prompts (identified by the note-taker system prompt).
    map_notes: Mutex<Vec<String>>,
}

#[async_trait]
impl LlmModel for CountingLlm {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }
    fn context_window(&self) -> u32 {
        self.inner.context_window()
    }
    fn count_tokens(&self, text: &str) -> usize {
        self.inner.count_tokens(text)
    }
    fn count_chat_tokens(&self, messages: &[ChatMessage]) -> usize {
        self.inner.count_chat_tokens(messages)
    }
    async fn generate(
        &self,
        prompt: &str,
        options: &GenerateOptions,
    ) -> Result<String, DomainError> {
        self.inner.generate(prompt, options).await
    }
    async fn generate_chat(
        &self,
        messages: &[ChatMessage],
        options: &GenerateOptions,
    ) -> Result<String, DomainError> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        let response = self.inner.generate_chat(messages, options).await?;
        if messages
            .iter()
            .any(|m| m.content.contains("meticulous meeting note-taker"))
        {
            self.map_notes.lock().unwrap().push(response.clone());
        }
        // Diagnostics only: shows whether a note reached its output cap.
        eprintln!(
            "call {n}: max_tokens={}, response={} tokens",
            options.max_tokens,
            self.inner.count_tokens(&response)
        );
        Ok(response)
    }
    async fn generate_chat_stream(
        &self,
        messages: &[ChatMessage],
        options: &GenerateOptions,
    ) -> Result<BoxStream<'static, Result<String, DomainError>>, DomainError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.generate_chat_stream(messages, options).await
    }
}

/// Runs the use case and returns `(summary text blob, calls, map notes)`,
/// asserting the structured General template came back.
async fn run_summary(utterances: &[String]) -> (String, usize, Vec<String>) {
    let model_path = model_from_env().expect("guarded by caller");
    let llm = Arc::new(CountingLlm {
        inner: Arc::new(LlamaCppLlm::load(&model_path).expect("load gguf")),
        calls: AtomicUsize::new(0),
        map_notes: Mutex::new(Vec::new()),
    });
    let (id, store) = seed_meeting(utterances).await;
    let chars: usize = utterances.iter().map(String::len).sum();
    eprintln!(
        "transcript: {} utterances, {chars} chars, window {} tokens",
        utterances.len(),
        llm.context_window()
    );

    let usecase = SummarizeMeeting::new(llm.clone(), store);
    let started = Instant::now();
    let summary = usecase
        .execute(id, "general", false, Some("es"))
        .await
        .expect("summary must succeed");
    let elapsed = started.elapsed().as_secs_f64();
    let calls = llm.calls.load(Ordering::SeqCst);

    let json = serde_json::to_string_pretty(&summary.content).unwrap();
    eprintln!("elapsed: {elapsed:.1}s, LLM calls: {calls}\n{json}");

    match &summary.content {
        SummaryContent::FreeText { text } => {
            panic!("expected structured General summary, got free text:\n{text}")
        }
        SummaryContent::General { .. } => {}
        other => panic!("expected General template, got {other:?}"),
    }
    let notes = llm.map_notes.lock().unwrap().clone();
    (json, calls, notes)
}

#[tokio::test]
#[ignore = "requires ECHO_LLM_MODEL to point at a .gguf file"]
async fn long_meeting_is_map_reduced_and_keeps_a_mid_transcript_fact_in_map_notes() {
    if model_from_env().is_none() {
        eprintln!("skipping: ECHO_LLM_MODEL not set");
        return;
    }
    // ~40k chars; the fact sits in the middle third.
    let count = 330;
    let utterances = synthetic_dialogue(count, Some(count / 2));
    let chars: usize = utterances.iter().map(String::len).sum();
    assert!(chars >= 35_000, "fixture too short: {chars} chars");

    let (json, calls, notes) = run_summary(&utterances).await;

    // >= 2 map calls + 1 reduce/final call.
    assert!(calls >= 3, "expected map-reduce (>= 3 calls), got {calls}");
    let holders: Vec<usize> = notes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.contains(PLANTED_NUMBER))
        .map(|(i, _)| i + 1)
        .collect();
    eprintln!(
        "map notes: {}; planted fact in map note(s) {holders:?}; in final summary: {}",
        notes.len(),
        json.contains(PLANTED_NUMBER)
    );
    assert!(
        !holders.is_empty(),
        "planted fact ({PLANTED_NUMBER}) lost by the map step; notes:\n{}",
        notes.join("\n=====\n")
    );
}

#[tokio::test]
#[ignore = "requires ECHO_LLM_MODEL to point at a .gguf file"]
async fn short_meeting_is_summarised_in_a_single_pass() {
    if model_from_env().is_none() {
        eprintln!("skipping: ECHO_LLM_MODEL not set");
        return;
    }
    let utterances = synthetic_dialogue(14, Some(7));
    let (_, calls, notes) = run_summary(&utterances).await;
    // A JSON-parse retry may add a second call, but never a map pass.
    assert!(notes.is_empty(), "a meeting that fits must not be mapped");
    assert!(calls <= 2, "expected a single pass, got {calls} calls");
}
