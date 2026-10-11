//! State shared between every llama.cpp-backed adapter in this crate.
//!
//! [`LoadedModel`] holds the heavy bits - the `LlamaBackend` singleton
//! (leaked, lives for the process), the loaded `LlamaModel` (~10 GB
//! for Qwen 3 14B Q4_K_M) and the configuration knobs that decide how
//! the decoding context is constructed. [`SharedModel`] pairs it with the
//! [`Worker`] thread that owns the one reusable context.
//!
//! Every adapter ([`crate::LlamaCppLlm`] for one-shot summaries,
//! [`crate::LlamaCppChat`] for streaming chat) wraps an
//! `Arc<SharedModel>`. Cloning the `Arc` is free, which is what lets
//! the application layer build both ports off the same loaded model:
//!
//! ```text
//! let llm  = LlamaCppLlm::load("qwen3-14b.gguf")?;   // 10 GB resident
//! let chat = llm.chat_handle();                       // 0 extra cost
//! ```
//!
//! ## Concurrency model
//!
//! `LlamaModel` is `Send + Sync`, but a `LlamaContext` borrows it, is not
//! `Send`, and cannot be decoded from two threads at once. The crate
//! therefore keeps **one** context (one KV cache, ~1.2 GB for Qwen3-8B at
//! the default `n_ctx`) alive on a dedicated worker thread and serves
//! summary and chat requests through it one at a time, in arrival order
//! (see [`crate::worker`]). Requests do not pay for context creation, and
//! the context is released after [`CONTEXT_IDLE_TIMEOUT`] without
//! requests so an idle app does not pin the KV cache.
//!
//! Dropping the last `Arc<SharedModel>` closes the worker's queue; the
//! thread drops its context and exits, and frees the model.
//!
//! ## Why this lives in its own module
//!
//! Two reasons. First, `LlamaCppLlm` and `LlamaCppChat` would
//! otherwise have to share state via `pub(crate)` fields on each
//! other, which makes the inheritance fragile. Second, putting the
//! load logic here keeps both adapters trivially testable: a mock /
//! fake adapter that doesn't go through llama.cpp doesn't depend on
//! this module at all.

use std::ops::Deref;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::{AddBos, LlamaModel};

use echo_domain::{estimate_tokens, DomainError};

use crate::llama_engine::LlamaSource;
use crate::worker::{Job, Worker};

/// How long the persistent context may sit unused before it is dropped to
/// free its KV cache. The next request recreates it.
pub(crate) const CONTEXT_IDLE_TIMEOUT: Duration = Duration::from_secs(5 * 60);

// ---------------------------------------------------------------------------
// Think-block stripping (Qwen 3 reasoning suppression)
// ---------------------------------------------------------------------------

/// Remove `<think>…</think>` reasoning blocks that Qwen 3 emits when
/// its internal "thinking" mode is active.  These blocks are never
/// useful to end-users — they contain raw chain-of-thought reasoning
/// that adds noise to summaries and chat replies.
///
/// Handles multiple blocks, nested whitespace, and a trailing
/// unclosed `<think>` (the model hit max_tokens mid-thought).
pub(crate) fn strip_think_blocks(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut remaining = text;

    while let Some(start) = remaining.find("<think>") {
        // Keep everything before <think>
        result.push_str(&remaining[..start]);

        // Find matching </think>
        let after_tag = &remaining[start + "<think>".len()..];
        if let Some(end) = after_tag.find("</think>") {
            remaining = &after_tag[end + "</think>".len()..];
        } else {
            // Unclosed <think> — discard everything after it
            remaining = "";
            break;
        }
    }

    result.push_str(remaining);

    // Trim leading whitespace left by a stripped block at the start
    let trimmed = result.trim_start();
    if trimmed.len() != result.len() {
        return trimmed.to_string();
    }
    result
}

/// Per-process loaded model + the knobs needed to build the decoding
/// context.
pub(crate) struct LoadedModel {
    /// Backend handle. Leaked once per process by [`crate::backend::backend_singleton`].
    pub backend: &'static LlamaBackend,
    /// The loaded model. `Send + Sync`; safe to access from any thread,
    /// including from the worker thread.
    pub model: LlamaModel,
    /// Stable id reported to the application layer. Either the file
    /// stem (e.g. `qwen3-14b-instruct-q4_k_m`) or the override the
    /// caller passed in [`crate::LoadOptions::with_model_id`].
    pub model_id: String,
    /// Path the model was loaded from. Surfaced for telemetry +
    /// "regenerate with model X" UX flows.
    pub model_path: PathBuf,
    /// Context window in tokens, applied to the `LlamaContext` the worker
    /// builds.
    pub n_ctx: u32,
    /// Optional decoder thread count. `None` defers to llama.cpp's
    /// default (typically `num_cpus`). Useful in tests.
    pub n_threads: Option<i32>,
}

impl LoadedModel {
    /// Token count of `text` under the model's own tokenizer, shared by
    /// the `count_tokens` overrides of every adapter.
    ///
    /// `AddBos::Never` mirrors how prompts are tokenized at generation
    /// time, so the count matches what the context will actually hold.
    /// A tokenizer failure falls back to the conservative
    /// [`estimate_tokens`] instead of erroring: a budget query must not
    /// be able to abort a request the estimate could still size safely.
    pub fn count_tokens(&self, text: &str) -> usize {
        token_count_or_estimate(self.model.str_to_token(text, AddBos::Never), text)
    }
}

/// A [`LoadedModel`] plus the worker thread that owns its reusable
/// context. Held inside `Arc<SharedModel>` by every adapter.
///
/// Derefs to the [`LoadedModel`] for read-only access to the model and its
/// settings.
pub(crate) struct SharedModel {
    loaded: Arc<LoadedModel>,
    worker: Worker,
}

impl SharedModel {
    /// Wrap `loaded`; the worker thread starts on the first request.
    pub(crate) fn new(loaded: LoadedModel) -> Self {
        let loaded = Arc::new(loaded);
        let source = Arc::new(LlamaSource::new(Arc::clone(&loaded)));
        Self {
            loaded,
            worker: Worker::new(source, CONTEXT_IDLE_TIMEOUT),
        }
    }

    /// Queue a request on the worker thread.
    pub(crate) fn submit(&self, job: Job) -> Result<(), DomainError> {
        self.worker.submit(job)
    }
}

impl Deref for SharedModel {
    type Target = LoadedModel;

    fn deref(&self) -> &LoadedModel {
        &self.loaded
    }
}

/// Length of a successful tokenization, or the [`estimate_tokens`]
/// fallback when the tokenizer failed. Split out of
/// [`LoadedModel::count_tokens`] so the fallback is testable without a
/// loaded model.
fn token_count_or_estimate<T, E: std::fmt::Display>(
    tokenized: Result<Vec<T>, E>,
    text: &str,
) -> usize {
    match tokenized {
        Ok(tokens) => tokens.len(),
        Err(e) => {
            tracing::warn!(error = %e, "tokenizer failed, using the bytes/2 estimate");
            estimate_tokens(text)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_count_uses_the_tokenizer_result_when_it_succeeds() {
        let tokenized: Result<Vec<u32>, String> = Ok(vec![1, 2, 3, 4]);
        assert_eq!(token_count_or_estimate(tokenized, "irrelevant text"), 4);
    }

    #[test]
    fn token_count_falls_back_to_the_estimate_when_tokenizing_fails() {
        let tokenized: Result<Vec<u32>, String> = Err("invalid utf-8".into());
        // 7 bytes -> ceil(7 / 2) = 4 tokens.
        assert_eq!(token_count_or_estimate(tokenized, "abcdefg"), 4);
    }

    #[test]
    fn strip_single_think_block() {
        let input = "<think>\nI need to analyze this.\n</think>\nHere is the summary.";
        assert_eq!(strip_think_blocks(input), "Here is the summary.");
    }

    #[test]
    fn strip_think_block_preserves_content_before_and_after() {
        let input = "Prefix <think>reasoning</think> suffix";
        assert_eq!(strip_think_blocks(input), "Prefix  suffix");
    }

    #[test]
    fn strip_multiple_think_blocks() {
        let input = "<think>a</think>Hello <think>b</think>world";
        assert_eq!(strip_think_blocks(input), "Hello world");
    }

    #[test]
    fn strip_unclosed_think_block() {
        let input = "<think>model hit max_tokens before closing";
        assert_eq!(strip_think_blocks(input), "");
    }

    #[test]
    fn no_think_block_returns_unchanged() {
        let input = "Just a plain response with no tags.";
        assert_eq!(strip_think_blocks(input), input);
    }

    #[test]
    fn strip_think_block_with_multiline_reasoning() {
        let input = "<think>\nStep 1: read transcript\nStep 2: summarize\nStep 3: format\n</think>\n\n# Meeting Summary\n\nWe discussed...";
        assert_eq!(
            strip_think_blocks(input),
            "# Meeting Summary\n\nWe discussed..."
        );
    }
}
