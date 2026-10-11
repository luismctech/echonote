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
//! summary and chat requests through it one at a time, chat first
//! (see [`crate::worker`]). Requests do not pay for context creation, and
//! the context is released after [`CONTEXT_IDLE_TIMEOUT`] without
//! requests so an idle app does not pin the KV cache.
//!
//! Dropping the last `Arc<SharedModel>` shuts the worker down: the running
//! request is aborted, queued ones fail, the thread drops its context and
//! exits, and the model is freed.
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
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::Duration;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::{AddBos, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::openai::OpenAIChatTemplateParams;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::token_type::LlamaTokenAttr;

use echo_domain::{estimate_tokens, ChatMessage, DomainError};

use crate::llama_engine::LlamaSource;
use crate::template::{self, PromptProfile, Rendered, TemplateOutput};
use crate::worker::{Job, Worker};

/// How long the persistent context may sit unused before it is dropped to
/// free its KV cache. The next request recreates it.
pub(crate) const CONTEXT_IDLE_TIMEOUT: Duration = Duration::from_secs(5 * 60);

// ---------------------------------------------------------------------------
// Think-block stripping (Qwen 3 reasoning suppression)
// ---------------------------------------------------------------------------

/// Reasoning-block delimiters, as `(open, close)`: Qwen 3's
/// `<think>…</think>` and Gemma 4's `<|channel>thought…<channel|>`.
const THINK_BLOCKS: &[(&str, &str)] =
    &[("<think>", "</think>"), ("<|channel>thought", "<channel|>")];

/// Remove reasoning blocks (see [`THINK_BLOCKS`]) that a model emits when
/// its internal "thinking" mode is active. They are never useful to
/// end-users: they contain raw chain-of-thought that adds noise to
/// summaries and chat replies. Gemma's are stripped defensively, since the
/// prompt renderer asks for thinking to be off.
///
/// Handles multiple blocks, nested whitespace, and a trailing unclosed
/// block (the model hit max_tokens mid-thought).
pub(crate) fn strip_think_blocks(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut remaining = text;

    while let Some((start, open, close)) = THINK_BLOCKS
        .iter()
        .filter_map(|&(open, close)| remaining.find(open).map(|at| (at, open, close)))
        .min_by_key(|&(at, _, _)| at)
    {
        // Keep everything before the opening marker
        result.push_str(&remaining[..start]);

        // Find the matching closing marker
        let after_tag = &remaining[start + open.len()..];
        if let Some(end) = after_tag.find(close) {
            remaining = &after_tag[end + close.len()..];
        } else {
            // Unclosed block — discard everything after it
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
    /// Prompt family, renderer and special-token texts, all decided once
    /// at load time.
    pub profile: PromptProfile,
    /// The chat template embedded in the GGUF, when it has one.
    pub template: Option<LlamaChatTemplate>,
}

impl LoadedModel {
    /// Render `messages` with this model's chat template (see
    /// [`crate::template`]), after stripping the model's special-token
    /// texts from their contents. `qwen_no_think` appends Qwen's
    /// `/no_think` to the last user turn (summaries); Jinja families are
    /// always rendered with thinking off, chat included.
    ///
    /// The prompt is tokenized with special-token parsing and no extra
    /// BOS (see `LlamaEngine`): a template that writes `<bos>` yields
    /// exactly one BOS token, and one that does not yields none.
    pub fn render_chat(&self, messages: &[ChatMessage], qwen_no_think: bool) -> Rendered {
        self.profile.render(messages, qwen_no_think, |messages| {
            self.apply_template(messages)
        })
    }

    /// Token count of `messages` rendered by [`Self::render_chat`], the
    /// way a request would carry them.
    pub fn count_chat_tokens(&self, messages: &[ChatMessage], qwen_no_think: bool) -> usize {
        self.count_tokens(&self.render_chat(messages, qwen_no_think).text)
    }

    /// Render the probe conversation with the GGUF's Jinja template; the
    /// load-time check behind [`template::Rendering::resolve`].
    pub fn probe_template(&self) -> Result<(), String> {
        let out = self.apply_template(&template::probe_messages())?;
        if out.prompt.contains("probe follow-up") {
            Ok(())
        } else {
            Err("the template dropped the conversation".to_string())
        }
    }

    /// Test support: the prompt the GGUF's own Jinja template renders for
    /// `messages`, whatever the family ([`Self::render_chat`] uses ChatML
    /// for Qwen).
    #[cfg(any(test, feature = "test-hooks"))]
    pub fn jinja_prompt(&self, messages: &[ChatMessage]) -> Result<String, String> {
        self.apply_template(messages).map(|out| out.prompt)
    }

    /// Texts of the control and framing-like user-defined tokens of
    /// `model`'s vocabulary (see [`template::is_template_marker`]).
    /// Enumerated once at load; a token whose attributes llama.cpp-2 cannot
    /// decode is skipped.
    pub fn special_token_texts(model: &LlamaModel) -> Vec<String> {
        let mut texts = Vec::new();
        for id in 0..model.n_vocab() {
            let token = LlamaToken(id);
            let attrs =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| model.token_attr(token)));
            let Ok(attrs) = attrs else { continue };
            let is_control = attrs.contains(LlamaTokenAttr::Control);
            if !is_control && !attrs.contains(LlamaTokenAttr::UserDefined) {
                continue;
            }
            let mut decoder = encoding_rs::UTF_8.new_decoder();
            if let Ok(text) = model.token_to_piece(token, &mut decoder, true, None) {
                if template::is_template_marker(&text, is_control) {
                    texts.push(text);
                }
            }
        }
        texts
    }

    /// Render with the GGUF's Jinja template through llama.cpp's
    /// OpenAI-compatible path, thinking disabled.
    fn apply_template(&self, messages: &[ChatMessage]) -> Result<TemplateOutput, String> {
        let template = self
            .template
            .as_ref()
            .ok_or_else(|| "the model has no chat template".to_string())?;
        let json = template::messages_json(messages);
        let params = OpenAIChatTemplateParams {
            messages_json: &json,
            tools_json: None,
            tool_choice: None,
            json_schema: None,
            grammar: None,
            reasoning_format: None,
            chat_template_kwargs: None,
            add_generation_prompt: true,
            use_jinja: true,
            parallel_tool_calls: false,
            enable_thinking: false,
            // The tokenizer adds no BOS (`AddBos::Never`), so a BOS the
            // template writes must stay in the text.
            add_bos: false,
            add_eos: false,
            parse_tool_calls: false,
        };
        self.model
            .apply_chat_template_oaicompat(template, &params)
            .map(|r| TemplateOutput {
                prompt: r.prompt,
                additional_stops: r.additional_stops,
            })
            .map_err(|e| e.to_string())
    }

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
    /// Prompt tokens the latest request reused from the KV cache; written
    /// by this model's engine only. Read through
    /// [`Self::last_reused_tokens`] (tests and the `test-hooks` feature).
    #[cfg_attr(not(any(test, feature = "test-hooks")), allow(dead_code))]
    reused: Arc<AtomicUsize>,
}

impl SharedModel {
    /// Wrap `loaded`; the worker thread starts on the first request.
    pub(crate) fn new(loaded: LoadedModel) -> Self {
        let loaded = Arc::new(loaded);
        let reused = Arc::new(AtomicUsize::new(0));
        let source = Arc::new(LlamaSource::new(Arc::clone(&loaded), Arc::clone(&reused)));
        Self {
            loaded,
            worker: Worker::new(source, CONTEXT_IDLE_TIMEOUT),
            reused,
        }
    }

    /// Prompt tokens the latest request on this model reused from the KV
    /// cache.
    #[cfg(any(test, feature = "test-hooks"))]
    pub(crate) fn last_reused_tokens(&self) -> usize {
        self.reused.load(std::sync::atomic::Ordering::Relaxed)
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
    fn strip_gemma_thought_channel() {
        let input = "<|channel>thought\nplanning the answer<channel|>The summary.";
        assert_eq!(strip_think_blocks(input), "The summary.");
    }

    #[test]
    fn strip_mixed_markers_in_order_and_an_unclosed_channel() {
        let input = "<think>a</think>One <|channel>thought\nb<channel|>two <|channel>thought\ncut";
        assert_eq!(strip_think_blocks(input), "One two ");
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
