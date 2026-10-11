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
/// `<think>…</think>` and Gemma 4's `<|channel>…<channel|>` (the thought
/// channel is `<|channel>thought\n…`; any channel a template opens before
/// the final answer is likewise not user-facing text).
const THINK_BLOCKS: &[(&str, &str)] = &[("<think>", "</think>"), ("<|channel>", "<channel|>")];

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

/// Streaming counterpart of [`strip_think_blocks`]: feed decoder pieces to
/// [`Self::push`], forward what it returns, then forward [`Self::finish`].
///
/// A marker may arrive split across pieces (`<|chan` + `nel>`), so text
/// that could still turn into an opening marker is held back until the next
/// piece settles it (at most the longest marker minus one byte); inside a
/// block only a tail that could be a partial closing marker is kept. Like
/// [`strip_think_blocks`] it drops an unclosed block at the end and trims
/// the whitespace a block leaves in front of the answer.
#[derive(Debug, Default)]
pub(crate) struct ReasoningFilter {
    pending: String,
    /// Closing marker of the block being skipped, if inside one.
    closing: Option<&'static str>,
    /// Trim leading whitespace of the next visible text (set when a block
    /// closes, cleared by the first non-blank visible text).
    trim: bool,
}

impl ReasoningFilter {
    /// A filter at the start of a reply.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Consume one decoder piece; returns the text now known to be visible.
    pub(crate) fn push(&mut self, piece: &str) -> String {
        self.pending.push_str(piece);
        let mut out = String::new();
        loop {
            if let Some(close) = self.closing {
                if let Some(at) = self.pending.find(close) {
                    self.pending.drain(..at + close.len());
                    self.closing = None;
                    self.trim = true;
                } else {
                    let mut cut = self.pending.len().saturating_sub(close.len() - 1);
                    while !self.pending.is_char_boundary(cut) {
                        cut += 1;
                    }
                    self.pending.drain(..cut);
                    return out;
                }
            } else if let Some((at, open, close)) = THINK_BLOCKS
                .iter()
                .filter_map(|&(open, close)| self.pending.find(open).map(|at| (at, open, close)))
                .min_by_key(|&(at, _, _)| at)
            {
                let visible: String = self.pending.drain(..at).collect();
                out.push_str(self.visible(&visible));
                self.pending.drain(..open.len());
                self.closing = Some(close);
            } else {
                let upto = self.pending.len() - partial_open_suffix(&self.pending);
                let visible: String = self.pending.drain(..upto).collect();
                out.push_str(self.visible(&visible));
                return out;
            }
        }
    }

    /// End of stream: releases held-back text that never became a marker; an
    /// unclosed block yields nothing.
    pub(crate) fn finish(&mut self) -> String {
        let rest = std::mem::take(&mut self.pending);
        if self.closing.take().is_some() {
            return String::new();
        }
        self.visible(&rest).to_string()
    }

    fn visible<'a>(&mut self, text: &'a str) -> &'a str {
        if !self.trim {
            return text;
        }
        let trimmed = text.trim_start();
        if !trimmed.is_empty() {
            self.trim = false;
        }
        trimmed
    }
}

/// Length of the longest suffix of `text` that is a proper prefix of an
/// opening marker (and so may still become one).
fn partial_open_suffix(text: &str) -> usize {
    let longest = THINK_BLOCKS
        .iter()
        .map(|(open, _)| open.len())
        .max()
        .unwrap_or(1);
    (1..longest)
        .rev()
        .filter(|&k| k <= text.len() && text.is_char_boundary(text.len() - k))
        .find(|&k| {
            let tail = &text[text.len() - k..];
            THINK_BLOCKS
                .iter()
                .any(|(open, _)| open.len() > k && open.starts_with(tail))
        })
        .unwrap_or(0)
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
    /// The prompt is tokenized by [`Self::tokenize`]: special tokens are
    /// parsed and exactly one BOS is present if the vocabulary wants one.
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
            // llama.cpp overrides this with the vocabulary's own `add_bos`
            // flag: for a model that wants a BOS (Gemma) the template's
            // `<bos>` is stripped from the text and the tokenizer adds it
            // (see [`LoadedModel::tokenize`]).
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
    /// [`Self::tokenize`] is how prompts are tokenized at generation time,
    /// so the count matches what the context will actually hold.
    /// A tokenizer failure falls back to the conservative
    /// [`estimate_tokens`] instead of erroring: a budget query must not
    /// be able to abort a request the estimate could still size safely.
    pub fn count_tokens(&self, text: &str) -> usize {
        token_count_or_estimate(self.tokenize(text), text)
    }

    /// Tokenize a prompt with special-token parsing and the vocabulary's own
    /// BOS policy: `AddBos::Always` makes llama.cpp prepend BOS exactly when
    /// the model's `tokenizer.ggml.add_bos_token` says so (Gemma yes, Qwen
    /// no), and a prompt that already starts with a BOS token (a raw prompt
    /// carrying `<bos>`) is not given a second one.
    ///
    /// The chat-template path relies on this: llama.cpp strips the template's
    /// leading `<bos>` for such vocabularies, so tokenizing with
    /// `AddBos::Never` would leave a Gemma prompt without any BOS. Empty
    /// text has no tokens (not a lone BOS).
    pub fn tokenize(&self, text: &str) -> Result<Vec<LlamaToken>, llama_cpp_2::StringToTokenError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let tokens = self.model.str_to_token(text, AddBos::Always)?;
        Ok(dedup_leading_bos(tokens, self.model.token_bos()))
    }
}

/// `tokens` without a doubled leading `bos` (only the first of two
/// consecutive BOS tokens at the start is dropped).
fn dedup_leading_bos(mut tokens: Vec<LlamaToken>, bos: LlamaToken) -> Vec<LlamaToken> {
    if tokens.len() >= 2 && tokens[0] == bos && tokens[1] == bos {
        tokens.remove(0);
    }
    tokens
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
    fn a_doubled_leading_bos_collapses_to_one() {
        let bos = LlamaToken(2);
        let t = |ids: &[i32]| ids.iter().map(|&i| LlamaToken(i)).collect::<Vec<_>>();
        assert_eq!(dedup_leading_bos(t(&[2, 2, 7, 8]), bos), t(&[2, 7, 8]));
        assert_eq!(dedup_leading_bos(t(&[2, 7, 2]), bos), t(&[2, 7, 2]));
        assert_eq!(dedup_leading_bos(t(&[7, 2, 2]), bos), t(&[7, 2, 2]));
        assert_eq!(dedup_leading_bos(t(&[2]), bos), t(&[2]));
        assert_eq!(dedup_leading_bos(t(&[]), bos), t(&[]));
    }

    // ---- streaming reasoning filter ----------------------------------------

    /// Feed `pieces` through a fresh filter; returns what each push shows and
    /// the final flush.
    fn filter_run(pieces: &[&str]) -> (Vec<String>, String) {
        let mut f = ReasoningFilter::new();
        let shown = pieces.iter().map(|p| f.push(p)).collect();
        (shown, f.finish())
    }

    fn filter_text(pieces: &[&str]) -> String {
        let (shown, tail) = filter_run(pieces);
        shown.concat() + &tail
    }

    #[test]
    fn filter_passes_plain_text_through_piece_by_piece() {
        let (shown, tail) = filter_run(&["Hola", " mundo"]);
        assert_eq!(shown, ["Hola", " mundo"]);
        assert_eq!(tail, "");
    }

    #[test]
    fn filter_suppresses_a_think_block_and_trims_the_gap_after_it() {
        assert_eq!(
            filter_text(&["<think>", "hidden", "</think>", "\n\nHola"]),
            "Hola"
        );
    }

    #[test]
    fn filter_suppresses_a_gemma_thought_channel() {
        let pieces = [
            "<|channel>",
            "thought",
            "\nplan it",
            "<channel|>",
            "Respuesta",
        ];
        assert_eq!(filter_text(&pieces), "Respuesta");
    }

    #[test]
    fn filter_suppresses_an_empty_gemma_channel_and_any_other_channel_name() {
        assert_eq!(filter_text(&["<|channel>thought\n<channel|>Hola"]), "Hola");
        assert_eq!(
            filter_text(&["<|channel>analysis\nx<channel|>Hola"]),
            "Hola"
        );
    }

    #[test]
    fn filter_handles_markers_split_in_the_middle() {
        let pieces = ["<|chan", "nel>thou", "ght\nx<chan", "nel|>Res", "puesta"];
        assert_eq!(filter_text(&pieces), "Respuesta");
        let pieces = ["<thi", "nk>a</th", "ink>ok"];
        assert_eq!(filter_text(&pieces), "ok");
    }

    #[test]
    fn filter_never_shows_a_split_open_marker_before_it_is_complete() {
        let mut f = ReasoningFilter::new();
        assert_eq!(f.push("Hola <|chan"), "Hola ");
        assert_eq!(f.push("nel>thought\nsecret<channel|>fin"), "fin");
    }

    #[test]
    fn filter_releases_a_held_back_prefix_that_turns_out_to_be_text() {
        let mut f = ReasoningFilter::new();
        assert_eq!(f.push("a <"), "a ");
        assert_eq!(f.push(" b"), "< b");
        // A prefix still pending when the stream ends is plain text.
        assert_eq!(f.push("x <"), "x ");
        assert_eq!(f.finish(), "<");
    }

    #[test]
    fn filter_drops_an_unclosed_block_at_the_end() {
        let (shown, tail) = filter_run(&["Hola <|channel>thought\ncut off"]);
        assert_eq!(shown.concat() + &tail, "Hola ");
    }

    #[test]
    fn filter_handles_several_blocks_and_chat_markers_that_are_not_reasoning() {
        let pieces = [
            "<think>a</think>One ",
            "<|channel>thought\nb<channel|>",
            "two<|im_end|>",
        ];
        assert_eq!(filter_text(&pieces), "One two<|im_end|>");
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
