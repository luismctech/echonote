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
use crate::template::{self, NoThink, PromptProfile, Rendered, TemplateOutput};
use crate::worker::{Job, Worker};

/// How long the persistent context may sit unused before it is dropped to
/// free its KV cache. The next request recreates it.
pub(crate) const CONTEXT_IDLE_TIMEOUT: Duration = Duration::from_secs(5 * 60);

// ---------------------------------------------------------------------------
// Reasoning suppression (Qwen 3 `<think>`, Gemma 4 thought channel)
// ---------------------------------------------------------------------------

/// Qwen 3's reasoning block.
const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
/// Gemma 4's channel: `<|channel>NAME\n…<channel|>`. Reasoning channels are
/// hidden; any other named channel (including `final`, the visible answer)
/// carries user-facing text.
const CHANNEL_OPEN: &str = "<|channel>";
const CHANNEL_CLOSE: &str = "<channel|>";
/// Channel names (lowercase) whose content is reasoning and never shown:
/// Gemma's `thought` and generic spellings (`thinking`, `reasoning`,
/// `analysis`). Only Gemma's `<|channel>NAME\n…<channel|>` syntax is
/// recognised: gpt-oss/harmony frames channels as
/// `<|channel|>analysis<|message|>…<|end|>`, which is not handled here.
const REASONING_CHANNELS: &[&str] = &["thought", "thinking", "reasoning", "analysis"];
/// A channel header (`NAME` up to a newline or closing marker) longer than
/// this is not a name: the channel is treated as plain text so nothing is
/// held back indefinitely. Applied to the name's length alone, never to how
/// much has arrived, so the outcome is the same for any chunking.
const MAX_CHANNEL_NAME: usize = 32;

/// Message of the error for a reply that ended inside a reasoning block
/// before any visible text.
const UNCLOSED_REASONING: &str = "model output ended inside a reasoning block; no answer produced";

/// What a [`ReasoningFilter`] is in the middle of.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Mode {
    /// Ordinary text.
    #[default]
    Text,
    /// Just after `<|channel>`: reading the channel name up to its newline.
    Header,
    /// Inside reasoning; skipping until this closing marker.
    Skip(&'static str),
    /// Inside a channel that is not reasoning: text flows, its closing
    /// marker is dropped.
    Channel,
}

/// A marker found in visible-mode text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Marker {
    Think,
    ChannelOpen,
    ChannelClose,
}

impl Marker {
    fn text(self) -> &'static str {
        match self {
            Self::Think => THINK_OPEN,
            Self::ChannelOpen => CHANNEL_OPEN,
            Self::ChannelClose => CHANNEL_CLOSE,
        }
    }
}

/// Removes reasoning (`<think>…</think>` and Gemma's `thought` channel)
/// from a reply: feed decoder pieces to [`Self::push`], forward what it
/// returns, then forward [`Self::finish`]. A complete reply is one `push`
/// plus the `finish`.
///
/// A marker may arrive split across pieces (`<|chan` + `nel>`), so text
/// that could still turn into a marker is held back until the next piece
/// settles it (at most the longest marker minus one byte); inside a block
/// only a tail that could be a partial closing marker is kept. After a
/// `<|channel>` the name is read before deciding: reasoning names
/// ([`REASONING_CHANNELS`]) are skipped to `<channel|>`, any other channel
/// keeps its content and loses only its markers and name line. The
/// whitespace a block leaves in front of the answer is trimmed.
#[derive(Debug, Default)]
pub(crate) struct ReasoningFilter {
    pending: String,
    mode: Mode,
    /// Trim leading whitespace of the next visible text (set when a block
    /// closes, cleared by the first non-blank visible text).
    trim: bool,
    /// Some non-blank visible text has been returned.
    produced: bool,
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
            match self.mode {
                Mode::Skip(close) => {
                    if let Some(at) = self.pending.find(close) {
                        self.pending.drain(..at + close.len());
                        self.mode = Mode::Text;
                        self.trim = true;
                    } else {
                        let mut cut = self.pending.len().saturating_sub(close.len() - 1);
                        while !self.pending.is_char_boundary(cut) {
                            cut += 1;
                        }
                        self.pending.drain(..cut);
                        return out;
                    }
                }
                Mode::Header => {
                    if !self.read_header() {
                        return out;
                    }
                }
                Mode::Text | Mode::Channel => {
                    if let Some((at, marker)) = self.next_marker() {
                        let visible: String = self.pending.drain(..at).collect();
                        out.push_str(self.visible(&visible));
                        self.pending.drain(..marker.text().len());
                        match marker {
                            Marker::Think => self.mode = Mode::Skip(THINK_CLOSE),
                            Marker::ChannelOpen => self.mode = Mode::Header,
                            Marker::ChannelClose => self.mode = Mode::Text,
                        }
                    } else {
                        let upto = self.pending.len() - self.partial_marker_suffix();
                        let visible: String = self.pending.drain(..upto).collect();
                        out.push_str(self.visible(&visible));
                        return out;
                    }
                }
            }
        }
    }

    /// End of stream: releases held-back text that never became a marker.
    ///
    /// A reply that ends inside a reasoning block drops that tail (logged);
    /// if it produced no visible text at all that is an error, so the
    /// caller never mistakes it for an empty answer.
    pub(crate) fn finish(&mut self) -> Result<String, DomainError> {
        let rest = std::mem::take(&mut self.pending);
        match std::mem::take(&mut self.mode) {
            Mode::Skip(_) | Mode::Header => {
                if self.produced {
                    tracing::warn!("reply ended inside a reasoning block; dropping its tail");
                    Ok(String::new())
                } else {
                    Err(DomainError::LlmFailed(UNCLOSED_REASONING.into()))
                }
            }
            Mode::Text | Mode::Channel => Ok(self.visible(&rest).to_string()),
        }
    }

    /// Resolve a pending channel header; `false` while it is incomplete.
    fn read_header(&mut self) -> bool {
        let newline = self.pending.find('\n');
        let close = self.pending.find(CHANNEL_CLOSE);
        let (end, skip, closed) = match (newline, close) {
            (Some(n), Some(c)) if c < n => (c, CHANNEL_CLOSE.len(), true),
            (Some(n), _) => (n, 1, false),
            (None, Some(c)) => (c, CHANNEL_CLOSE.len(), true),
            (None, None) => {
                // A closing marker starting within the budget is complete once
                // this many bytes have arrived, so deciding earlier would
                // depend on where the pieces split.
                if self.pending.len() > MAX_CHANNEL_NAME + CHANNEL_CLOSE.len() {
                    self.mode = Mode::Channel;
                    return true;
                }
                return false;
            }
        };
        if end > MAX_CHANNEL_NAME {
            // Not a name: keep everything after `<|channel>` as channel text.
            self.mode = Mode::Channel;
            return true;
        }
        let name = self.pending[..end].trim().to_lowercase();
        self.pending.drain(..end + skip);
        let reasoning = REASONING_CHANNELS.contains(&name.as_str());
        self.mode = match (closed, reasoning) {
            (true, _) => Mode::Text,
            (false, true) => Mode::Skip(CHANNEL_CLOSE),
            (false, false) => Mode::Channel,
        };
        if closed {
            self.trim = true;
        }
        true
    }

    /// Markers recognised in the current visible mode.
    fn active_markers(&self) -> &'static [Marker] {
        if self.mode == Mode::Channel {
            &[Marker::Think, Marker::ChannelOpen, Marker::ChannelClose]
        } else {
            &[Marker::Think, Marker::ChannelOpen]
        }
    }

    /// The earliest marker in `pending` as `(byte offset, marker)`.
    fn next_marker(&self) -> Option<(usize, Marker)> {
        self.active_markers()
            .iter()
            .filter_map(|&m| self.pending.find(m.text()).map(|at| (at, m)))
            .min_by_key(|&(at, _)| at)
    }

    /// Length of the longest suffix of `pending` that is a proper prefix of
    /// a marker (and so may still become one).
    fn partial_marker_suffix(&self) -> usize {
        let text = &self.pending;
        let longest = self
            .active_markers()
            .iter()
            .map(|m| m.text().len())
            .max()
            .unwrap_or(1);
        (1..longest)
            .rev()
            .filter(|&k| k <= text.len() && text.is_char_boundary(text.len() - k))
            .find(|&k| {
                let tail = &text[text.len() - k..];
                self.active_markers()
                    .iter()
                    .any(|m| m.text().len() > k && m.text().starts_with(tail))
            })
            .unwrap_or(0)
    }

    fn visible<'a>(&mut self, text: &'a str) -> &'a str {
        let shown = if self.trim {
            let trimmed = text.trim_start();
            if !trimmed.is_empty() {
                self.trim = false;
            }
            trimmed
        } else {
            text
        };
        if shown.chars().any(|c| !c.is_whitespace()) {
            self.produced = true;
        }
        shown
    }
}

/// The one place a reply's visible text and stop sequences are decided,
/// shared by every sink ([`crate::LlamaCppLlm::generate`],
/// [`crate::LlamaCppLlm::generate_stream`] and the chat stream) so they cannot
/// disagree.
///
/// Each piece first goes through a [`ReasoningFilter`], so reasoning is never
/// forwarded and never matches a stop. The visible text is appended to a
/// running total searched for the stop sequences; the stop sequence is never
/// part of the output and ends the reply. A stop may be split across pieces
/// (`<|` then `im_end|>`), so a visible tail that could still become a stop
/// is held back (at most the longest stop minus one byte) until the next
/// piece settles it, and released by [`Self::finish`] if it never did.
pub(crate) struct StreamState {
    /// Visible (reasoning-free) text so far, cut at the stop sequence.
    total: String,
    /// Bytes of `total` already handed out as deltas.
    emitted: usize,
    stops: Vec<String>,
    filter: ReasoningFilter,
    stopped: bool,
}

/// What one piece (or the end of the reply) adds to the visible text.
pub(crate) struct PieceOutcome {
    /// Part of the visible text that should be forwarded. Empty when the
    /// piece is reasoning, held back, or entirely (the tail of) a stop
    /// sequence.
    pub(crate) delta: String,
    /// `true` once a stop sequence matched. The decoder loop should
    /// exit immediately after emitting the (possibly empty) delta.
    pub(crate) stop_hit: bool,
}

impl StreamState {
    /// A state that ends the reply at the first of `stops` (empty ones are
    /// ignored).
    pub(crate) fn new(stops: Vec<String>) -> Self {
        Self {
            total: String::new(),
            emitted: 0,
            stops: stops.into_iter().filter(|s| !s.is_empty()).collect(),
            filter: ReasoningFilter::new(),
            stopped: false,
        }
    }

    /// Consume one decoder piece.
    pub(crate) fn push_piece(&mut self, piece: &str) -> PieceOutcome {
        let visible = self.filter.push(piece);
        self.apply_stops(&visible, true)
    }

    /// End of the reply: releases text held back as a possible marker or
    /// stop that never became one.
    ///
    /// Errors when the reply was only an unclosed reasoning block (see
    /// [`ReasoningFilter::finish`]). A reply that already hit a stop
    /// sequence has nothing left to release.
    pub(crate) fn finish(&mut self) -> Result<PieceOutcome, DomainError> {
        if self.stopped {
            return Ok(PieceOutcome {
                delta: String::new(),
                stop_hit: false,
            });
        }
        let rest = self.filter.finish()?;
        Ok(self.apply_stops(&rest, false))
    }

    fn apply_stops(&mut self, visible: &str, hold_back: bool) -> PieceOutcome {
        self.total.push_str(visible);

        // The *earliest* stop wins so no content beyond it is emitted.
        let cut = self
            .stops
            .iter()
            .filter_map(|stop| self.total.find(stop.as_str()))
            .min();
        if let Some(cut) = cut {
            let delta = self.total[self.emitted..cut].to_string();
            self.total.truncate(cut);
            self.emitted = cut;
            self.stopped = true;
            return PieceOutcome {
                delta,
                stop_hit: true,
            };
        }

        let upto = if hold_back {
            self.total.len() - self.partial_stop_suffix()
        } else {
            self.total.len()
        };
        let delta = self.total[self.emitted..upto].to_string();
        self.emitted = upto;
        PieceOutcome {
            delta,
            stop_hit: false,
        }
    }

    /// Length of the longest unemitted suffix of `total` that is a proper
    /// prefix of a stop (and so may still become one).
    fn partial_stop_suffix(&self) -> usize {
        let text = &self.total[self.emitted..];
        let longest = self.stops.iter().map(String::len).max().unwrap_or(0);
        (1..longest)
            .rev()
            .filter(|&k| k <= text.len() && text.is_char_boundary(text.len() - k))
            .find(|&k| {
                let tail = &text[text.len() - k..];
                self.stops
                    .iter()
                    .any(|stop| stop.len() > k && stop.starts_with(tail))
            })
            .unwrap_or(0)
    }

    /// The visible reply so far, cut at the stop sequence if one matched.
    pub(crate) fn into_text(self) -> String {
        self.total
    }
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
    /// Whether prompts get a leading BOS (see [`LoadedModel::tokenize`]);
    /// probed once at load.
    pub add_bos: bool,
}

impl LoadedModel {
    /// Render `messages` with this model's chat template (see
    /// [`crate::template`]), after stripping the model's special-token
    /// texts from their contents. `no_think` places Qwen's
    /// `/no_think` (last turn for summaries, every turn for chat); Jinja
    /// families are always rendered with thinking off, chat included.
    ///
    /// The prompt is tokenized by [`Self::tokenize`]: special tokens are
    /// parsed and exactly one BOS is present if the vocabulary wants one.
    pub fn render_chat(&self, messages: &[ChatMessage], no_think: NoThink) -> Rendered {
        self.profile
            .render(messages, no_think, |messages| self.apply_template(messages))
    }

    /// Token count of `messages` rendered by [`Self::render_chat`], the
    /// way a request would carry them.
    pub fn count_chat_tokens(&self, messages: &[ChatMessage], no_think: NoThink) -> usize {
        self.count_tokens(&self.render_chat(messages, no_think).text)
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
    /// BOS policy: no special tokens are added by llama.cpp (its single
    /// `add_special` switch would also append an EOS for a GGUF that sets
    /// `add_eos_token`, ending every prompt), and the BOS is prepended here
    /// exactly when the model wants one (`add_bos`, probed once at load by
    /// [`probe_bos_policy`]: the wrapper does not expose the GGUF flag). A prompt that
    /// already starts with a BOS token (a raw prompt carrying `<bos>`) is not
    /// given a second one, and no EOS is ever appended.
    ///
    /// The chat-template path relies on this: llama.cpp strips the template's
    /// leading `<bos>` for such vocabularies (Gemma yes, Qwen no), so a
    /// tokenization without a BOS would leave the prompt without one. Empty
    /// text has no tokens (not a lone BOS).
    pub fn tokenize(&self, text: &str) -> Result<Vec<LlamaToken>, llama_cpp_2::StringToTokenError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        prompt_tokens(self.model.token_bos(), self.add_bos, |add_special| {
            let mode = if add_special {
                AddBos::Always
            } else {
                AddBos::Never
            };
            self.model.str_to_token(text, mode)
        })
    }
}

/// What llama.cpp's own special-token handling does to a tokenization, as
/// observed at load by [`probe_bos_policy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BosPolicy {
    /// A BOS is prepended when special tokens are added.
    pub add_bos: bool,
    /// An EOS is appended too (a GGUF with `add_eos_token`); never wanted
    /// in a prompt, which is why [`prompt_tokens`] avoids `add_special`.
    pub add_eos: bool,
}

/// Derive the vocabulary's [`BosPolicy`] by tokenizing a probe string with
/// and without llama.cpp's special-token additions and comparing the edges.
pub(crate) fn probe_bos_policy(model: &LlamaModel) -> BosPolicy {
    let (bos, eos) = (model.token_bos(), model.token_eos());
    let (Ok(plain), Ok(special)) = (
        model.str_to_token("x", AddBos::Never),
        model.str_to_token("x", AddBos::Always),
    ) else {
        return BosPolicy {
            add_bos: false,
            add_eos: false,
        };
    };
    bos_policy_from(&plain, &special, bos, eos)
}

/// [`probe_bos_policy`]'s comparison, on token lists: `plain` was tokenized
/// without special additions, `special` with them.
fn bos_policy_from<T: Copy + PartialEq>(plain: &[T], special: &[T], bos: T, eos: T) -> BosPolicy {
    BosPolicy {
        add_bos: special.first() == Some(&bos) && plain.first() != Some(&bos),
        add_eos: special.last() == Some(&eos) && plain.last() != Some(&eos),
    }
}

/// Prompt tokens under the BOS policy: tokenize without special additions
/// (`tokenize(false)`; special-token text still parses), then prepend `bos`
/// when `add_bos` and the prompt does not already start with it. An EOS is
/// never appended. Generic over the token type so the policy is testable
/// without a model.
fn prompt_tokens<T: Copy + PartialEq, E>(
    bos: T,
    add_bos: bool,
    tokenize: impl FnOnce(bool) -> Result<Vec<T>, E>,
) -> Result<Vec<T>, E> {
    let mut tokens = tokenize(false)?;
    if add_bos && tokens.first() != Some(&bos) {
        tokens.insert(0, bos);
    }
    Ok(tokens)
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

    /// A whole reply through a fresh [`ReasoningFilter`].
    fn strip_think_blocks(text: &str) -> Result<String, DomainError> {
        let mut filter = ReasoningFilter::new();
        let mut out = filter.push(text);
        out.push_str(&filter.finish()?);
        Ok(out)
    }

    /// [`strip_think_blocks`] for inputs that must succeed.
    fn strip(text: &str) -> String {
        strip_think_blocks(text).expect("strip")
    }

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
        assert_eq!(strip(input), "The summary.");
    }

    #[test]
    fn strip_keeps_a_non_reasoning_channel_and_drops_a_thought_one() {
        let input = "<|channel>thought\nplan<channel|>A <|channel>final\nB<channel|> C";
        assert_eq!(strip(input), "A B C");
    }

    #[test]
    fn strip_mixed_markers_in_order_and_an_unclosed_channel() {
        let input = "<think>a</think>One <|channel>thought\nb<channel|>two <|channel>thought\ncut";
        assert_eq!(strip(input), "One two ");
    }

    // ---- BOS policy ---------------------------------------------------------

    const BOS: i32 = 2;
    const EOS: i32 = 1;

    /// A tokenizer double: `body` tokens, plus what llama.cpp's
    /// `add_special` would add (BOS, and EOS for an `add_eos_token` GGUF).
    /// Records the `add_special` values it was called with.
    fn fake_tokenizer<'a>(
        body: &'a [i32],
        wants_bos: bool,
        wants_eos: bool,
        calls: &'a std::cell::RefCell<Vec<bool>>,
    ) -> impl FnOnce(bool) -> Result<Vec<i32>, String> + 'a {
        move |add_special| {
            calls.borrow_mut().push(add_special);
            let mut out = Vec::new();
            if add_special && wants_bos && body.first() != Some(&BOS) {
                out.push(BOS);
            }
            out.extend_from_slice(body);
            if add_special && wants_eos {
                out.push(EOS);
            }
            Ok(out)
        }
    }

    #[test]
    fn prompt_gets_one_bos_and_no_eos_even_when_the_gguf_adds_eos() {
        let calls = std::cell::RefCell::new(Vec::new());
        let tokens =
            prompt_tokens(BOS, true, fake_tokenizer(&[7, 8, 9], true, true, &calls)).unwrap();
        assert_eq!(tokens, [BOS, 7, 8, 9]);
        // `add_special` is never requested: it is what appends the EOS.
        assert_eq!(*calls.borrow(), [false]);
    }

    #[test]
    fn a_raw_leading_bos_is_not_doubled() {
        let calls = std::cell::RefCell::new(Vec::new());
        let tokens =
            prompt_tokens(BOS, true, fake_tokenizer(&[BOS, 7], true, false, &calls)).unwrap();
        assert_eq!(tokens, [BOS, 7]);
        // Two explicit BOS tokens in the text are the caller's doing and are
        // kept as they are; only the automatic BOS is deduplicated.
        let tokens = prompt_tokens(
            BOS,
            true,
            fake_tokenizer(&[BOS, BOS, 7], true, false, &calls),
        )
        .unwrap();
        assert_eq!(tokens, [BOS, BOS, 7]);
    }

    #[test]
    fn a_vocabulary_without_add_bos_gets_the_tokens_untouched() {
        let calls = std::cell::RefCell::new(Vec::new());
        let tokens =
            prompt_tokens(BOS, false, fake_tokenizer(&[7, 8], false, false, &calls)).unwrap();
        assert_eq!(tokens, [7, 8]);
        // A raw `<bos>` in the text of such a model is the caller's choice.
        let tokens =
            prompt_tokens(BOS, false, fake_tokenizer(&[BOS, 7], false, false, &calls)).unwrap();
        assert_eq!(tokens, [BOS, 7]);
    }

    #[test]
    fn a_tokenizer_error_is_propagated() {
        let err = prompt_tokens(BOS, true, |_| Err::<Vec<i32>, _>("bad".to_string()));
        assert_eq!(err, Err("bad".to_string()));
    }

    #[test]
    fn bos_policy_is_derived_from_the_two_tokenizations() {
        // Gemma-like: BOS added; Qwen-like: nothing added; Llama-2-like
        // with `add_eos_token`: BOS and EOS added.
        let p = bos_policy_from(&[7], &[BOS, 7], BOS, EOS);
        assert_eq!((p.add_bos, p.add_eos), (true, false));
        let p = bos_policy_from(&[7], &[7], BOS, EOS);
        assert_eq!((p.add_bos, p.add_eos), (false, false));
        let p = bos_policy_from(&[7], &[BOS, 7, EOS], BOS, EOS);
        assert_eq!((p.add_bos, p.add_eos), (true, true));
        // A probe that itself starts with BOS says nothing about the flag.
        let p = bos_policy_from(&[BOS, 7], &[BOS, 7], BOS, EOS);
        assert!(!p.add_bos);
    }

    // ---- streaming reasoning filter ----------------------------------------

    /// Feed `pieces` through a fresh filter; returns what each push shows and
    /// the final flush.
    fn filter_run(pieces: &[&str]) -> (Vec<String>, String) {
        let mut f = ReasoningFilter::new();
        let shown = pieces.iter().map(|p| f.push(p)).collect();
        (shown, f.finish().unwrap())
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
    fn filter_suppresses_an_empty_thought_channel_and_reasoning_like_names() {
        assert_eq!(filter_text(&["<|channel>thought\n<channel|>Hola"]), "Hola");
        assert_eq!(
            filter_text(&["<|channel>Thinking\nx<channel|>Hola"]),
            "Hola"
        );
        assert_eq!(
            filter_text(&["<|channel>reasoning\nx<channel|>Hola"]),
            "Hola"
        );
        // A name that arrives split across pieces is still read whole.
        assert_eq!(
            filter_text(&["<|channel>tho", "ught\nx<channel|>Hola"]),
            "Hola"
        );
    }

    #[test]
    fn filter_keeps_the_content_of_a_non_reasoning_channel() {
        // Only the channel markers (and the name line) are dropped: whatever
        // the model put in another channel is user-facing text.
        assert_eq!(
            filter_text(&["<|channel>final\nLa respuesta<channel|> y más"]),
            "La respuesta y más"
        );
        // ... also with markers split across pieces and no closing marker.
        let pieces = ["<|chan", "nel>fin", "al\nLa ", "respu", "esta"];
        assert_eq!(filter_text(&pieces), "La respuesta");
    }

    #[test]
    fn filter_does_not_hold_back_a_channel_without_a_name_line_forever() {
        // No newline within the name budget: not a channel header worth
        // waiting for, so it is treated as a plain channel and flows.
        let long = "x".repeat(200);
        let text = filter_text(&[&format!("<|channel>{long}"), " tail"]);
        assert!(text.ends_with(" tail"), "{text:?}");
    }

    /// A reply through a fresh filter in the given pieces: the visible text
    /// or the error message.
    fn run_pieces(pieces: &[&str]) -> Result<String, String> {
        let mut f = ReasoningFilter::new();
        let mut out = String::new();
        for p in pieces {
            out.push_str(&f.push(p));
        }
        f.finish().map(|t| out + &t).map_err(|e| e.to_string())
    }

    #[test]
    fn filter_output_does_not_depend_on_how_the_reply_is_chunked() {
        let long = "x".repeat(40);
        let mut replies: Vec<String> = vec![
            "<|channel>thought\nplan<channel|>Answer".into(),
            "<|channel>analysis\nplan<channel|>Answer".into(),
            "<|channel>final\nAnswer<channel|> tail".into(),
            "Hi <|channel>thought\nplan<channel|>A <think>b</think>C".into(),
            format!("<|channel>{long}\nbody<channel|> tail"),
            format!("<|channel>{long} no newline here at all"),
            "<|channel>thought\ncut".into(),
        ];
        // Names around the budget, ended by a newline or directly by a close.
        for len in 28..=45 {
            let name = "y".repeat(len);
            replies.push(format!("<|channel>{name}\nbody<channel|> t"));
            replies.push(format!("<|channel>{name}<channel|>t"));
        }
        for reply in &replies {
            let whole = run_pieces(&[reply]);
            let chars: Vec<String> = reply.chars().map(String::from).collect();
            let chars: Vec<&str> = chars.iter().map(String::as_str).collect();
            assert_eq!(run_pieces(&chars), whole, "per-char: {reply:?}");
            for at in (0..=reply.len()).filter(|&i| reply.is_char_boundary(i)) {
                let (a, b) = reply.split_at(at);
                assert_eq!(run_pieces(&[a, b]), whole, "split {at}: {reply:?}");
            }
        }
    }

    #[test]
    fn filter_treats_an_over_long_channel_name_as_a_plain_channel_even_with_a_newline() {
        let long = "x".repeat(40);
        let text = run_pieces(&[&format!("<|channel>{long}\nbody<channel|>!")]).unwrap();
        assert_eq!(text, format!("{long}\nbody!"));
    }

    #[test]
    fn filter_suppresses_a_harmony_analysis_channel_and_keeps_final_and_commentary() {
        // `thought` is Gemma's reasoning channel, `analysis` gpt-oss/harmony's.
        assert_eq!(
            filter_text(&["<|channel>analysis\nplan<channel|>Hola"]),
            "Hola"
        );
        // `final` is the visible answer channel; `commentary` is not reasoning.
        assert_eq!(filter_text(&["<|channel>final\nHola<channel|>"]), "Hola");
        assert_eq!(
            filter_text(&["<|channel>commentary\nHola<channel|>"]),
            "Hola"
        );
    }

    #[test]
    fn filter_reports_an_unclosed_reasoning_block_only_when_nothing_was_visible() {
        let mut f = ReasoningFilter::new();
        f.push("<|channel>thought\nplanning");
        let err = f.finish().unwrap_err();
        assert!(err.to_string().contains("reasoning block"), "{err}");
        // With visible text before it the tail is dropped, not an error.
        let mut f = ReasoningFilter::new();
        assert_eq!(f.push("Hola <think>cut"), "Hola ");
        assert_eq!(f.finish().unwrap(), "");
        // An unclosed channel header at the end counts as unclosed.
        let mut f = ReasoningFilter::new();
        f.push("<|channel>tho");
        assert!(f.finish().is_err());
        // Whitespace alone is not an answer.
        let mut f = ReasoningFilter::new();
        f.push("  \n<think>cut");
        assert!(f.finish().is_err());
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
        assert_eq!(f.finish().unwrap(), "<");
    }

    #[test]
    fn filter_drops_an_unclosed_block_at_the_end_after_visible_text() {
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
        assert_eq!(strip(input), "Here is the summary.");
    }

    #[test]
    fn strip_think_block_preserves_content_before_and_after() {
        let input = "Prefix <think>reasoning</think> suffix";
        // The whitespace a block leaves in front of the text after it is trimmed.
        assert_eq!(strip(input), "Prefix suffix");
    }

    #[test]
    fn strip_multiple_think_blocks() {
        let input = "<think>a</think>Hello <think>b</think>world";
        assert_eq!(strip(input), "Hello world");
    }

    #[test]
    fn an_unclosed_think_block_with_no_other_text_is_an_error() {
        let err = strip_think_blocks("<think>model hit max_tokens before closing").unwrap_err();
        assert!(err.to_string().contains("reasoning block"), "{err}");
    }

    #[test]
    fn an_unclosed_think_block_after_visible_text_is_dropped() {
        assert_eq!(strip("Answer.<think>cut off"), "Answer.");
    }

    #[test]
    fn no_think_block_returns_unchanged() {
        let input = "Just a plain response with no tags.";
        assert_eq!(strip(input), input);
    }

    #[test]
    fn strip_think_block_with_multiline_reasoning() {
        let input = "<think>\nStep 1: read transcript\nStep 2: summarize\nStep 3: format\n</think>\n\n# Meeting Summary\n\nWe discussed...";
        assert_eq!(strip(input), "# Meeting Summary\n\nWe discussed...");
    }
}
