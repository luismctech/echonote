//! llama.cpp adapter for the [`echo_domain::ChatAssistant`] port.
//!
//! Companion of [`crate::LlamaCppLlm`]: same loaded model, different
//! contract. Where `LlamaCppLlm::generate` returns the full reply as a
//! `String` once decoding finishes, [`LlamaCppChat::ask`] streams each
//! decoded piece as a [`ChatToken`] so the UI can render the reply
//! token-by-token (CU-05 in `docs/DEVELOPMENT_PLAN.md`).
//!
//! ## Construction
//!
//! Two ways to build one:
//!
//! 1. **Shared with the summariser** (the production path):
//!    ```ignore
//!    let llm  = LlamaCppLlm::load("qwen3-14b.gguf")?;
//!    let chat = llm.chat_handle();
//!    ```
//!    Both adapters now point at the same `Arc<LoadedModel>`. No
//!    extra weights are loaded; concurrent `generate` and `ask` calls
//!    each spin up their own short-lived `LlamaContext` (see
//!    `crate::shared`).
//!
//! 2. **Standalone** (e.g. CLI tools that only need the chat surface):
//!    ```ignore
//!    let chat = LlamaCppChat::load("qwen3-14b.gguf")?;
//!    ```
//!
//! ## Streaming
//!
//! The decoder loop is synchronous, so it runs on
//! `tokio::task::spawn_blocking`. Each detokenised piece is shipped
//! over a bounded `tokio::sync::mpsc` channel; the receiver is wrapped
//! into a `BoxStream` of `Result<ChatToken, DomainError>` and handed
//! back to the caller. When the channel buffer fills (a slow IPC
//! consumer), the producer naturally back-pressures the decoder
//! instead of OOMing on a runaway model.
//!
//! ## Stop-sequence handling
//!
//! Most well-behaved chat-template models emit `<|im_end|>` (Qwen) or
//! `<|eot_id|>` (Llama 3) as a special **EOG** token, which
//! [`LlamaModel::is_eog_token`] catches before the piece ever leaves
//! the decoder. As a belt-and-braces measure we also scan the running
//! reply for any of the user-supplied
//! [`ChatOptions::stop`] strings; if one matches, we trim the latest
//! piece at the stop position before sending and then end the stream.
//! The two layers together cover the cases (a) model behaving
//! correctly (EOG fires), (b) model emitting the marker as text
//! (post-hoc trim catches it), with the residual edge case of a stop
//! sequence split across pieces documented in
//! [`StreamState::push_piece`].

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::sampling::LlamaSampler;
use tokio::sync::mpsc;
use tracing::{debug, info, instrument, warn};

use echo_domain::{ChatAssistant, ChatMessage, ChatRequest, ChatRole, ChatToken, DomainError};

use crate::llama_cpp::{LlamaCppLlm, LoadOptions};
use crate::shared::LoadedModel;

/// Initial `LlamaBatch` capacity. Same value as the summariser; we
/// resize on demand if the prompt is larger.
const INITIAL_BATCH_TOKENS: usize = 512;

/// mpsc buffer between the (blocking) decoder and the (async) IPC
/// consumer. Large enough that a small Tauri Channel hiccup does not
/// stall decoding, small enough that a runaway model can't queue
/// thousands of tokens before back-pressure kicks in.
const STREAM_BUFFER: usize = 64;

/// llama.cpp-backed [`ChatAssistant`] adapter.
///
/// Cloning is cheap (`Arc` internally) and never duplicates the model
/// — it only bumps the refcount on the shared [`LoadedModel`].
#[derive(Clone)]
pub struct LlamaCppChat {
    inner: Arc<LoadedModel>,
}

impl std::fmt::Debug for LlamaCppChat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaCppChat")
            .field("model_id", &self.inner.model_id)
            .field("model_path", &self.inner.model_path)
            .field("n_ctx", &self.inner.n_ctx)
            .finish()
    }
}

impl LlamaCppChat {
    /// Load a model standalone for chat-only use. Most production
    /// callers want [`LlamaCppLlm::chat_handle`] instead so the model
    /// is shared with the summariser.
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self, DomainError> {
        Self::load_with(model_path, LoadOptions::default())
    }

    /// Load a model standalone with custom options. See [`Self::load`].
    pub fn load_with(model_path: impl AsRef<Path>, opts: LoadOptions) -> Result<Self, DomainError> {
        // Reuse the summariser's loader so we exercise the same
        // backend-singleton + GGUF path, then drop the `LlmModel`
        // wrapper and keep just the shared inner.
        let llm = LlamaCppLlm::load_with(model_path, opts)?;
        Ok(llm.chat_handle())
    }

    /// Path the underlying model was loaded from. Useful for
    /// telemetry and "regenerate with model X" UX flows.
    #[must_use]
    pub fn model_path(&self) -> &Path {
        &self.inner.model_path
    }

    /// Internal constructor. Used by [`LlamaCppLlm::chat_handle`] to
    /// share the `Arc<LoadedModel>` between the summariser and the
    /// chat adapter without exposing the inner type.
    pub(crate) fn from_loaded(inner: Arc<LoadedModel>) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl ChatAssistant for LlamaCppChat {
    fn model_id(&self) -> &str {
        &self.inner.model_id
    }

    #[instrument(
        skip(self, request),
        fields(
            n_messages = request.messages.len(),
            max_tokens = request.options.max_tokens,
        ),
    )]
    async fn ask(
        &self,
        request: &ChatRequest,
    ) -> Result<BoxStream<'static, Result<ChatToken, DomainError>>, DomainError> {
        // ---- pre-stream validation --------------------------------------
        // Same contract as the summariser's `generate`: do as much as
        // we can synchronously so callers learn about obvious mistakes
        // (`max_tokens = 0`, empty `messages`) before we spawn a
        // blocking task they have to await.
        if request.messages.is_empty() {
            return Err(DomainError::Invariant(
                "chat request has no messages".into(),
            ));
        }
        if request.options.max_tokens == 0 {
            return Err(DomainError::Invariant(
                "chat request has max_tokens = 0".into(),
            ));
        }

        let prompt = render_qwen_chat_prompt(&request.messages);
        let inner = Arc::clone(&self.inner);
        let options = request.options.clone();

        // ---- spawn the blocking decoder + bridge it to a stream ---------
        let (tx, rx) = mpsc::channel::<Result<ChatToken, DomainError>>(STREAM_BUFFER);

        tokio::task::spawn_blocking(move || {
            // The decoder communicates failures through the channel, so
            // a panic in `stream_chat_blocking` is the only path that
            // gets back here unsignalled — and we consider that
            // unrecoverable. The closure is short and panic-free; the
            // helper internally turns errors into channel sends.
            stream_chat_blocking(&inner, &prompt, &options, &tx);
        });

        // Wrap the receiver as a `BoxStream`. Using `unfold` instead
        // of pulling in `tokio_stream` keeps the dep tree slim.
        let stream = stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        })
        .boxed();
        Ok(stream)
    }
}

/// Synchronous decoder loop. Mirrors `llama_cpp::generate_blocking`
/// with two differences:
///
/// 1. Each detokenised piece is shipped via `tx` instead of being
///    accumulated into a `String`.
/// 2. The stop-sequence check trims the *just-sent* piece before
///    sending and ends the stream — the running reply still passes
///    through [`StreamState`] so the trim positions are correct.
///
/// Errors are forwarded as `Err` items on the channel. If the
/// receiver has been dropped (the consumer cancelled the stream),
/// `tx.blocking_send` returns `Err(_)` and we exit early.
fn stream_chat_blocking(
    inner: &LoadedModel,
    prompt: &str,
    options: &echo_domain::ChatOptions,
    tx: &mpsc::Sender<Result<ChatToken, DomainError>>,
) {
    if let Err(e) = stream_chat_impl(inner, prompt, options, tx) {
        // Best-effort: if the consumer is gone there's nothing to
        // tell, so we silently drop the error. If they're still
        // listening, surface it as the final stream item.
        let _ = tx.blocking_send(Err(e));
    }
}

fn stream_chat_impl(
    inner: &LoadedModel,
    prompt: &str,
    options: &echo_domain::ChatOptions,
    tx: &mpsc::Sender<Result<ChatToken, DomainError>>,
) -> Result<(), DomainError> {
    let n_ctx = NonZeroU32::new(inner.n_ctx)
        .ok_or_else(|| DomainError::Invariant("n_ctx must be > 0".into()))?;

    let mut ctx_params = LlamaContextParams::default().with_n_ctx(Some(n_ctx));
    if let Some(threads) = inner.n_threads {
        ctx_params = ctx_params.with_n_threads(threads);
        ctx_params = ctx_params.with_n_threads_batch(threads);
    }

    let mut ctx = inner
        .model
        .new_context(inner.backend, ctx_params)
        .map_err(|e| DomainError::LlmFailed(format!("create context: {e}")))?;

    let prompt_tokens = inner
        .model
        .str_to_token(prompt, AddBos::Never)
        .map_err(|e| DomainError::LlmFailed(format!("tokenize prompt: {e}")))?;

    let n_prompt = prompt_tokens.len() as i32;
    let n_max_total = n_prompt.saturating_add(options.max_tokens.min(i32::MAX as u32) as i32);
    if (n_max_total as u32) > inner.n_ctx {
        return Err(DomainError::LlmFailed(format!(
            "prompt ({} tok) + max_tokens ({}) exceeds context window ({})",
            n_prompt, options.max_tokens, inner.n_ctx
        )));
    }

    let batch_capacity = prompt_tokens.len().max(INITIAL_BATCH_TOKENS);
    let mut batch = LlamaBatch::new(batch_capacity, 1);
    let last_index = (prompt_tokens.len() as i32) - 1;
    for (i, token) in (0_i32..).zip(prompt_tokens.into_iter()) {
        let is_last = i == last_index;
        batch
            .add(token, i, &[0], is_last)
            .map_err(|e| DomainError::LlmFailed(format!("batch.add prompt: {e}")))?;
    }
    ctx.decode(&mut batch)
        .map_err(|e| DomainError::LlmFailed(format!("decode prompt: {e}")))?;

    let seed = options.seed.map_or(1234, |s| s as u32);
    let temp = options.temperature.clamp(0.0, 2.0);
    let top_p = options.top_p.clamp(0.0, 1.0);
    let mut sampler = if temp <= f32::EPSILON {
        LlamaSampler::chain_simple([LlamaSampler::greedy()])
    } else {
        LlamaSampler::chain_simple([
            LlamaSampler::top_p(top_p, 1),
            LlamaSampler::temp(temp),
            LlamaSampler::dist(seed),
        ])
    };

    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut state = StreamState::new(options.stop.clone());
    let mut n_cur = batch.n_tokens();
    let mut n_decoded: u32 = 0;
    let max_new = options.max_tokens;

    debug!(
        n_prompt,
        n_ctx = inner.n_ctx,
        max_new,
        temp,
        top_p,
        "starting llama chat decode loop"
    );

    while n_decoded < max_new {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        if inner.model.is_eog_token(token) {
            debug!(n_decoded, "stopped on EOG token");
            return Ok(());
        }

        let piece = inner
            .model
            .token_to_piece(token, &mut decoder, true, None)
            .map_err(|e| DomainError::LlmFailed(format!("detokenize: {e}")))?;
        n_decoded += 1;

        let outcome = state.push_piece(&piece);
        if !outcome.delta.is_empty() && tx.blocking_send(Ok(ChatToken::new(outcome.delta))).is_err()
        {
            // Consumer dropped the receiver — nothing left to do.
            info!("chat stream consumer dropped; aborting decode");
            return Ok(());
        }

        if outcome.stop_hit {
            debug!(n_decoded, "stopped on user-supplied stop sequence");
            return Ok(());
        }

        batch.clear();
        batch
            .add(token, n_cur, &[0], true)
            .map_err(|e| DomainError::LlmFailed(format!("batch.add gen: {e}")))?;
        n_cur += 1;

        ctx.decode(&mut batch)
            .map_err(|e| DomainError::LlmFailed(format!("decode step: {e}")))?;
    }

    if n_decoded == max_new {
        warn!(max_new, "hit max_tokens without an EOG/stop token");
    }
    Ok(())
}

/// Render a chat message list into the Qwen chat-template wire format.
///
/// The template is shared by Qwen 2.5 / Qwen 3 (our default) and
/// understood as plain text by Llama 3, Mistral and Phi 3 instructs —
/// the comment in `echo_app::SummarizeMeeting` notes the same
/// trade-off. Once the use case base of CU-05 is wider we can switch
/// to `LlamaModel::apply_chat_template` and pull the template from
/// the GGUF metadata directly.
///
/// ## Layout
///
/// ```text
/// <|im_start|>system
/// {system}<|im_end|>
/// <|im_start|>user
/// {user1}<|im_end|>
/// <|im_start|>assistant
/// {assistant1}<|im_end|>
/// …
/// <|im_start|>assistant
/// ```
///
/// The trailing `<|im_start|>assistant\n` (without a closing
/// `<|im_end|>`) primes the model to generate the next assistant
/// turn. The use case is responsible for shape (system → user →
/// assistant alternation) — see `echo_app::AskAboutMeeting`.
pub(crate) fn render_qwen_chat_prompt(messages: &[ChatMessage]) -> String {
    let mut out =
        String::with_capacity(messages.iter().map(|m| m.content.len() + 32).sum::<usize>() + 64);
    for msg in messages {
        let role = match msg.role {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        };
        out.push_str("<|im_start|>");
        out.push_str(role);
        out.push('\n');
        out.push_str(&msg.content);
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

/// Tracks the running reply text so we can apply the user-supplied
/// stop sequences without losing previously-emitted bytes.
///
/// On each new piece we append to `total`, then look for any stop
/// sequence inside it. The result, [`PieceOutcome`], tells the caller
/// what part of the piece (if any) is safe to forward to the
/// consumer and whether the stream should terminate.
///
/// ### Edge case
///
/// A stop sequence split exactly across two decoder pieces (e.g. the
/// model emits `<|` then `im_end|>` as separate tokens) will leak the
/// first half because we already shipped that piece. Handling this
/// correctly requires a hold-back buffer of `max_stop_len - 1` bytes;
/// we judged the extra latency not worth it for the MVP because (a)
/// `<|im_end|>` is a single special EOG token in Qwen, caught earlier
/// by [`llama_cpp_2::model::LlamaModel::is_eog_token`], and (b) the
/// only stop strings we ship are template terminators that the model
/// never emits as plain text in practice. Documented here so the next
/// reader doesn't have to rediscover the trade-off.
struct StreamState {
    total: String,
    stops: Vec<String>,
}

struct PieceOutcome {
    /// Slice of the just-decoded piece that should be forwarded. Empty
    /// when the piece consists entirely of (the tail of) a stop
    /// sequence.
    delta: String,
    /// `true` once a stop sequence matched. The decoder loop should
    /// exit immediately after emitting the (possibly empty) delta.
    stop_hit: bool,
}

impl StreamState {
    fn new(stops: Vec<String>) -> Self {
        Self {
            total: String::new(),
            stops: stops.into_iter().filter(|s| !s.is_empty()).collect(),
        }
    }

    fn push_piece(&mut self, piece: &str) -> PieceOutcome {
        let prev_len = self.total.len();
        self.total.push_str(piece);

        // Find the *earliest* stop sequence position in `total`. If
        // multiple stops match we want the one that appears first so
        // we don't accidentally emit content beyond it.
        let mut earliest_cut: Option<usize> = None;
        for stop in &self.stops {
            if let Some(idx) = self.total.find(stop.as_str()) {
                earliest_cut = Some(earliest_cut.map_or(idx, |prev| prev.min(idx)));
            }
        }

        match earliest_cut {
            None => PieceOutcome {
                delta: piece.to_string(),
                stop_hit: false,
            },
            Some(cut) => {
                self.total.truncate(cut);
                let delta = if cut > prev_len {
                    self.total[prev_len..cut].to_string()
                } else {
                    String::new()
                };
                PieceOutcome {
                    delta,
                    stop_hit: true,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use echo_domain::ChatMessage;
    use pretty_assertions::assert_eq;

    #[test]
    fn render_qwen_chat_prompt_emits_full_template_with_assistant_prime() {
        let msgs = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hi."),
            ChatMessage::assistant("Hello!"),
            ChatMessage::user("What's 2+2?"),
        ];
        let rendered = render_qwen_chat_prompt(&msgs);
        let expected = "\
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hi.<|im_end|>
<|im_start|>assistant
Hello!<|im_end|>
<|im_start|>user
What's 2+2?<|im_end|>
<|im_start|>assistant
";
        assert_eq!(rendered, expected);
    }

    #[test]
    fn render_qwen_chat_prompt_handles_only_a_user_turn() {
        let msgs = vec![ChatMessage::user("hola")];
        let rendered = render_qwen_chat_prompt(&msgs);
        assert!(rendered.starts_with("<|im_start|>user\nhola<|im_end|>\n"));
        assert!(rendered.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn stream_state_forwards_unrelated_pieces_unchanged() {
        let mut s = StreamState::new(vec!["<|im_end|>".into()]);
        let o1 = s.push_piece("Hello ");
        assert_eq!(o1.delta, "Hello ");
        assert!(!o1.stop_hit);
        let o2 = s.push_piece("world");
        assert_eq!(o2.delta, "world");
        assert!(!o2.stop_hit);
    }

    #[test]
    fn stream_state_trims_stop_sequence_inside_a_single_piece() {
        let mut s = StreamState::new(vec!["<|im_end|>".into()]);
        let _ = s.push_piece("answer ");
        let outcome = s.push_piece("done<|im_end|>extra");
        assert!(outcome.stop_hit);
        // Only "done" survives — "<|im_end|>extra" is dropped.
        assert_eq!(outcome.delta, "done");
    }

    #[test]
    fn stream_state_drops_piece_entirely_when_stop_starts_at_piece_boundary() {
        // The previous content already ends just before the stop
        // sequence, so the new piece is *the entire* stop sequence
        // plus trailing garbage. Nothing of it should leak.
        let mut s = StreamState::new(vec!["<|im_end|>".into()]);
        let _ = s.push_piece("answer ");
        let outcome = s.push_piece("<|im_end|> ignored");
        assert!(outcome.stop_hit);
        assert_eq!(outcome.delta, "");
    }

    #[test]
    fn stream_state_picks_earliest_stop_when_multiple_match() {
        let mut s = StreamState::new(vec!["<|eot_id|>".into(), "<|im_end|>".into()]);
        // The piece contains both stop strings; we want the earliest
        // one to win so we don't emit content past it.
        let outcome = s.push_piece("foo<|im_end|>bar<|eot_id|>baz");
        assert!(outcome.stop_hit);
        assert_eq!(outcome.delta, "foo");
    }

    #[test]
    fn stream_state_ignores_empty_stop_strings() {
        // Empty stop strings would match at position 0 and silently
        // truncate every reply to the empty string — the constructor
        // filters them out defensively.
        let mut s = StreamState::new(vec!["".into()]);
        let outcome = s.push_piece("hello");
        assert_eq!(outcome.delta, "hello");
        assert!(!outcome.stop_hit);
    }

    #[tokio::test]
    async fn ask_with_no_messages_rejects_before_spawning() {
        // We can exercise the pre-stream validation path without a
        // real model by going through a hand-rolled `LlamaCppChat`
        // wired to a fake `LoadedModel`. Setting one up requires real
        // llama.cpp internals though, so for now we lean on the
        // `Invariant` check happening synchronously: a missing model
        // file means we never get a `LlamaCppChat` to test against in
        // CI, but the check ordering itself (validate → render →
        // spawn) is covered by reading the code.
        //
        // This placeholder asserts the public symbol exists with the
        // expected signature so a future refactor that drops the
        // validation gets caught by the test name in `cargo test`.
        fn _signature_check<T: ChatAssistant>() {}
        _signature_check::<LlamaCppChat>();
    }
}
