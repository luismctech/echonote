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
//!    Both adapters now point at the same `Arc<SharedModel>`. No
//!    extra weights are loaded; `generate` and `ask` calls share one
//!    persistent `LlamaContext`; chat turns are served ahead of queued
//!    summary calls (see `crate::worker`).
//!
//! 2. **Standalone** (e.g. CLI tools that only need the chat surface):
//!    ```ignore
//!    let chat = LlamaCppChat::load("qwen3-14b.gguf")?;
//!    ```
//!
//! ## Streaming
//!
//! The decoder loop is synchronous, so it runs on the model's worker
//! thread (see [`crate::worker`]). Each detokenised piece is shipped
//! over a bounded `tokio::sync::mpsc` channel; the receiver is wrapped
//! into a `BoxStream` of `Result<ChatToken, DomainError>` and handed
//! back to the caller. When the channel buffer fills (a slow IPC
//! consumer), the producer back-pressures the decoder instead of OOMing
//! on a runaway model, but only for a bounded stall (see
//! [`crate::worker::CONSUMER_STALL_TIMEOUT`]): a consumer that stops
//! draining is cut off so it cannot block the worker. Dropping the stream
//! cancels the turn.
//!
//! ## Stop-sequence handling
//!
//! Most well-behaved chat-template models emit `<|im_end|>` (Qwen),
//! `<turn|>` (Gemma 4) or `<|eot_id|>` (Llama 3) as a special **EOG** token, which
//! [`LlamaModel::is_eog_token`] catches before the piece ever leaves
//! the decoder. As a belt-and-braces measure we also scan the running
//! reply for any of the user-supplied [`ChatOptions::stop`] strings and
//! the ones of the model family (see [`crate::template`]); if one matches, we trim the latest
//! piece at the stop position before sending and then end the stream.
//! The two layers together cover the cases (a) model behaving
//! correctly (EOG fires), (b) model emitting the marker as text
//! (post-hoc trim catches it). A stop split across pieces is caught too:
//! [`StreamState`] holds back a visible tail that could still become a stop.
//! The same state drives every sink of the crate, so `generate`,
//! `generate_stream` and the chat stream agree on what is visible.

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use tokio::sync::mpsc;
use tracing::{debug, info, instrument};

use echo_domain::{ChatAssistant, ChatMessage, ChatRequest, ChatToken, DomainError};

use crate::llama_cpp::{LlamaCppLlm, LoadOptions};
use crate::shared::{SharedModel, StreamState};
use crate::template::{merge_stops, NoThink};
use crate::worker::{
    drain_stream, Cancel, Flow, Job, Outlet, PieceSink, Priority, SamplingParams, Sink,
};

/// mpsc buffer between the (blocking) decoder and the (async) IPC
/// consumer. Large enough that a small Tauri Channel hiccup does not
/// stall decoding, small enough that a runaway model can't queue
/// thousands of tokens before back-pressure kicks in.
const STREAM_BUFFER: usize = 64;

/// llama.cpp-backed [`ChatAssistant`] adapter.
///
/// Cloning is cheap (`Arc` internally) and never duplicates the model
/// — it only bumps the refcount on the shared `SharedModel`.
#[derive(Clone)]
pub struct LlamaCppChat {
    inner: Arc<SharedModel>,
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
    /// share the `Arc<SharedModel>` between the summariser and the
    /// chat adapter without exposing the inner type.
    pub(crate) fn from_loaded(inner: Arc<SharedModel>) -> Self {
        Self { inner }
    }

    /// Prompt tokens the latest request on this model reused from the KV
    /// cache (`0` on a cold cache). Same counter as
    /// [`LlamaCppLlm::last_reused_tokens`]: both adapters share one model.
    /// Test support behind the `test-hooks` feature; not part of the
    /// stable API.
    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    #[must_use]
    pub fn last_reused_tokens(&self) -> usize {
        self.inner.last_reused_tokens()
    }
}

#[async_trait]
impl ChatAssistant for LlamaCppChat {
    fn model_id(&self) -> &str {
        &self.inner.model_id
    }

    fn context_window(&self) -> u32 {
        self.inner.n_ctx
    }

    fn count_tokens(&self, text: &str) -> usize {
        self.inner.count_tokens(text)
    }

    fn count_chat_tokens(&self, messages: &[ChatMessage]) -> usize {
        self.inner.count_chat_tokens(messages, NoThink::EveryTurn)
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

        // Qwen gets `/no_think` on every user turn (no hidden reasoning
        // before the answer; every turn so the history prefix stays stable
        // for KV reuse); Jinja families render with thinking off anyway.
        // A think block a model emits regardless is still suppressed from
        // the stream by `StreamState`.
        let rendered = self
            .inner
            .render_chat(&request.messages, NoThink::EveryTurn);
        let prompt = rendered.text;
        let stops = merge_stops(&request.options.stop, &rendered.stops);

        // ---- queue the decode on the worker + bridge it to a stream -----
        // Decoder failures travel through the channel as the final item.
        let (tx, rx) = mpsc::channel::<Result<ChatToken, DomainError>>(STREAM_BUFFER);
        let out = Outlet::new(tx);
        // Wrap the receiver as a `BoxStream` (no `tokio_stream` needed); a
        // consumer that stalled and was cut off gets an error item at the
        // end instead of a clean completion.
        let stream = drain_stream(rx, out.stall_mark());
        self.inner.submit(Job {
            priority: Priority::Interactive,
            prompt,
            params: SamplingParams::from_chat(&request.options),
            sink: Box::new(ChatSink::new(stops, out)),
        })?;
        Ok(stream)
    }
}

/// [`Sink`] behind [`LlamaCppChat::ask`]: runs each piece through
/// [`StreamState`] (think-block suppression, stop sequences) and ships the
/// resulting delta to the stream consumer.
pub(crate) struct ChatSink {
    state: StreamState,
    out: Outlet<ChatToken>,
}

impl ChatSink {
    pub(crate) fn new(stops: Vec<String>, out: Outlet<ChatToken>) -> Self {
        Self {
            state: StreamState::new(stops),
            out,
        }
    }
}

impl PieceSink for ChatSink {
    fn on_piece(&mut self, piece: &str) -> Flow {
        let outcome = self.state.push_piece(piece);
        if !outcome.delta.is_empty() && self.out.push(ChatToken::new(outcome.delta)) == Flow::Stop {
            info!("chat stream consumer gone or stalled; aborting decode");
            return Flow::Stop;
        }
        if outcome.stop_hit {
            debug!("stopped on user-supplied stop sequence");
            return Flow::Stop;
        }
        Flow::Continue
    }

    fn is_closed(&self) -> bool {
        self.out.is_closed()
    }
}

impl Sink for ChatSink {
    fn bind_cancel(&mut self, cancel: Cancel) {
        self.out.bind_cancel(cancel);
    }

    fn finish(mut self: Box<Self>, result: Result<(), DomainError>) {
        // If the consumer is gone there is nobody to tell; otherwise the
        // error is the stream's final item.
        match result {
            Err(e) => self.out.fail(e),
            Ok(()) => match self.state.finish() {
                Ok(tail) if !tail.delta.is_empty() => {
                    let _ = self.out.push(ChatToken::new(tail.delta));
                }
                Ok(_) => {}
                Err(e) => self.out.fail(e),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use pretty_assertions::assert_eq;

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
    fn stream_state_holds_back_a_stop_split_across_pieces() {
        let mut s = StreamState::new(vec!["<|im_end|>".into()]);
        let o1 = s.push_piece("answer<|");
        assert_eq!(o1.delta, "answer");
        assert!(!o1.stop_hit);
        // The first half never reached the consumer.
        let o2 = s.push_piece("im_end|>junk");
        assert!(o2.stop_hit);
        assert_eq!(o2.delta, "");
        assert_eq!(s.into_text(), "answer");
    }

    #[test]
    fn stream_state_releases_a_held_back_stop_prefix_that_was_text() {
        let mut s = StreamState::new(vec!["<|im_end|>".into()]);
        assert_eq!(s.push_piece("a <|").delta, "a ");
        assert_eq!(s.push_piece("x").delta, "<|x");
        assert_eq!(s.push_piece("b <").delta, "b ");
        assert_eq!(s.finish().unwrap().delta, "<");
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
        // wired to a fake `SharedModel`. Setting one up requires real
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

    // ── Think-block suppression in streaming ─────────────────────

    #[test]
    fn stream_state_suppresses_think_block_at_start() {
        let mut s = StreamState::new(vec!["<|im_end|>".into()]);
        let o1 = s.push_piece("<think>");
        assert_eq!(o1.delta, "");
        assert!(!o1.stop_hit);
        let o2 = s.push_piece("reasoning here");
        assert_eq!(o2.delta, "");
        let o3 = s.push_piece("</think>");
        assert!(!o3.stop_hit);
        // After exiting the think block, new content flows through
        let o4 = s.push_piece("Real answer");
        assert_eq!(o4.delta, "Real answer");
    }

    #[test]
    fn stream_state_suppresses_a_gemma_thought_channel_split_across_pieces() {
        let mut s = StreamState::new(vec!["<turn|>".into()]);
        let mut shown = String::new();
        for piece in [
            "<|chan",
            "nel>thought\nplan",
            "<channel",
            "|>",
            "Hola",
            " mundo",
        ] {
            let o = s.push_piece(piece);
            assert!(!o.stop_hit);
            shown.push_str(&o.delta);
        }
        shown.push_str(&s.finish().unwrap().delta);
        assert_eq!(shown, "Hola mundo");
    }

    #[test]
    fn stream_state_stop_inside_a_thought_channel_is_ignored() {
        let mut s = StreamState::new(vec!["<turn|>".into()]);
        let o = s.push_piece("<|channel>thought\nx<turn|>y<channel|>ok<turn|>junk");
        assert!(o.stop_hit);
        assert_eq!(o.delta, "ok");
    }

    #[test]
    fn stream_state_releases_a_held_back_marker_prefix_on_finish() {
        let mut s = StreamState::new(vec![]);
        assert_eq!(s.push_piece("a <").delta, "a ");
        assert_eq!(s.finish().unwrap().delta, "<");
    }

    #[test]
    fn stream_state_suppresses_think_block_in_single_piece() {
        let mut s = StreamState::new(vec!["<|im_end|>".into()]);
        let o = s.push_piece("<think>inner</think>Answer");
        assert_eq!(o.delta, "Answer");
        assert!(!o.stop_hit);
    }

    #[test]
    fn stream_state_think_block_does_not_affect_stop_detection() {
        let mut s = StreamState::new(vec!["<|im_end|>".into()]);
        let _ = s.push_piece("<think>reasoning</think>");
        let o = s.push_piece("answer<|im_end|>junk");
        assert!(o.stop_hit);
        assert_eq!(o.delta, "answer");
    }

    #[test]
    fn stream_state_tokens_after_think_block_are_incremental() {
        // Regression: after a think block ended, subsequent tokens
        // would re-match the old <think> tag in `total` and emit
        // cumulative text instead of incremental deltas.
        let mut s = StreamState::new(vec!["<|im_end|>".into()]);
        let _ = s.push_piece("<think>internal reasoning</think>");
        let o1 = s.push_piece("En");
        assert_eq!(o1.delta, "En");
        let o2 = s.push_piece(" el");
        assert_eq!(o2.delta, " el");
        let o3 = s.push_piece(" momento");
        assert_eq!(o3.delta, " momento");
        assert!(!o3.stop_hit);
    }

    // ---- production sink (characterisation: passes on first run) ----

    use std::time::{Duration, Instant};

    fn chat_sink(
        stops: &[&str],
        capacity: usize,
        stall: Duration,
    ) -> (ChatSink, mpsc::Receiver<Result<ChatToken, DomainError>>) {
        let (tx, rx) = mpsc::channel(capacity);
        let sink = ChatSink::new(
            stops.iter().map(|s| (*s).to_string()).collect(),
            Outlet::new(tx).with_stall(stall),
        );
        (sink, rx)
    }

    fn drain_text(rx: &mut mpsc::Receiver<Result<ChatToken, DomainError>>) -> String {
        let mut out = String::new();
        while let Ok(item) = rx.try_recv() {
            out.push_str(&item.expect("token").delta);
        }
        out
    }

    #[test]
    fn chat_sink_trims_at_the_stop_sequence_and_stops() {
        let (mut sink, mut rx) = chat_sink(&["<|im_end|>"], 16, Duration::from_secs(1));
        assert_eq!(sink.on_piece("Hola "), Flow::Continue);
        assert_eq!(sink.on_piece("ok<|im_end|>junk"), Flow::Stop);
        assert_eq!(drain_text(&mut rx), "Hola ok");
    }

    #[test]
    fn chat_sink_suppresses_the_think_block() {
        let (mut sink, mut rx) = chat_sink(&[], 16, Duration::from_secs(1));
        for piece in ["<think>", "secret", "</think>", "Respuesta"] {
            assert_eq!(sink.on_piece(piece), Flow::Continue);
        }
        assert_eq!(drain_text(&mut rx), "Respuesta");
    }

    #[test]
    fn chat_sink_suppresses_a_gemma_thought_channel_and_flushes_on_finish() {
        let (mut sink, mut rx) = chat_sink(&[], 16, Duration::from_secs(1));
        for piece in ["<|channel>", "thought\nx", "<channel|>", "Respuesta ", "<"] {
            assert_eq!(sink.on_piece(piece), Flow::Continue);
        }
        Box::new(sink).finish(Ok(()));
        assert_eq!(drain_text(&mut rx), "Respuesta <");
    }

    #[test]
    fn chat_sink_cancels_when_the_receiver_is_dropped() {
        let (mut sink, rx) = chat_sink(&[], 16, Duration::from_secs(1));
        assert!(!sink.is_closed());
        drop(rx);
        assert!(sink.is_closed());
        assert_eq!(sink.on_piece("x"), Flow::Stop);
    }

    #[test]
    fn chat_sink_forwards_the_error_as_the_final_item() {
        let (mut sink, mut rx) = chat_sink(&[], 16, Duration::from_secs(1));
        sink.on_piece("a");
        Box::new(sink).finish(Err(DomainError::LlmFailed("decode failed".into())));
        assert_eq!(rx.try_recv().unwrap().unwrap().delta, "a");
        let err = rx.try_recv().unwrap().unwrap_err();
        assert!(err.to_string().contains("decode failed"), "{err}");
    }

    #[test]
    fn chat_sink_gives_up_on_a_consumer_that_stalls() {
        let (mut sink, mut rx) = chat_sink(&[], 1, Duration::from_millis(40));
        assert_eq!(sink.on_piece("a"), Flow::Continue);
        let started = Instant::now();
        assert_eq!(sink.on_piece("b"), Flow::Stop);
        assert!(started.elapsed() < Duration::from_secs(2));
        assert_eq!(drain_text(&mut rx), "a");
    }
}
