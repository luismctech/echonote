//! Chat-with-transcript port (CU-05).
//!
//! [`ChatAssistant`] is the abstraction the chat use case sits on top
//! of. It accepts a fully assembled list of [`ChatMessage`]s — system
//! prompt, prior turns, current user question — and streams back the
//! assistant's reply as a sequence of [`ChatToken`]s.
//!
//! ## Why a stream
//!
//! CU-05 (`docs/DEVELOPMENT_PLAN.md` §3.1) explicitly requires
//! incremental token rendering: "Sistema streamea la respuesta al
//! usuario (tokens aparecen incrementalmente)". A non-streaming `String`
//! return would force the whole reply to be buffered before the user
//! sees a single character — unacceptable on the 30 B-class models the
//! Quality profile will use, where first-token latency hides multi-second
//! decoding stalls. The port therefore returns a [`BoxStream`] so the IPC
//! layer can pipe the stream straight into a `tauri::Channel<ChatToken>`
//! without any intermediate buffering.
//!
//! ## Why stateless
//!
//! Adapters do **not** keep conversation state. Each call ships the
//! complete message history. Three reasons:
//!
//! 1. **Concurrent meetings.** Two chat panels on two different
//!    meetings must share a single loaded model without serializing on
//!    a per-conversation lock.
//! 2. **History trimming.** When the message list grows past the
//!    model's context window, the use case decides whether to drop
//!    early turns, summarize them, or refuse the call. That policy is
//!    a use-case concern, not the model adapter's.
//! 3. **Symmetry with [`crate::LlmModel`].** The summary port is also
//!    stateless; sharing the same shape keeps both adapters
//!    interchangeable when the use case wants a one-shot answer.
//!
//! ## Why citations live outside the port
//!
//! CU-05 expects answers to cite the segments they came from. The port
//! intentionally does not know about [`crate::SegmentId`]: the model
//! emits free text, and the use case parses citation markers (e.g.
//! `[seg:01HXYZ…]`) post-hoc. Keeping the port narrow lets us swap
//! llama.cpp for a different backend (candle, MLX, a remote API in
//! tests) without re-teaching it the EchoNote-specific citation format.
//!
//! ## What the port guarantees
//!
//! * Adapters MUST honour `options.max_tokens`, `options.temperature`,
//!   `options.top_p`, `options.seed` and `options.stop` with the same
//!   contract as [`crate::GenerateOptions`].
//! * Adapters MUST emit at least one [`ChatToken`] per logical token
//!   the model produced — token coalescing is the use case's job, not
//!   the adapter's.
//! * Adapters MUST end the stream cleanly when the model emits its
//!   end-of-sequence token, when `max_tokens` is hit, or when one of
//!   `options.stop` strings matches. Errors mid-stream are surfaced as
//!   `Err(DomainError::LlmFailed(_))` items and SHOULD terminate the
//!   stream right after.
//! * Adapters MAY return [`DomainError::ModelNotLoaded`] from `ask` (the
//!   call that produces the stream) when the underlying model file is
//!   missing.
//! * Calls are async-safe; adapters are responsible for offloading the
//!   synchronous decoder loop to a blocking thread (typically
//!   `tokio::task::spawn_blocking`).

use async_trait::async_trait;
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};

use crate::DomainError;

/// Author of a single message in a chat exchange.
///
/// The three roles mirror the OpenAI / llama.cpp chat-template
/// vocabulary exactly so the adapter can fold them into `<|im_start|>`,
/// `<s>[INST]` or any other model-specific framing without translation
/// tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    /// Top-of-conversation instructions — typically the transcript
    /// excerpt and the "you are a helpful assistant…" preamble.
    /// At most one per request, conventionally the first message.
    System,
    /// A turn authored by the human user.
    User,
    /// A turn authored by the model in a previous round of the same
    /// conversation. Re-fed verbatim so the model can stay coherent
    /// across turns.
    Assistant,
}

/// A single message inside a [`ChatRequest`].
///
/// `content` is plain UTF-8 text — adapters that need a different chat
/// template format (Qwen's `<|im_start|>system\n…<|im_end|>`, Llama's
/// `[INST] … [/INST]`, …) wrap each `ChatMessage` accordingly. The
/// domain layer stays template-agnostic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct ChatMessage {
    /// Who authored this message.
    pub role: ChatRole,
    /// Plain-text content. Empty strings are tolerated (the use case
    /// occasionally sends empty assistant messages to "prime" the
    /// model into a specific format).
    pub content: String,
}

impl ChatMessage {
    /// Convenience constructor for the very common "build a system /
    /// user message inline" pattern used in tests and prompt assembly.
    #[must_use]
    pub fn new(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }

    /// `system` shorthand. Keeps prompt assembly readable:
    /// `ChatMessage::system("You are EchoNote's assistant…")`.
    #[must_use]
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(ChatRole::System, content)
    }

    /// `user` shorthand.
    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(ChatRole::User, content)
    }

    /// `assistant` shorthand.
    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(ChatRole::Assistant, content)
    }
}

/// Knobs the application layer can pass on every chat call.
///
/// Defaults are tuned for **conversational answers**: higher temperature
/// than the summary port (which biases toward structured JSON output)
/// and a generous token budget to fit multi-paragraph answers. The use
/// case overrides these per call when the user picks a "more focused"
/// or "more creative" preset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatOptions {
    /// Hard cap on tokens emitted in this turn. Adapters MUST stop
    /// streaming once this is reached, even mid-sentence.
    pub max_tokens: u32,
    /// Sampling temperature in `[0.0, 2.0]`. `0.7` is a pragmatic
    /// default for chat — coherent enough to not ramble, loose enough
    /// to phrase the same answer differently across re-rolls.
    pub temperature: f32,
    /// Nucleus-sampling parameter in `(0.0, 1.0]`. `1.0` disables it.
    pub top_p: f32,
    /// Seed for the RNG. `Some(_)` makes inference reproducible across
    /// runs (used heavily in tests); `None` lets the adapter pick a
    /// random one each call.
    pub seed: Option<u64>,
    /// Optional list of strings that, when produced, terminate
    /// decoding for this turn. Useful to stop on a model-specific
    /// end-of-turn marker the adapter does not handle natively.
    pub stop: Vec<String>,
}

impl Default for ChatOptions {
    fn default() -> Self {
        Self {
            max_tokens: 1_024,
            temperature: 0.7,
            top_p: 0.95,
            seed: None,
            stop: Vec::new(),
        }
    }
}

/// Everything the adapter needs to produce one assistant turn.
///
/// Built fresh by the use case on every user submission — never
/// retained inside the adapter (see "Why stateless" in the module
/// docs). The use case is responsible for inserting the system prompt,
/// any transcript context, prior turns, and the new user question, in
/// that order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    /// Full conversation, in chronological order. Adapters MUST treat
    /// this as already-templated content and feed it through their
    /// chat template verbatim.
    pub messages: Vec<ChatMessage>,
    /// Sampling and length knobs for this turn.
    pub options: ChatOptions,
}

impl ChatRequest {
    /// Convenience constructor used heavily in tests and the use case.
    #[must_use]
    pub fn new(messages: Vec<ChatMessage>, options: ChatOptions) -> Self {
        Self { messages, options }
    }
}

/// One incremental chunk of the model's reply.
///
/// Adapters SHOULD emit one `ChatToken` per logical token the
/// underlying model decoded — the IPC layer relays them straight to
/// React, which re-renders on every chunk. Coalescing for performance
/// (e.g. flushing every N tokens) is allowed but should be configured
/// at the use case layer, not silently inside the adapter.
///
/// `delta` may contain partial UTF-8 only at multi-byte boundaries the
/// adapter splits across tokens — callers that re-assemble the reply
/// MUST tolerate non-UTF-8-safe slicing in the middle of a stream and
/// only validate at the end. In practice every llama.cpp tokenizer we
/// target emits whole code points per token, so this is a documented
/// edge-case rather than a routine concern.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatToken {
    /// The text added to the assistant reply by this token.
    pub delta: String,
}

impl ChatToken {
    /// Build a token from any string-like value. Convenience for tests
    /// and adapters that already own a `String`.
    #[must_use]
    pub fn new(delta: impl Into<String>) -> Self {
        Self {
            delta: delta.into(),
        }
    }
}

/// Async streaming chat port. Adapters wrap llama.cpp (today: pending
/// `echo_llm::LlamaCppChat`), candle, MLX, or any other engine and
/// expose a uniform "messages in, tokens out" contract.
///
/// Implementors are expected to be cheap to clone (`Arc` internally)
/// because the use-case layer keeps a single instance behind
/// `Arc<dyn ChatAssistant>` and shares it across all chat panels.
#[async_trait]
pub trait ChatAssistant: Send + Sync {
    /// Short, stable identifier of the model currently loaded — stored
    /// on each persisted chat turn for provenance, the same way
    /// [`crate::LlmModel::model_id`] does for summaries. Examples:
    /// `"qwen3-14b-instruct-q4_k_m"`, `"llama-3.2-3b-instruct-q4_k_m"`.
    fn model_id(&self) -> &str;

    /// Run one chat turn and return a stream of [`ChatToken`]s.
    ///
    /// The returned stream:
    ///
    /// - emits at least one token per decoded model token;
    /// - ends cleanly on EOS, on `max_tokens`, or when a `stop` string
    ///   is matched (the matching string is **not** included in the
    ///   final delta);
    /// - surfaces mid-decode failures as `Err(DomainError::LlmFailed)`
    ///   items and terminates immediately afterwards.
    ///
    /// `ask` itself returns `Err` only for failures detected before
    /// decoding starts (model file missing → [`DomainError::ModelNotLoaded`],
    /// invalid `request` → [`DomainError::Invariant`]).
    async fn ask(
        &self,
        request: &ChatRequest,
    ) -> Result<BoxStream<'static, Result<ChatToken, DomainError>>, DomainError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream::{self, StreamExt};
    use pretty_assertions::assert_eq;
    use std::sync::{Arc, Mutex};

    #[test]
    fn default_chat_options_are_conversational() {
        // Pin the defaults: if someone bumps temperature to 0.2 the
        // chat will start sounding like a JSON dump (compare with
        // `GenerateOptions::default` which intentionally biases the
        // *summary* port toward structured output). Make any future
        // change explicit by failing this test.
        let opts = ChatOptions::default();
        assert_eq!(opts.max_tokens, 1_024);
        assert!((opts.temperature - 0.7).abs() < f32::EPSILON);
        assert!((opts.top_p - 0.95).abs() < f32::EPSILON);
        assert!(opts.seed.is_none());
        assert!(opts.stop.is_empty());
    }

    #[test]
    fn chat_request_round_trips_through_serde_with_camelcase_options() {
        // The IPC layer ships `ChatRequest` from React to Rust on
        // every user submission, so make sure the wire format keeps
        // its camelCase contract.
        let req = ChatRequest::new(
            vec![
                ChatMessage::system("You are EchoNote's assistant."),
                ChatMessage::user("¿Cuál fue la decisión sobre el roadmap?"),
            ],
            ChatOptions {
                max_tokens: 256,
                temperature: 0.0,
                top_p: 1.0,
                seed: Some(7),
                stop: vec!["<|im_end|>".into()],
            },
        );

        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"maxTokens\":256"), "got: {json}");
        assert!(json.contains("\"topP\":1.0"), "got: {json}");
        assert!(json.contains("\"role\":\"system\""), "got: {json}");
        assert!(json.contains("\"role\":\"user\""), "got: {json}");

        let back: ChatRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back, req);
    }

    #[test]
    fn chat_role_lowercases_on_the_wire() {
        // OpenAI / llama.cpp templates expect lowercase role names.
        // Pin the serialization so a future `derive(Deserialize)` rename
        // does not silently break adapters that pass the field through.
        assert_eq!(
            serde_json::to_string(&ChatRole::System).unwrap(),
            "\"system\""
        );
        assert_eq!(serde_json::to_string(&ChatRole::User).unwrap(), "\"user\"");
        assert_eq!(
            serde_json::to_string(&ChatRole::Assistant).unwrap(),
            "\"assistant\""
        );
    }

    #[test]
    fn chat_token_round_trips_with_camelcase() {
        // Tokens cross the IPC boundary on every model step; the
        // shape has to stay tiny and stable.
        let token = ChatToken::new("Hola");
        let json = serde_json::to_string(&token).unwrap();
        assert_eq!(json, "{\"delta\":\"Hola\"}");
        let back: ChatToken = serde_json::from_str(&json).unwrap();
        assert_eq!(back, token);
    }

    /// Minimal in-process [`ChatAssistant`] used both by the tests in
    /// this module and (re-exported) by use-case tests in `echo-app`.
    /// Replays a fixed list of tokens for any request and records every
    /// request it received, so callers can assert both on the produced
    /// stream and on what reached the adapter.
    pub struct MockChatAssistant {
        tokens: Vec<ChatToken>,
        recorded: Arc<Mutex<Vec<ChatRequest>>>,
    }

    impl MockChatAssistant {
        pub fn new(tokens: Vec<ChatToken>) -> Self {
            Self {
                tokens,
                recorded: Arc::new(Mutex::new(Vec::new())),
            }
        }

        pub fn recorded(&self) -> Vec<ChatRequest> {
            self.recorded.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl ChatAssistant for MockChatAssistant {
        fn model_id(&self) -> &str {
            "mock-chat"
        }

        async fn ask(
            &self,
            request: &ChatRequest,
        ) -> Result<BoxStream<'static, Result<ChatToken, DomainError>>, DomainError> {
            self.recorded.lock().unwrap().push(request.clone());
            let tokens = self.tokens.clone();
            Ok(stream::iter(tokens.into_iter().map(Ok)).boxed())
        }
    }

    #[tokio::test]
    async fn ask_returns_full_stream_in_order() {
        // The use case will collect the stream into the assistant's
        // reply; the order of deltas must match the order the adapter
        // produced them, with no reordering and no swallowed items.
        let assistant = MockChatAssistant::new(vec![
            ChatToken::new("Hola"),
            ChatToken::new(", "),
            ChatToken::new("¿qué tal?"),
        ]);

        let request = ChatRequest::new(vec![ChatMessage::user("hi")], ChatOptions::default());

        let stream = assistant.ask(&request).await.expect("ask");
        let collected: Vec<_> = stream.collect().await;
        let deltas: Vec<String> = collected
            .into_iter()
            .map(|t| t.expect("token").delta)
            .collect();

        assert_eq!(deltas, vec!["Hola", ", ", "¿qué tal?"]);
    }

    #[tokio::test]
    async fn ask_records_the_request_for_inspection() {
        // The adapter is stateless, so the use case re-passes the full
        // history every turn. The mock's `recorded()` accessor is what
        // use-case tests will use to assert "the system prompt was
        // built correctly" and "history was trimmed before reaching the
        // model".
        let assistant = MockChatAssistant::new(vec![ChatToken::new("ok")]);
        let request = ChatRequest::new(
            vec![ChatMessage::system("ctx"), ChatMessage::user("¿pregunta?")],
            ChatOptions::default(),
        );

        let _ = assistant.ask(&request).await.expect("ask");
        let recorded = assistant.recorded();

        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0], request);
    }

    #[tokio::test]
    async fn empty_token_list_yields_an_empty_stream_not_an_error() {
        // A model that immediately emits EOS (e.g. when the user
        // submits a message that fully matches a stop sequence) is
        // valid. The stream just terminates without items. Use cases
        // must handle this without surfacing it as a failure.
        let assistant = MockChatAssistant::new(vec![]);
        let request = ChatRequest::new(vec![ChatMessage::user("")], ChatOptions::default());

        let stream = assistant.ask(&request).await.expect("ask");
        let collected: Vec<_> = stream.collect().await;
        assert!(collected.is_empty());
    }
}
