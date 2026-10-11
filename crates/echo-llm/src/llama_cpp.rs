//! llama.cpp adapter via [`llama-cpp-2`].
//!
//! Loads a `.gguf` model into memory once, then serves
//! [`echo_domain::LlmModel::generate`] calls through one persistent
//! decoding context owned by a dedicated worker thread (see
//! [`crate::worker`]). The context, and with it the ~1.2 GB KV cache, is
//! created by the first request, reused by the following ones and dropped
//! after five minutes without requests; the next request recreates it.
//!
//! ## Threading model
//!
//! * `LlamaBackend::init()` is a process-wide singleton - guarded by a
//!   `OnceLock` so multiple `LlamaCppLlm::load` calls (tests + runtime)
//!   don't trip the "backend already initialised" error.
//! * The model is loaded once into an `Arc<SharedModel>` and shared
//!   across every generation request - and across every adapter
//!   built from it (see [`Self::chat_handle`]).
//! * A `LlamaContext` is not `Send`, so the worker thread creates it and
//!   serves every request (summary and chat alike) one at a time; queued
//!   chat turns go ahead of queued summary calls, a running call is never
//!   preempted. The async side only queues a job and awaits its
//!   output, so a long summary doesn't stall the rest of the runtime.
//!   A caller that drops its stream or future cancels the job: decoding
//!   stops at the next token and the queue moves on. A caller that stops
//!   draining its stream is cut off after a bounded stall and, once it
//!   resumes, sees an error item rather than a clean end.
//! * The worker keeps the tokens resident in the KV cache and, for each
//!   request, only evaluates the part of the prompt after the first
//!   difference with the previous one (see [`crate::llama_engine`]): a
//!   follow-up chat turn skips the system prompt and transcript it shares
//!   with the turn before.

use std::path::Path;
use std::pin::pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::{self, BoxStream};
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, info, instrument};

use echo_domain::{ChatMessage, DomainError, GenerateOptions, LlmModel};

use crate::backend::backend_singleton;
use crate::llama_cpp_chat::LlamaCppChat;
use crate::shared::{LoadedModel, SharedModel};
use crate::template::{merge_stops, PromptFormat, PromptProfile, Rendering};
use crate::worker::{
    drain_stream, Cancel, Flow, Job, Outlet, PieceSink, Priority, SamplingParams, Sink,
};

/// Default context window when the caller does not override it. Both
/// Qwen 3 8B and 14B ship with `n_ctx_train = 32_768`; 8192 holds the
/// system prompt, chat history and 1 k output tokens next to roughly
/// 20 minutes of Spanish speech (~3.5 chars/token). Summaries of longer
/// meetings are split by the application layer (map-reduce) against
/// [`echo_domain::LlmModel::context_window`]; the chat sends the whole
/// transcript when it fits and, otherwise, the excerpts retrieved for the
/// question.
///
/// The KV cache is the cost that scales with it, and it stays allocated
/// while the persistent context lives (from the first request until five
/// minutes without requests). Computed (not
/// measured) from the published Qwen3-8B config — 36 layers, 8 KV
/// heads, head_dim 128, f16 keys and values — it takes
/// 36 * 8 * 128 * 2 * 2 B = 144 KiB per token, about 1.2 GB per
/// context at 8,192 tokens (the figure scales with the model: Qwen3-14B
/// has more layers). Callers that need more (or want to save RAM) can
/// pass `LlamaCppLlm::load_with(path, opts.with_n_ctx(...))`.
const DEFAULT_N_CTX: u32 = 8_192;

/// Number of model layers to offload to the GPU when Metal/CUDA is
/// available. `999` means "all of them"; llama.cpp clamps internally.
const DEFAULT_N_GPU_LAYERS: u32 = 999;

/// mpsc buffer for streaming generation. Same reasoning as
/// `llama_cpp_chat::STREAM_BUFFER`.
const STREAM_BUFFER: usize = 64;

/// Configurable load knobs for [`LlamaCppLlm::load_with`].
///
/// Sensible defaults are encoded in [`LoadOptions::default`]; the
/// builder methods exist so the CLI / config layer can override them
/// without forcing every caller to spell out every field.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadOptions {
    /// Context window in tokens. Larger values cost more KV-cache RAM
    /// proportional to model size. See [`DEFAULT_N_CTX`].
    pub n_ctx: u32,
    /// How many transformer layers to push onto the GPU. Ignored on
    /// CPU-only builds.
    pub n_gpu_layers: u32,
    /// Override the model identifier reported by
    /// [`LlmModel::model_id`]. When `None`, the file stem of the
    /// `.gguf` path is used (e.g. `qwen2.5-7b-instruct-q4_k_m`).
    pub model_id: Option<String>,
    /// Optional thread count for the decoder. `None` lets llama.cpp
    /// auto-pick (typically `num_cpus`). Useful in tests to keep
    /// runs deterministic.
    pub n_threads: Option<i32>,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            n_ctx: DEFAULT_N_CTX,
            n_gpu_layers: DEFAULT_N_GPU_LAYERS,
            model_id: None,
            n_threads: None,
        }
    }
}

impl LoadOptions {
    /// Override the context window in tokens.
    #[must_use]
    pub fn with_n_ctx(mut self, n_ctx: u32) -> Self {
        self.n_ctx = n_ctx;
        self
    }

    /// Override how many transformer layers run on the GPU.
    #[must_use]
    pub fn with_n_gpu_layers(mut self, n: u32) -> Self {
        self.n_gpu_layers = n;
        self
    }

    /// Pin a custom identifier on the loaded model (e.g. for tests).
    #[must_use]
    pub fn with_model_id(mut self, id: impl Into<String>) -> Self {
        self.model_id = Some(id.into());
        self
    }

    /// Override the decoder thread count.
    #[must_use]
    pub fn with_n_threads(mut self, n: i32) -> Self {
        self.n_threads = Some(n);
        self
    }
}

/// llama.cpp-backed [`LlmModel`] adapter for one-shot generation
/// (summaries, structured-JSON prompts, …).
///
/// Cloning is cheap (`Arc` internally); share one instance across the
/// process via the application state. To serve the chat use case
/// from the same loaded model, call [`Self::chat_handle`].
#[derive(Clone)]
pub struct LlamaCppLlm {
    inner: Arc<SharedModel>,
}

impl std::fmt::Debug for LlamaCppLlm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaCppLlm")
            .field("model_id", &self.inner.model_id)
            .field("model_path", &self.inner.model_path)
            .field("n_ctx", &self.inner.n_ctx)
            .finish()
    }
}

impl LlamaCppLlm {
    /// Load a model with default options. Convenience over
    /// [`LlamaCppLlm::load_with`].
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self, DomainError> {
        Self::load_with(model_path, LoadOptions::default())
    }

    /// Load a `.gguf` model from disk.
    ///
    /// Returns [`DomainError::ModelNotLoaded`] when the file is
    /// missing or llama.cpp rejects it. Loading is heavy (memory-maps
    /// the file and warms up Metal/CPU kernels) — call this once and
    /// share the resulting handle.
    #[instrument(skip_all, fields(path = %model_path.as_ref().display()))]
    pub fn load_with(model_path: impl AsRef<Path>, opts: LoadOptions) -> Result<Self, DomainError> {
        let path = model_path.as_ref();
        if !path.exists() {
            return Err(DomainError::ModelNotLoaded(format!(
                "{} does not exist",
                path.display()
            )));
        }

        let backend = backend_singleton()?;

        let model_id = opts.model_id.clone().unwrap_or_else(|| {
            path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string()
        });

        info!(path = %path.display(), model_id = %model_id, "loading llama.cpp model");
        let model_params = pin!(LlamaModelParams::default().with_n_gpu_layers(opts.n_gpu_layers));
        let model = LlamaModel::load_from_file(backend, path, &model_params)
            .map_err(|e| DomainError::ModelNotLoaded(format!("llama.cpp: {e}")))?;

        let architecture = model.meta_val_str("general.architecture").ok();
        let template = model.chat_template(None).ok();
        let format = PromptFormat::detect(
            architecture.as_deref(),
            template.as_ref().and_then(|t| t.to_str().ok()),
        );
        let specials = LoadedModel::special_token_texts(&model);
        let mut loaded = LoadedModel {
            backend,
            model,
            model_id,
            model_path: path.to_path_buf(),
            n_ctx: opts.n_ctx,
            n_threads: opts.n_threads,
            profile: PromptProfile::new(format, Rendering::ChatMl, specials),
            template,
        };
        // Decided once, here: a template that cannot render the probe is
        // never tried again on a request.
        loaded.profile.rendering = Rendering::resolve(format, || loaded.probe_template());
        info!(
            architecture = architecture.as_deref().unwrap_or("unknown"),
            has_template = loaded.template.is_some(),
            ?format,
            rendering = ?loaded.profile.rendering,
            special_tokens = loaded.profile.specials.len(),
            "detected prompt format"
        );

        Ok(Self {
            inner: Arc::new(SharedModel::new(loaded)),
        })
    }

    /// Path the model was loaded from. Useful for telemetry and
    /// "regenerate with model X" UX flows.
    #[must_use]
    pub fn model_path(&self) -> &Path {
        &self.inner.model_path
    }

    /// Build a chat-streaming adapter that serves the
    /// [`echo_domain::ChatAssistant`] port off of **the same** loaded
    /// model. Cloning the underlying `Arc` is the only cost — no
    /// extra weights are loaded, no background work is started.
    ///
    /// Per `docs/SPRINT-1-STATUS.md` §8.3, summaries and chat are
    /// expected to share the Qwen 3 14B context. This is the entry
    /// point that actually wires that up: at startup, `src-tauri`
    /// loads the model once and dependency-injects both ports
    /// (`Arc<LlamaCppLlm>` + `Arc<LlamaCppChat>`) into the application
    /// layer.
    #[must_use]
    pub fn chat_handle(&self) -> LlamaCppChat {
        LlamaCppChat::from_loaded(Arc::clone(&self.inner))
    }

    /// Prompt tokens the latest request on this model reused from the KV
    /// cache (`0` on a cold cache).
    ///
    /// Scoped to this model, so other models or tests cannot disturb it.
    /// Test support for the gated real-model suites, behind the
    /// `test-hooks` feature; not part of the stable API.
    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    #[must_use]
    pub fn last_reused_tokens(&self) -> usize {
        self.inner.last_reused_tokens()
    }

    /// Prompt family detected from the GGUF (`"Qwen"`, `"Gemma4"` or
    /// `"Other"`). Test support behind the `test-hooks` feature.
    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    #[must_use]
    pub fn prompt_format_name(&self) -> String {
        format!("{:?}", self.inner.profile.format)
    }

    /// How prompts are rendered for this model, decided once at load:
    /// `Jinja` or `ChatMl` (always the case for Qwen, or when the GGUF's
    /// template failed the load-time probe). Test support behind the
    /// `test-hooks` feature.
    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    #[must_use]
    pub fn prompt_rendering_name(&self) -> String {
        format!("{:?}", self.inner.profile.rendering)
    }

    /// How many special-token texts are stripped from message contents.
    /// Test support behind `test-hooks`.
    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    #[must_use]
    pub fn special_token_count(&self) -> usize {
        self.inner.profile.specials.len()
    }

    /// The prompt [`LlmModel::generate_chat`] would send for `messages`
    /// (reasoning off) and the stop sequences it adds. Test support behind
    /// the `test-hooks` feature.
    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    #[must_use]
    pub fn rendered_chat_prompt(&self, messages: &[ChatMessage]) -> (String, Vec<String>) {
        let r = self.inner.render_chat(messages, true);
        (r.text, r.stops)
    }

    /// The prompt the GGUF's own Jinja template renders for `messages`,
    /// independent of the family. Test support behind the `test-hooks`
    /// feature.
    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    pub fn jinja_chat_prompt(&self, messages: &[ChatMessage]) -> Result<String, String> {
        self.inner.jinja_prompt(messages)
    }
}

impl LlamaCppLlm {
    /// Render `messages` for this model (reasoning switched off) and add
    /// the model's own stop sequences to `options`.
    fn chat_request(
        &self,
        messages: &[ChatMessage],
        options: &GenerateOptions,
    ) -> Result<(String, GenerateOptions), DomainError> {
        if messages.is_empty() {
            return Err(DomainError::Invariant("chat prompt has no messages".into()));
        }
        let rendered = self.inner.render_chat(messages, true);
        let options = GenerateOptions {
            stop: merge_stops(&options.stop, &rendered.stops),
            ..options.clone()
        };
        Ok((rendered.text, options))
    }
}

#[async_trait]
impl LlmModel for LlamaCppLlm {
    fn model_id(&self) -> &str {
        &self.inner.model_id
    }

    fn context_window(&self) -> u32 {
        self.inner.n_ctx
    }

    fn count_tokens(&self, text: &str) -> usize {
        self.inner.count_tokens(text)
    }

    #[instrument(skip(self, prompt, options), fields(prompt_chars = prompt.len(), max_tokens = options.max_tokens))]
    async fn generate(
        &self,
        prompt: &str,
        options: &GenerateOptions,
    ) -> Result<String, DomainError> {
        if prompt.is_empty() {
            return Ok(String::new());
        }

        // llama.cpp's decode loop is CPU/GPU-bound and synchronous; the
        // worker thread runs it so other IPC handlers stay responsive.
        let (done, rx) = oneshot::channel();
        self.inner.submit(Job {
            priority: Priority::Background,
            prompt: prompt.to_string(),
            params: SamplingParams::from_generate(options),
            sink: Box::new(CollectSink {
                output: String::new(),
                stops: options.stop.clone(),
                done,
            }),
        })?;
        rx.await
            .map_err(|_| DomainError::LlmFailed("generation worker dropped the request".into()))?
    }

    async fn generate_chat(
        &self,
        messages: &[ChatMessage],
        options: &GenerateOptions,
    ) -> Result<String, DomainError> {
        let (prompt, options) = self.chat_request(messages, options)?;
        self.generate(&prompt, &options).await
    }

    async fn generate_chat_stream(
        &self,
        messages: &[ChatMessage],
        options: &GenerateOptions,
    ) -> Result<BoxStream<'static, Result<String, DomainError>>, DomainError> {
        let (prompt, options) = self.chat_request(messages, options)?;
        self.generate_stream(&prompt, &options).await
    }

    fn count_chat_tokens(&self, messages: &[ChatMessage]) -> usize {
        self.inner.count_chat_tokens(messages, true)
    }

    #[instrument(skip(self, prompt, options), fields(prompt_chars = prompt.len(), max_tokens = options.max_tokens))]
    async fn generate_stream(
        &self,
        prompt: &str,
        options: &GenerateOptions,
    ) -> Result<BoxStream<'static, Result<String, DomainError>>, DomainError> {
        if prompt.is_empty() {
            return Ok(Box::pin(stream::once(async { Ok(String::new()) })));
        }

        let (tx, rx) = mpsc::channel::<Result<String, DomainError>>(STREAM_BUFFER);
        let out = Outlet::new(tx);
        let stream = drain_stream(rx, out.stall_mark());
        self.inner.submit(Job {
            priority: Priority::Background,
            prompt: prompt.to_string(),
            params: SamplingParams::from_generate(options),
            sink: Box::new(StreamSink::new(options.stop.clone(), out)),
        })?;
        Ok(stream)
    }
}

/// [`Sink`] behind [`LlamaCppLlm::generate`]: accumulates the reply,
/// trims it at the first stop sequence and resolves a oneshot with the
/// think-stripped text.
struct CollectSink {
    output: String,
    stops: Vec<String>,
    done: oneshot::Sender<Result<String, DomainError>>,
}

impl PieceSink for CollectSink {
    fn on_piece(&mut self, piece: &str) -> Flow {
        self.output.push_str(piece);
        let hit = self
            .stops
            .iter()
            .find(|s| !s.is_empty() && self.output.contains(s.as_str()))
            .and_then(|stop| self.output.find(stop.as_str()));
        match hit {
            Some(idx) => {
                // Contract: the stop sequence is NOT part of the output
                // (see `GenerateOptions::stop`).
                self.output.truncate(idx);
                Flow::Stop
            }
            None => Flow::Continue,
        }
    }

    fn is_closed(&self) -> bool {
        self.done.is_closed()
    }
}

impl Sink for CollectSink {
    fn finish(self: Box<Self>, result: Result<(), DomainError>) {
        let Self { output, done, .. } = *self;
        // Qwen 3 <think>…</think> blocks are never useful to callers
        // (summaries, chat one-shots, etc.).
        let _ = done.send(result.map(|()| crate::shared::strip_think_blocks(&output)));
    }
}

/// [`Sink`] behind [`LlamaCppLlm::generate_stream`]: forwards each piece
/// outside `<think>` blocks. The consumer aggregates the text; a stop
/// sequence ends the stream without trimming the piece that completed it.
/// A consumer that stops draining is cut off after a bounded stall (see
/// [`Outlet`]).
struct StreamSink {
    output: String,
    stops: Vec<String>,
    in_think: bool,
    out: Outlet<String>,
}

impl StreamSink {
    fn new(stops: Vec<String>, out: Outlet<String>) -> Self {
        Self {
            output: String::new(),
            stops,
            in_think: false,
            out,
        }
    }
}

impl PieceSink for StreamSink {
    fn on_piece(&mut self, piece: &str) -> Flow {
        self.output.push_str(piece);

        // Detect <think> blocks and suppress them from streaming output.
        if self.output.contains("<think>") && !self.output.contains("</think>") {
            self.in_think = true;
        }
        if self.in_think && self.output.contains("</think>") {
            // The piece closing the block is part of it: not forwarded.
            self.in_think = false;
        } else if !self.in_think && self.out.push(piece.to_string()) == Flow::Stop {
            return Flow::Stop;
        }

        let stopped = self
            .stops
            .iter()
            .any(|s| !s.is_empty() && self.output.contains(s.as_str()));
        if stopped {
            debug!("stream stopped on stop sequence");
            Flow::Stop
        } else {
            Flow::Continue
        }
    }

    fn is_closed(&self) -> bool {
        self.out.is_closed()
    }
}

impl Sink for StreamSink {
    fn bind_cancel(&mut self, cancel: Cancel) {
        self.out.bind_cancel(cancel);
    }

    fn finish(mut self: Box<Self>, result: Result<(), DomainError>) {
        if let Err(e) = result {
            self.out.fail(e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_options_builder_round_trip() {
        let o = LoadOptions::default()
            .with_n_ctx(8_192)
            .with_n_gpu_layers(0)
            .with_model_id("test")
            .with_n_threads(2);
        assert_eq!(o.n_ctx, 8_192);
        assert_eq!(o.n_gpu_layers, 0);
        assert_eq!(o.model_id.as_deref(), Some("test"));
        assert_eq!(o.n_threads, Some(2));
    }

    #[test]
    fn missing_model_returns_model_not_loaded() {
        let err =
            LlamaCppLlm::load("/nonexistent/path/to/model.gguf").expect_err("expected load error");
        assert!(matches!(err, DomainError::ModelNotLoaded(_)), "got {err:?}");
    }

    // ---- production sinks (characterisation: they pass on first run) ----

    use std::time::{Duration, Instant};

    fn collect_sink(
        stops: &[&str],
    ) -> (CollectSink, oneshot::Receiver<Result<String, DomainError>>) {
        let (done, rx) = oneshot::channel();
        let sink = CollectSink {
            output: String::new(),
            stops: stops.iter().map(|s| (*s).to_string()).collect(),
            done,
        };
        (sink, rx)
    }

    fn stream_sink(
        stops: &[&str],
        capacity: usize,
        stall: Duration,
    ) -> (StreamSink, mpsc::Receiver<Result<String, DomainError>>) {
        let (tx, rx) = mpsc::channel(capacity);
        let sink = StreamSink::new(
            stops.iter().map(|s| (*s).to_string()).collect(),
            Outlet::new(tx).with_stall(stall),
        );
        (sink, rx)
    }

    fn drain(rx: &mut mpsc::Receiver<Result<String, DomainError>>) -> Vec<String> {
        let mut out = Vec::new();
        while let Ok(item) = rx.try_recv() {
            out.push(item.expect("piece"));
        }
        out
    }

    #[test]
    fn collect_sink_truncates_at_the_stop_sequence_and_stops() {
        let (mut sink, mut rx) = collect_sink(&["###"]);
        assert_eq!(sink.on_piece("hello "), Flow::Continue);
        assert_eq!(sink.on_piece("wor#"), Flow::Continue);
        assert_eq!(sink.on_piece("## tail"), Flow::Stop);
        Box::new(sink).finish(Ok(()));
        assert_eq!(rx.try_recv().unwrap().unwrap(), "hello wor");
    }

    #[test]
    fn collect_sink_ignores_empty_stop_sequences() {
        let (mut sink, mut rx) = collect_sink(&[""]);
        assert_eq!(sink.on_piece("abc"), Flow::Continue);
        Box::new(sink).finish(Ok(()));
        assert_eq!(rx.try_recv().unwrap().unwrap(), "abc");
    }

    #[test]
    fn collect_sink_strips_think_blocks_on_finish() {
        let (mut sink, mut rx) = collect_sink(&[]);
        sink.on_piece("<think>plan</think>");
        sink.on_piece("answer");
        Box::new(sink).finish(Ok(()));
        assert_eq!(rx.try_recv().unwrap().unwrap(), "answer");
    }

    #[test]
    fn collect_sink_reports_closed_when_the_future_is_dropped() {
        let (sink, rx) = collect_sink(&[]);
        assert!(!sink.is_closed());
        drop(rx);
        assert!(sink.is_closed());
    }

    #[test]
    fn collect_sink_forwards_an_error_instead_of_the_text() {
        let (mut sink, mut rx) = collect_sink(&[]);
        sink.on_piece("partial");
        Box::new(sink).finish(Err(DomainError::LlmFailed("decode failed".into())));
        let err = rx.try_recv().unwrap().unwrap_err();
        assert!(err.to_string().contains("decode failed"), "{err}");
    }

    #[test]
    fn stream_sink_suppresses_think_blocks_and_forwards_the_rest() {
        let (mut sink, mut rx) = stream_sink(&[], 16, Duration::from_secs(1));
        for piece in ["<think>", "hidden", "</think>", "Hola", " mundo"] {
            assert_eq!(sink.on_piece(piece), Flow::Continue);
        }
        assert_eq!(drain(&mut rx), ["Hola", " mundo"]);
    }

    #[test]
    fn stream_sink_stops_after_the_piece_completing_a_stop_sequence() {
        let (mut sink, mut rx) = stream_sink(&["END"], 16, Duration::from_secs(1));
        assert_eq!(sink.on_piece("a"), Flow::Continue);
        // Documented contract: the completing piece is still forwarded.
        assert_eq!(sink.on_piece("END"), Flow::Stop);
        assert_eq!(drain(&mut rx), ["a", "END"]);
    }

    #[test]
    fn stream_sink_cancels_when_the_receiver_is_dropped() {
        let (mut sink, rx) = stream_sink(&[], 16, Duration::from_secs(1));
        assert!(!sink.is_closed());
        drop(rx);
        assert!(sink.is_closed());
        assert_eq!(sink.on_piece("x"), Flow::Stop);
    }

    #[test]
    fn stream_sink_forwards_the_error_as_the_final_item() {
        let (mut sink, mut rx) = stream_sink(&[], 16, Duration::from_secs(1));
        sink.on_piece("a");
        Box::new(sink).finish(Err(DomainError::LlmFailed("decode failed".into())));
        assert_eq!(rx.try_recv().unwrap().unwrap(), "a");
        let err = rx.try_recv().unwrap().unwrap_err();
        assert!(err.to_string().contains("decode failed"), "{err}");
    }

    #[test]
    fn stream_sink_gives_up_on_a_consumer_that_stalls() {
        let (mut sink, mut rx) = stream_sink(&[], 1, Duration::from_millis(40));
        assert_eq!(sink.on_piece("a"), Flow::Continue);
        let started = Instant::now();
        // Buffer full and nobody draining: bounded wait, then cancel.
        assert_eq!(sink.on_piece("b"), Flow::Stop);
        assert!(started.elapsed() < Duration::from_secs(2));
        // The final error is dropped rather than waiting a second time.
        let started = Instant::now();
        Box::new(sink).finish(Err(DomainError::LlmFailed("late".into())));
        assert!(started.elapsed() < Duration::from_millis(30));
        assert_eq!(drain(&mut rx), ["a"]);
    }

    #[test]
    fn stream_sink_stops_waiting_when_the_worker_cancels() {
        let (mut sink, _rx) = stream_sink(&[], 1, Duration::from_secs(60));
        let cancel = Cancel::default();
        sink.bind_cancel(cancel.clone());
        assert_eq!(sink.on_piece("a"), Flow::Continue);
        let canceller = {
            let cancel = cancel.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(30));
                cancel.set();
            })
        };
        let started = Instant::now();
        assert_eq!(sink.on_piece("b"), Flow::Stop);
        assert!(started.elapsed() < Duration::from_secs(5));
        canceller.join().unwrap();
    }
}
