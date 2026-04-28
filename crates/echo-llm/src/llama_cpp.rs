//! llama.cpp adapter via [`llama-cpp-2`].
//!
//! Loads a `.gguf` model into memory once, then serves
//! [`echo_domain::LlmModel::generate`] calls by spinning up a fresh
//! decoding context per request. Generation runs on
//! `tokio::task::spawn_blocking` so the async runtime is never blocked
//! by the inference loop.
//!
//! ## Threading model
//!
//! * `LlamaBackend::init()` is a process-wide singleton — guarded by a
//!   `OnceLock` so multiple `LlamaCppLlm::load` calls (tests + runtime)
//!   don't trip the "backend already initialised" error.
//! * The model is loaded once into an `Arc<LoadedModel>` and shared
//!   across every generation request — and across every adapter
//!   built from it (see [`Self::chat_handle`]). Contexts are cheap to
//!   create relative to model loading, so each request gets its own.
//! * The inference loop is synchronous; we wrap it in `spawn_blocking`
//!   so a long summary doesn't stall the rest of the runtime.

use std::num::NonZeroU32;
use std::path::Path;
use std::pin::pin;
use std::sync::Arc;

use async_trait::async_trait;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use tracing::{debug, info, instrument, warn};

use echo_domain::{DomainError, GenerateOptions, LlmModel};

use crate::backend::backend_singleton;
use crate::llama_cpp_chat::LlamaCppChat;
use crate::shared::LoadedModel;

/// Default context window when the caller does not override it. Both
/// Qwen 3 8B and 14B ship with `n_ctx_train = 32_768`; 8192 is
/// generous enough for a full meeting transcript (~6 k chars ≈ 1.7 k
/// tokens) + system prompt + chat history + 1 k output tokens, while
/// keeping the per-context KV-cache allocation under ~200 MB for 8B
/// Q4_K_M. Callers that need more (or want to save RAM) can pass
/// `LlamaCppLlm::load_with(path, opts.with_n_ctx(...))`.
const DEFAULT_N_CTX: u32 = 8_192;

/// Number of model layers to offload to the GPU when Metal/CUDA is
/// available. `999` means "all of them"; llama.cpp clamps internally.
const DEFAULT_N_GPU_LAYERS: u32 = 999;

/// Initial size of the decoding batch. We resize on demand if the
/// prompt is larger.
const INITIAL_BATCH_TOKENS: usize = 512;

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
    inner: Arc<LoadedModel>,
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

        Ok(Self {
            inner: Arc::new(LoadedModel {
                backend,
                model,
                model_id,
                model_path: path.to_path_buf(),
                n_ctx: opts.n_ctx,
                n_threads: opts.n_threads,
            }),
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
}

#[async_trait]
impl LlmModel for LlamaCppLlm {
    fn model_id(&self) -> &str {
        &self.inner.model_id
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

        let inner = Arc::clone(&self.inner);
        let prompt = prompt.to_string();
        let options = options.clone();

        // llama.cpp's decode loop is CPU/GPU-bound and synchronous.
        // Move it off the runtime so other IPC handlers stay
        // responsive.
        tokio::task::spawn_blocking(move || generate_blocking(&inner, &prompt, &options))
            .await
            .map_err(|e| DomainError::LlmFailed(format!("generate join: {e}")))?
    }
}

/// Synchronous inference loop. Lives in its own function (rather than
/// inlined into the trait impl) so it's easy to call from
/// `cargo test --features ...` smoke tests without going through the
/// async machinery.
fn generate_blocking(
    inner: &LoadedModel,
    prompt: &str,
    options: &GenerateOptions,
) -> Result<String, DomainError> {
    let n_ctx = NonZeroU32::new(inner.n_ctx)
        .ok_or_else(|| DomainError::Invariant("n_ctx must be > 0".into()))?;

    let mut ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(n_ctx))
        .with_n_batch(inner.n_ctx);
    if let Some(threads) = inner.n_threads {
        ctx_params = ctx_params.with_n_threads(threads);
        ctx_params = ctx_params.with_n_threads_batch(threads);
    }

    let mut ctx = inner
        .model
        .new_context(inner.backend, ctx_params)
        .map_err(|e| DomainError::LlmFailed(format!("create context: {e}")))?;

    // ---- tokenize prompt ----------------------------------------------------
    // The use-case layer is responsible for wrapping the prompt with
    // the model's chat template (`<|im_start|>` for Qwen, `<s>[INST]`
    // for Llama, …) — including any BOS markers that template
    // requires. We therefore pass `AddBos::Never` so we don't end up
    // with a duplicate BOS for templates that already include one.
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

    // ---- prime the KV cache with the prompt --------------------------------
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

    // ---- sampler chain ------------------------------------------------------
    // Greedy when the caller asks for it (temp == 0), otherwise a
    // top-p + temperature stack. `dist` is always last so it produces
    // the final token from whatever distribution the previous samplers
    // shaped.
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

    // ---- decode loop --------------------------------------------------------
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut output = String::new();
    let mut n_cur = batch.n_tokens();
    let mut n_decoded: u32 = 0;
    let max_new = options.max_tokens;

    debug!(
        n_prompt,
        n_ctx = inner.n_ctx,
        max_new,
        temp,
        top_p,
        "starting llama decode loop"
    );

    while n_decoded < max_new {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        if inner.model.is_eog_token(token) {
            debug!(n_decoded, "stopped on EOG token");
            break;
        }

        // `lstrip = None` matches the upstream `simple` example —
        // llama.cpp will only insert leading whitespace when the
        // tokenizer originally produced it, which is the right
        // default for free-form generation.
        let piece = inner
            .model
            .token_to_piece(token, &mut decoder, true, None)
            .map_err(|e| DomainError::LlmFailed(format!("detokenize: {e}")))?;
        output.push_str(&piece);
        n_decoded += 1;

        if let Some(stop) = options
            .stop
            .iter()
            .find(|s| !s.is_empty() && output.contains(s.as_str()))
        {
            // Trim everything from the stop sequence onwards. We
            // already consumed the tokens that produced it, but the
            // contract is "stop sequence is NOT included in the
            // output" — see GenerateOptions::stop.
            if let Some(idx) = output.find(stop.as_str()) {
                output.truncate(idx);
            }
            debug!(n_decoded, %stop, "stopped on stop sequence");
            break;
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

    // Strip Qwen 3 <think>…</think> reasoning blocks — they are
    // never useful to callers (summaries, chat one-shots, etc.).
    let output = crate::shared::strip_think_blocks(&output);

    Ok(output)
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
}
