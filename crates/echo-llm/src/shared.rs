//! State shared between every llama.cpp-backed adapter in this crate.
//!
//! [`LoadedModel`] holds the heavy bits — the `LlamaBackend` singleton
//! (leaked, lives for the process), the loaded `LlamaModel` (~10 GB
//! for Qwen 3 14B Q4_K_M) and the configuration knobs that decide how
//! per-request `LlamaContext`s are constructed.
//!
//! Every adapter ([`crate::LlamaCppLlm`] for one-shot summaries,
//! [`crate::LlamaCppChat`] for streaming chat) wraps an
//! `Arc<LoadedModel>`. Cloning the `Arc` is free, which is what lets
//! the application layer build both ports off the same loaded model
//! and serve them concurrently:
//!
//! ```text
//! let llm  = LlamaCppLlm::load("qwen3-14b.gguf")?;   // 10 GB resident
//! let chat = llm.chat_handle();                       // 0 extra cost
//! ```
//!
//! ## Concurrency model
//!
//! `LlamaModel` is `Send + Sync`. llama.cpp tolerates **multiple
//! `LlamaContext`s** built from the same model running in parallel —
//! each context owns its own KV cache, so the only constraint is host
//! RAM (one ~50 MB KV cache per concurrent request at the default
//! `n_ctx`). The single decoding constraint is **per-context**: a
//! single context cannot be decoded from two threads at once, but the
//! adapters in this crate never share a context across calls (each
//! call constructs a fresh one and drops it on completion).
//!
//! That removes any need for a `tokio::sync::Mutex` between summary
//! and chat. The decision was originally documented in
//! `docs/SPRINT-1-STATUS.md` §8.3 as "share with mutex"; reading the
//! existing adapter code revealed mutex-free sharing is what we
//! actually have, and the doc has been updated accordingly.
//!
//! ## Why this lives in its own module
//!
//! Two reasons. First, `LlamaCppLlm` and `LlamaCppChat` would
//! otherwise have to share state via `pub(crate)` fields on each
//! other, which makes the inheritance fragile. Second, putting the
//! load logic here keeps both adapters trivially testable: a mock /
//! fake adapter that doesn't go through llama.cpp doesn't depend on
//! this module at all.

use std::path::PathBuf;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;

/// Per-process loaded model + the knobs needed to spawn a per-request
/// context.
///
/// Held inside `Arc<LoadedModel>` by every adapter — the `Arc` is the
/// only sharing primitive needed.
pub(crate) struct LoadedModel {
    /// Backend handle. Leaked once per process by [`crate::backend::backend_singleton`].
    pub backend: &'static LlamaBackend,
    /// The loaded model. `Send + Sync`; safe to access from any thread,
    /// including from concurrent `new_context` calls.
    pub model: LlamaModel,
    /// Stable id reported to the application layer. Either the file
    /// stem (e.g. `qwen3-14b-instruct-q4_k_m`) or the override the
    /// caller passed in [`crate::LoadOptions::with_model_id`].
    pub model_id: String,
    /// Path the model was loaded from. Surfaced for telemetry +
    /// "regenerate with model X" UX flows.
    pub model_path: PathBuf,
    /// Context window in tokens, applied to every per-request
    /// `LlamaContext` we build.
    pub n_ctx: u32,
    /// Optional decoder thread count. `None` defers to llama.cpp's
    /// default (typically `num_cpus`). Useful in tests.
    pub n_threads: Option<i32>,
}
