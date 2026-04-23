//! Process-wide [`LlamaBackend`] singleton.
//!
//! `LlamaBackend::init()` errors on the second call (it pulls in the
//! global llama.cpp logger and the GPU device list), so we cache the
//! successfully initialised handle in a `OnceLock` and hand out
//! `&'static` references. The leak is intentional: there is one
//! backend per process and it lives for the entire process lifetime,
//! so it would be dropped on shutdown anyway.
//!
//! Lives in its own module so [`crate::LlamaCppLlm`] and
//! [`crate::LlamaCppChat`] both reach the same singleton without
//! either of them owning the other's loading code.

use std::sync::OnceLock;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::{send_logs_to_tracing, LogOptions};

use echo_domain::DomainError;

/// Lazily initialise (and leak) a process-wide [`LlamaBackend`].
pub(crate) fn backend_singleton() -> Result<&'static LlamaBackend, DomainError> {
    static BACKEND: OnceLock<&'static LlamaBackend> = OnceLock::new();
    if let Some(b) = BACKEND.get() {
        return Ok(*b);
    }

    // Forward llama.cpp's chatty C++ logs to `tracing`. Disabled by
    // default to avoid drowning the rest of the logs; the env-filter
    // can re-enable them with `RUST_LOG=llama_cpp_2=info`.
    send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));

    let backend = LlamaBackend::init()
        .map_err(|e| DomainError::LlmFailed(format!("LlamaBackend::init: {e}")))?;
    let leaked: &'static LlamaBackend = Box::leak(Box::new(backend));
    let _ = BACKEND.set(leaked);
    Ok(BACKEND.get().copied().expect("backend just set"))
}
