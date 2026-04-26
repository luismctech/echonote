//! Local LLM port.
//!
//! The application layer feeds an [`LlmModel`] adapter a prepared
//! prompt and receives back a single string response. Concrete
//! adapters (today: `echo_llm::LlamaCppLlm`) wrap llama.cpp behind
//! this trait so the use-case layer doesn't care which engine is
//! actually doing the inference.
//!
//! ## Why a "string in / string out" port
//!
//! It would be tempting to push prompt templating, JSON parsing and
//! retry logic into the port — but those concerns belong to the use
//! case (`SummarizeMeeting`), not to the model. The port stays narrow
//! so adapters can be swapped (llama.cpp today, candle/MLX tomorrow,
//! a remote API in tests) without touching application logic.
//!
//! ## What the port guarantees
//!
//! * Adapters MUST honour `options.max_tokens` and `options.stop`.
//! * Adapters MUST set the random seed when `options.seed` is `Some`,
//!   so callers can produce deterministic output in tests.
//! * Adapters MAY return [`DomainError::ModelNotLoaded`] when the
//!   underlying model file is missing and [`DomainError::LlmFailed`]
//!   for any other runtime failure (OOM, context overflow, decode
//!   error, …).
//! * Calls are async-safe; the adapter is responsible for offloading
//!   the synchronous inference loop to a blocking thread (typically
//!   `tokio::task::spawn_blocking`).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::DomainError;

/// Knobs the application layer can pass to the LLM. Defaults are
/// tuned for "give me a JSON summary": low temperature, sane token
/// budget, no nucleus sampling tricks. The use case overrides what it
/// needs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerateOptions {
    /// Hard cap on tokens generated. The adapter MUST stop emitting
    /// tokens once this is reached, even if the model has not produced
    /// an end-of-sequence token yet.
    pub max_tokens: u32,

    /// Sampling temperature in `[0.0, 2.0]`. `0.0` yields greedy
    /// decoding; values around `0.2`-`0.4` are appropriate for
    /// structured output, `0.7`+ for creative writing. Adapters
    /// SHOULD clamp out-of-range values rather than error.
    pub temperature: f32,

    /// Nucleus sampling parameter in `(0.0, 1.0]`. `1.0` disables it.
    pub top_p: f32,

    /// Seed for the RNG. `Some(_)` makes inference reproducible
    /// across runs (used heavily in tests); `None` lets the adapter
    /// pick a random one.
    pub seed: Option<u64>,

    /// Optional list of strings that, when produced, terminate
    /// decoding. Useful to stop on `"\n```"` after a JSON block, for
    /// instance. The matching string is *not* included in the output.
    pub stop: Vec<String>,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        // Defaults aimed at structured-output use cases (summaries).
        // The chat use case in CU-05 will override `temperature` and
        // `max_tokens` when it lands.
        Self {
            max_tokens: 1_024,
            temperature: 0.2,
            top_p: 0.95,
            seed: None,
            stop: Vec::new(),
        }
    }
}

/// Async port over a local large language model.
///
/// Implementors are expected to be cheap to clone (`Arc` internally)
/// because the use-case layer keeps a single instance behind
/// `Arc<dyn LlmModel>` and shares it across all summary requests.
#[async_trait]
pub trait LlmModel: Send + Sync {
    /// A short, stable identifier for the model currently loaded —
    /// used as provenance metadata on each [`crate::entities::summary::Summary`].
    /// Examples: `"qwen2.5-7b-instruct-q4_k_m"`, `"llama-3.2-3b-instruct-q4_k_m"`.
    fn model_id(&self) -> &str;

    /// Generate a completion for `prompt`. The prompt is expected to
    /// already include any chat template wrapping (`<|im_start|>` for
    /// Qwen, `<s>[INST]` for Llama, …) — the adapter passes it
    /// through verbatim so the use-case layer keeps full control over
    /// the conversation shape.
    ///
    /// Returns the decoded text, **without** the prompt prefix and
    /// without any trailing stop-sequence match.
    async fn generate(
        &self,
        prompt: &str,
        options: &GenerateOptions,
    ) -> Result<String, DomainError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_options_are_structured_output_friendly() {
        // Pin the defaults: if someone bumps temperature to 0.7 the
        // summarizer will start hallucinating — make that decision
        // explicit by failing this test.
        let opts = GenerateOptions::default();
        assert_eq!(opts.max_tokens, 1_024);
        assert!((opts.temperature - 0.2).abs() < f32::EPSILON);
        assert!((opts.top_p - 0.95).abs() < f32::EPSILON);
        assert!(opts.seed.is_none());
        assert!(opts.stop.is_empty());
    }

    #[test]
    fn options_round_trip_through_serde() {
        // Used by the IPC layer + CLI flags, so make sure custom
        // values survive a serialize/deserialize round-trip.
        let opts = GenerateOptions {
            max_tokens: 512,
            temperature: 0.0,
            top_p: 1.0,
            seed: Some(42),
            stop: vec!["\n```".into(), "<|im_end|>".into()],
        };
        let json = serde_json::to_string(&opts).unwrap();
        let back: GenerateOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(back, opts);
    }
}
