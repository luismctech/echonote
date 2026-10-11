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
use futures::stream::BoxStream;
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

/// Context window assumed by [`LlmModel::context_window`] and
/// [`crate::ChatAssistant::context_window`] when an adapter does not
/// report its own. Matches the default `n_ctx` of the llama.cpp adapter.
pub const DEFAULT_CONTEXT_WINDOW: u32 = 8_192;

/// Conservative token estimate: one token per 2 UTF-8 bytes, rounded up.
///
/// Byte-level BPE tokenizers split accented and non-Latin characters
/// into several byte tokens, so a character count can undercount by a
/// wide margin. Measured with the Qwen3-8B tokenizer: natural Spanish
/// and English run at 3.5-5 bytes per token, but a text made only of
/// accented vowels (`áéíóúñ`) costs about one token per 2.1 bytes, 2.5x
/// more than `chars / 3`. Two bytes per token stays at or above every
/// measured sample, including that worst case. Over-counting is the safe
/// direction for a budget: a prompt sized with this estimate can only
/// end up smaller than the window, never larger. Adapters with a real
/// tokenizer should override `count_tokens` and may fall back to this
/// function when tokenization fails.
#[must_use]
pub fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(2)
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

    /// Size of the context window in tokens: prompt plus generated
    /// output must fit in it. Use cases derive their prompt budget from
    /// this value instead of a fixed character cap.
    ///
    /// The default ([`DEFAULT_CONTEXT_WINDOW`]) suits fakes and adapters
    /// that cannot introspect their model.
    fn context_window(&self) -> u32 {
        DEFAULT_CONTEXT_WINDOW
    }

    /// Number of tokens `text` occupies for this model.
    ///
    /// The default is [`estimate_tokens`], a deliberately pessimistic
    /// bytes/2 estimate, so budgets built on it stay safe when no real
    /// tokenizer is available. Adapters SHOULD override it with the
    /// model's own tokenizer.
    fn count_tokens(&self, text: &str) -> usize {
        estimate_tokens(text)
    }

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

    /// Stream a completion token-by-token. Each item is a decoded text
    /// piece (may be a partial word). The default implementation calls
    /// [`Self::generate`] and yields the full text as a single chunk.
    ///
    /// Adapters SHOULD override this to provide real incremental
    /// streaming so the UI can render tokens as they arrive.
    async fn generate_stream(
        &self,
        prompt: &str,
        options: &GenerateOptions,
    ) -> Result<BoxStream<'static, Result<String, DomainError>>, DomainError> {
        let text = self.generate(prompt, options).await?;
        Ok(Box::pin(futures::stream::once(async move { Ok(text) })))
    }
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

    struct MinimalLlm;

    #[async_trait]
    impl LlmModel for MinimalLlm {
        fn model_id(&self) -> &str {
            "minimal"
        }

        async fn generate(&self, _: &str, _: &GenerateOptions) -> Result<String, DomainError> {
            Ok(String::new())
        }
    }

    #[test]
    fn token_budget_defaults_are_conservative() {
        // Fakes that don't override the budget methods must still get a
        // usable window and an estimate that errs on the high side.
        let llm = MinimalLlm;
        assert_eq!(llm.context_window(), 8_192);
        assert_eq!(llm.count_tokens("abcdefg"), 4);
        assert_eq!(llm.count_tokens(""), 0);
    }

    #[test]
    fn estimate_is_half_the_utf8_bytes_rounded_up() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abc"), 2);
        // Accented letters weigh two bytes each: 6 chars, 12 bytes.
        assert_eq!(estimate_tokens("áéíóúñ"), 6);
    }

    #[test]
    fn estimate_never_drops_below_the_previous_chars_over_three() {
        for text in [
            "",
            "a",
            "hello world",
            "Además, la configuración",
            "日本語のテキスト",
            "Reunión de planificación 🚀🚀 ñandú",
        ] {
            // The estimator used to be chars/3; it must never undercount
            // relative to that, which is the floor callers were tuned on.
            let char_floor = text.chars().count().div_ceil(3);
            assert!(estimate_tokens(text) >= char_floor, "{text:?}");
        }
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
