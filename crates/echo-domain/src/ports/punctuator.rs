//! Punctuation restoration port.
//!
//! [`Punctuator`] takes raw ASR text (which may lack terminal punctuation
//! or proper capitalisation) and returns a corrected version. Adapters
//! range from simple rule-based heuristics to neural ONNX models.

/// Synchronous, infallible punctuation corrector.
///
/// Implementations must be `Send + Sync` so they can be held behind an
/// `Arc` and called from async contexts.
pub trait Punctuator: Send + Sync {
    /// Return a punctuated and capitalised version of `text`.
    ///
    /// - `text`     — raw ASR segment text, possibly already punctuated.
    /// - `language` — ISO-639-1 hint (e.g. `"en"`, `"es"`). `None`
    ///   means unknown; implementations should default to English.
    ///
    /// Implementations **must not** alter semantics: only add/correct
    /// punctuation and capitalisation; never change words.
    fn punctuate(&self, text: &str, language: Option<&str>) -> String;
}
