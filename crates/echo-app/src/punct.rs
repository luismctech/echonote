//! Rule-based punctuation restoration.
//!
//! [`NaivePunctuator`] is a zero-dependency, always-available fallback
//! that handles the most common deficiencies in raw Whisper output:
//!
//! 1. **Missing terminal punctuation** — adds `.` (or `?` for detected
//!    questions) when the segment ends without `.`, `!`, `?`, `…`, `:`,
//!    or `;`.
//! 2. **Lower-case first letter** — capitalises the first character of
//!    each segment so that segments can be read independently of context.
//!
//! ## What it intentionally does NOT do
//!
//! - Insert commas mid-sentence — too error-prone without a language model.
//! - Rewrite existing punctuation — if Whisper already added `.` we trust it.
//! - Change any word — semantics are untouched.
//!
//! ## Question detection heuristics
//!
//! English: segment starts with a known interrogative/auxiliary verb
//! (`who`, `what`, `when`, `where`, `why`, `how`, `is`, `are`, …).
//!
//! Spanish: segment starts with `¿` (already a question opener) or with
//! a known interrogative word with or without written accent.

use echo_domain::Punctuator;

/// Zero-dependency, rule-based punctuation restorer.
///
/// Thread-safe: holds no mutable state. Cheap to clone or share via
/// `Arc<NaivePunctuator>`.
#[derive(Debug, Clone, Default)]
pub struct NaivePunctuator;

impl Punctuator for NaivePunctuator {
    fn punctuate(&self, text: &str, language: Option<&str>) -> String {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return String::new();
        }

        // If the segment already ends with terminal punctuation, only
        // ensure the first character is capitalised.
        if ends_with_terminal(trimmed) {
            return capitalize_first(trimmed);
        }

        let capitalized = capitalize_first(trimmed);
        if is_question(trimmed, language) {
            format!("{capitalized}?")
        } else {
            format!("{capitalized}.")
        }
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn ends_with_terminal(s: &str) -> bool {
    matches!(s.chars().last(), Some('.' | '!' | '?' | '…' | ':' | ';'))
}

fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

fn is_question(text: &str, language: Option<&str>) -> bool {
    // Spanish inverted question mark is an unambiguous signal.
    if text.starts_with('¿') {
        return true;
    }

    let lower = text.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();
    if words.is_empty() {
        return false;
    }

    let lang = language.unwrap_or("en");

    if lang.starts_with("es") {
        is_spanish_question(&words)
    } else {
        is_english_question(&words)
    }
}

fn is_english_question(words: &[&str]) -> bool {
    const INTERROGATIVES: &[&str] = &[
        "who", "what", "when", "where", "why", "how", "which", "whose", "whom",
    ];
    const AUXILIARIES: &[&str] = &[
        "is",
        "are",
        "was",
        "were",
        "am",
        "do",
        "does",
        "did",
        "can",
        "could",
        "would",
        "should",
        "will",
        "shall",
        "have",
        "has",
        "had",
        "may",
        "might",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "don't",
        "doesn't",
        "didn't",
        "can't",
        "couldn't",
        "wouldn't",
        "shouldn't",
        "won't",
    ];
    let first = words.first().copied().unwrap_or("");
    INTERROGATIVES.contains(&first) || AUXILIARIES.contains(&first)
}

fn is_spanish_question(words: &[&str]) -> bool {
    const SINGLE: &[&str] = &[
        // with accent (Whisper usually outputs these)
        "quién", "quiénes", "qué", "cuándo", "dónde", "cómo", "cuál", "cuáles", "cuánto", "cuánta",
        "cuántos", "cuántas",
        // without accent (fallback for models that strip diacritics)
        "quien", "quienes", "que", "cuando", "donde", "como", "cual", "cuales", "cuanto", "cuanta",
        "cuantos", "cuantas",
    ];
    const DOUBLE_FIRST: &[&str] = &["por", "en", "de", "a", "para", "desde", "hasta", "con"];

    let first = words.first().copied().unwrap_or("");
    let second = words.get(1).copied().unwrap_or("");

    if SINGLE.contains(&first) {
        return true;
    }
    // "por qué / en qué / de qué / a quién" etc.
    if DOUBLE_FIRST.contains(&first) {
        let candidates = ["qué", "que", "quién", "quien", "quiénes", "quienes"];
        if candidates.contains(&second) {
            return true;
        }
    }
    false
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use echo_domain::Punctuator;

    fn p(text: &str, lang: &str) -> String {
        NaivePunctuator.punctuate(text, Some(lang))
    }

    #[test]
    fn adds_period_to_unpunctuated_statement() {
        assert_eq!(p("hello world", "en"), "Hello world.");
        assert_eq!(
            p("the meeting starts at nine", "en"),
            "The meeting starts at nine."
        );
    }

    #[test]
    fn adds_question_mark_to_english_question() {
        assert_eq!(
            p("what time is the meeting", "en"),
            "What time is the meeting?"
        );
        assert_eq!(
            p("how are you doing today", "en"),
            "How are you doing today?"
        );
        assert_eq!(p("is the report ready", "en"), "Is the report ready?");
        assert_eq!(
            p("can you send me the file", "en"),
            "Can you send me the file?"
        );
    }

    #[test]
    fn does_not_alter_already_punctuated_text() {
        assert_eq!(p("Hello world.", "en"), "Hello world.");
        assert_eq!(p("Are you sure?", "en"), "Are you sure?");
        assert_eq!(p("Watch out!", "en"), "Watch out!");
    }

    #[test]
    fn capitalises_first_letter() {
        assert_eq!(p("the quick brown fox", "en"), "The quick brown fox.");
    }

    #[test]
    fn adds_question_mark_to_spanish_question_with_inverted_mark() {
        assert_eq!(p("¿cómo estás", "es"), "¿cómo estás?");
    }

    #[test]
    fn adds_question_mark_to_spanish_interrogative() {
        assert_eq!(p("cómo estás hoy", "es"), "Cómo estás hoy?");
        assert_eq!(p("qué hora es", "es"), "Qué hora es?");
        assert_eq!(p("por qué no viniste", "es"), "Por qué no viniste?");
    }

    #[test]
    fn adds_period_to_spanish_statement() {
        assert_eq!(
            p("la reunión empieza a las nueve", "es"),
            "La reunión empieza a las nueve."
        );
    }

    #[test]
    fn empty_input_returns_empty() {
        assert_eq!(p("", "en"), "");
        assert_eq!(p("   ", "en"), "");
    }

    #[test]
    fn handles_unknown_language_as_english() {
        assert_eq!(
            NaivePunctuator.punctuate("what is this", None),
            "What is this?"
        );
    }
}
