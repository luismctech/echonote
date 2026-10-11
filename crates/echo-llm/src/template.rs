//! Prompt rendering per model family.
//!
//! The application layer hands the adapters structured
//! [`ChatMessage`]s; this module turns them into the text one model
//! expects. The family is detected once at load time from the GGUF
//! metadata ([`PromptFormat::detect`]) and decides three things: how the
//! prompt is rendered, which extra stop sequences the model needs, and
//! whether Qwen3's `/no_think` switch applies.
//!
//! | Family   | Rendering                                   | Stops        | `/no_think` |
//! |----------|---------------------------------------------|--------------|-------------|
//! | `Qwen`   | ChatML ([`echo_domain::render_chatml`])      | `<|im_end|>` | yes         |
//! | `Gemma4` | the GGUF's Jinja template (llama.cpp)        | `<turn|>`    | no          |
//! | `Other`  | the GGUF's Jinja template (llama.cpp)        | none (EOG)   | no          |
//!
//! Qwen deliberately skips its Jinja template: with thinking disabled,
//! Qwen3's template appends an empty `<think></think>` block to the
//! generation prompt, which would change the prompt (and the output) the
//! app has been tuned and measured with. ChatML is the exact wire format
//! of that template for the roles we use, so nothing is lost.
//!
//! When a Jinja template is missing or fails to render, the prompt falls
//! back to ChatML with a warning rather than failing the request.

use echo_domain::{render_chatml, ChatMessage, ChatRole};
use tracing::warn;

/// Suffix that switches Qwen3's reasoning off, appended to the last user
/// turn.
const QWEN_NO_THINK: &str = "\n/no_think";

/// Prompt family of a loaded model. See the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PromptFormat {
    /// Qwen 2/3: ChatML, `/no_think`, `<|im_end|>` stop.
    Qwen,
    /// Gemma 4: `<|turn>` / `<turn|>` framing, no `/no_think`.
    Gemma4,
    /// Any other model: its own template, only EOG tokens end a turn.
    Other,
}

impl PromptFormat {
    /// Family from the GGUF's `general.architecture` and chat template.
    /// Pure so it is testable without a model.
    pub(crate) fn detect(architecture: Option<&str>, template: Option<&str>) -> Self {
        let architecture = architecture.map(str::to_ascii_lowercase);
        let architecture = architecture.as_deref().unwrap_or("");
        if architecture.starts_with("qwen") {
            Self::Qwen
        } else if architecture.starts_with("gemma4")
            || template.is_some_and(|t| t.contains("<|turn>"))
        {
            Self::Gemma4
        } else {
            Self::Other
        }
    }

    /// Text stop sequences the model needs on top of the caller's. Most
    /// are also EOG tokens, which end decoding earlier; the text form
    /// covers a build that emits them as plain text.
    pub(crate) fn stops(self) -> &'static [&'static str] {
        match self {
            Self::Qwen => &["<|im_end|>"],
            Self::Gemma4 => &["<turn|>"],
            Self::Other => &[],
        }
    }

    /// Suffix appended to the last user turn of a request that wants no
    /// reasoning, if the family has such a switch.
    fn no_think_suffix(self) -> Option<&'static str> {
        (self == Self::Qwen).then_some(QWEN_NO_THINK)
    }

    /// Whether prompts go through the GGUF's Jinja template.
    fn uses_jinja(self) -> bool {
        self != Self::Qwen
    }
}

/// What the Jinja renderer returns: the prompt and the stop strings the
/// template asks for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TemplateOutput {
    /// Rendered prompt, primed for the assistant turn.
    pub prompt: String,
    /// Stop strings contributed by the template.
    pub additional_stops: Vec<String>,
}

/// A rendered prompt with the stop sequences the model adds to the
/// caller's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Rendered {
    /// Prompt text to tokenize (special tokens are parsed).
    pub text: String,
    /// Model-specific stop sequences.
    pub stops: Vec<String>,
}

/// Render `messages` for a model of family `format`.
///
/// `suppress_thinking` appends the family's reasoning switch (Qwen's
/// `/no_think`) to the last user turn; Jinja families get
/// `enable_thinking = false` from `jinja` regardless. `jinja` renders with
/// the GGUF's template and is only called for families that use it; an
/// `Err` falls back to ChatML.
pub(crate) fn render_chat(
    format: PromptFormat,
    messages: &[ChatMessage],
    suppress_thinking: bool,
    jinja: impl FnOnce(&[ChatMessage]) -> Result<TemplateOutput, String>,
) -> Rendered {
    let switched;
    let messages = match format.no_think_suffix().filter(|_| suppress_thinking) {
        Some(suffix) => {
            switched = with_suffix_on_last_user_turn(messages, suffix);
            &switched[..]
        }
        None => messages,
    };

    if format.uses_jinja() {
        match jinja(messages) {
            Ok(out) => {
                let mut stops: Vec<String> = format.stops().iter().map(|s| (*s).into()).collect();
                stops.extend(out.additional_stops);
                return Rendered {
                    text: out.prompt,
                    stops: merge_stops(&[], &stops),
                };
            }
            Err(reason) => {
                warn!(%reason, ?format, "chat template unavailable, falling back to ChatML");
            }
        }
    }

    let mut stops: Vec<String> = vec!["<|im_end|>".into()];
    stops.extend(format.stops().iter().map(|s| (*s).to_string()));
    Rendered {
        text: render_chatml(messages),
        stops: merge_stops(&[], &stops),
    }
}

/// `messages` with `suffix` appended to the content of the last user
/// message (none are changed when there is no user message).
fn with_suffix_on_last_user_turn(messages: &[ChatMessage], suffix: &str) -> Vec<ChatMessage> {
    let mut out = messages.to_vec();
    if let Some(last_user) = out.iter_mut().rev().find(|m| m.role == ChatRole::User) {
        last_user.content.push_str(suffix);
    }
    out
}

/// `request` stops followed by the `extra` ones the model needs, without
/// duplicates or empty strings.
pub(crate) fn merge_stops(request: &[String], extra: &[String]) -> Vec<String> {
    let mut merged: Vec<String> = Vec::with_capacity(request.len() + extra.len());
    for stop in request.iter().chain(extra) {
        if !stop.is_empty() && !merged.contains(stop) {
            merged.push(stop.clone());
        }
    }
    merged
}

/// OpenAI-style JSON array of `messages`, the input of llama.cpp's Jinja
/// renderer.
pub(crate) fn messages_json(messages: &[ChatMessage]) -> String {
    let items: Vec<serde_json::Value> = messages
        .iter()
        .map(|m| {
            let role = match m.role {
                ChatRole::System => "system",
                ChatRole::User => "user",
                ChatRole::Assistant => "assistant",
            };
            serde_json::json!({ "role": role, "content": m.content })
        })
        .collect();
    serde_json::Value::Array(items).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn convo() -> Vec<ChatMessage> {
        vec![
            ChatMessage::system("SYS"),
            ChatMessage::user("first"),
            ChatMessage::assistant("answer"),
            ChatMessage::user("second"),
        ]
    }

    fn never_called(_: &[ChatMessage]) -> Result<TemplateOutput, String> {
        panic!("the Jinja renderer must not be used");
    }

    // ---- detection --------------------------------------------------------

    #[test]
    fn qwen_architectures_are_the_qwen_family() {
        for arch in ["qwen2", "qwen3", "qwen3moe", "Qwen3"] {
            assert_eq!(PromptFormat::detect(Some(arch), None), PromptFormat::Qwen);
        }
        // The architecture wins over whatever template the file carries.
        assert_eq!(
            PromptFormat::detect(Some("qwen3"), Some("{{ '<|turn>' }}")),
            PromptFormat::Qwen
        );
    }

    #[test]
    fn gemma4_is_detected_by_architecture_or_by_its_turn_marker() {
        assert_eq!(
            PromptFormat::detect(Some("gemma4"), None),
            PromptFormat::Gemma4
        );
        assert_eq!(
            PromptFormat::detect(None, Some("{{ bos_token }}<|turn>user\n")),
            PromptFormat::Gemma4
        );
    }

    #[test]
    fn everything_else_is_other() {
        assert_eq!(
            PromptFormat::detect(Some("llama"), Some("[INST]")),
            PromptFormat::Other
        );
        // Gemma 1-3 use another template and need no Gemma 4 handling.
        assert_eq!(
            PromptFormat::detect(Some("gemma3"), Some("<start_of_turn>user")),
            PromptFormat::Other
        );
        assert_eq!(PromptFormat::detect(None, None), PromptFormat::Other);
    }

    // ---- families ---------------------------------------------------------

    #[test]
    fn qwen_renders_chatml_and_adds_no_think_to_the_last_user_turn_only() {
        let out = render_chat(PromptFormat::Qwen, &convo(), true, never_called);
        let expected = "\
<|im_start|>system
SYS<|im_end|>
<|im_start|>user
first<|im_end|>
<|im_start|>assistant
answer<|im_end|>
<|im_start|>user
second
/no_think<|im_end|>
<|im_start|>assistant
";
        assert_eq!(out.text, expected);
        assert_eq!(out.stops, vec!["<|im_end|>".to_string()]);
    }

    #[test]
    fn qwen_golden_matches_the_prompt_the_app_used_to_build_by_hand() {
        // Shape of `generate_summary::wrap_qwen_prompt` before the change.
        let (system, user) = ("Be brief.", "Transcript:\n---\nhi\n---");
        let old = format!(
            "<|im_start|>system\n{system}<|im_end|>\n\
             <|im_start|>user\n{user}\n/no_think<|im_end|>\n\
             <|im_start|>assistant\n"
        );
        let messages = [ChatMessage::system(system), ChatMessage::user(user)];
        let out = render_chat(PromptFormat::Qwen, &messages, true, never_called);
        assert_eq!(out.text, old);
    }

    #[test]
    fn qwen_without_the_thinking_switch_is_plain_chatml() {
        let out = render_chat(PromptFormat::Qwen, &convo(), false, never_called);
        assert_eq!(out.text, render_chatml(&convo()));
        assert!(!out.text.contains("/no_think"));
    }

    #[test]
    fn gemma4_uses_the_template_without_no_think_and_stops_on_turn_end() {
        let seen = std::cell::RefCell::new(Vec::new());
        let out = render_chat(PromptFormat::Gemma4, &convo(), true, |messages| {
            *seen.borrow_mut() = messages.to_vec();
            Ok(TemplateOutput {
                prompt: "<bos>RENDERED".into(),
                additional_stops: vec!["<turn|>".into(), "<eos>".into()],
            })
        });
        assert_eq!(
            seen.into_inner(),
            convo(),
            "messages reach the template as-is"
        );
        assert_eq!(out.text, "<bos>RENDERED");
        assert_eq!(out.stops, vec!["<turn|>".to_string(), "<eos>".to_string()]);
    }

    #[test]
    fn other_models_add_only_the_stops_their_template_asks_for() {
        let out = render_chat(PromptFormat::Other, &convo(), true, |_| {
            Ok(TemplateOutput {
                prompt: "P".into(),
                additional_stops: vec![],
            })
        });
        assert_eq!(out.text, "P");
        assert!(out.stops.is_empty());
    }

    #[test]
    fn a_failing_template_falls_back_to_chatml() {
        for format in [PromptFormat::Gemma4, PromptFormat::Other] {
            let out = render_chat(format, &convo(), true, |_| Err("no template".into()));
            assert_eq!(out.text, render_chatml(&convo()), "{format:?}");
            assert!(out.stops.contains(&"<|im_end|>".to_string()), "{format:?}");
            assert!(!out.text.contains("/no_think"), "{format:?}");
        }
    }

    // ---- helpers ----------------------------------------------------------

    #[test]
    fn merge_stops_keeps_order_and_drops_duplicates_and_empties() {
        let merged = merge_stops(
            &["a".to_string(), String::new(), "b".to_string()],
            &["b".to_string(), "c".to_string()],
        );
        assert_eq!(merged, vec!["a", "b", "c"]);
    }

    #[test]
    fn messages_are_serialised_as_an_openai_array() {
        let json = messages_json(&[ChatMessage::system("S \"q\""), ChatMessage::user("U")]);
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(
            value,
            serde_json::json!([
                {"role": "system", "content": "S \"q\""},
                {"role": "user", "content": "U"},
            ])
        );
    }
}
