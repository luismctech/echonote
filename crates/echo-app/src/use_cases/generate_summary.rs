//! `generate_summary` use case (CU-04 in `docs/DEVELOPMENT_PLAN.md`).
//!
//! Loads a finalized meeting from the store, asks the local LLM to
//! produce a structured summary using the **General** template, parses
//! the JSON response and persists it back to the store. Other templates
//! (1:1, sprint review, interview, sales call, lecture) reuse this
//! plumbing in Sprint 2 — only the prompt and the
//! [`echo_domain::SummaryContent`] variant change.
//!
//! ## Reliability
//!
//! Local 7-8B quantized models occasionally produce JSON that's not
//! quite RFC-8259 compliant (trailing commas, unescaped newlines inside
//! strings, prose before/after the block). The use case mitigates this
//! with three layers:
//!
//! 1. The system prompt forbids prose and pins the schema.
//! 2. We strip the first ```json … ``` fence (or any `{ … }` block) out
//!    of the response before parsing — many models emit a markdown
//!    fence even when explicitly told not to.
//! 3. On the first parse failure we retry once, this time including
//!    the parser's error message in the user turn so the model can
//!    self-correct (the same trick OpenAI's "function calling" used
//!    before structured outputs).
//! 4. If parsing still fails, we fall back to
//!    [`echo_domain::SummaryContent::FreeText`] so the user always
//!    sees *something* and the meeting isn't blocked behind a flaky
//!    model.
//!
//! The fallback path is reported via tracing so we can monitor how
//! often it fires once telemetry lands.

use std::sync::Arc;

use thiserror::Error;
use time::OffsetDateTime;
use tracing::{info, instrument, warn};

use echo_domain::{
    DomainError, GenerateOptions, LlmModel, Meeting, MeetingId, MeetingStore, Speaker, Summary,
    SummaryContent, SummaryId,
};

/// Maximum characters of transcript text fed to the model. Qwen 3
/// (and Qwen 2.5 as legacy fallback) ship with 32 k+ context but the
/// KV cache cost scales linearly with it; ~6 k characters fits ~30
/// minutes of speech and stays well within a 4 k token budget after
/// Qwen's BPE tokenization (~3.5 chars/token in Spanish text). Longer
/// meetings are summarised on a head + tail window — see
/// [`render_transcript`].
const MAX_TRANSCRIPT_CHARS: usize = 6_000;

/// Token budget for the model's response. Generous enough for a full
/// general summary with several action items; small enough that a
/// runaway loop is bounded.
const SUMMARY_MAX_TOKENS: u32 = 1_024;

/// Errors the use case can surface. Mirrors the structure used by
/// [`crate::RenameSpeaker`]: typed cases for things the IPC layer
/// must distinguish (`NotFound`, `EmptyTranscript`), a single
/// pass-through for storage failures and a dedicated variant for LLM
/// failures.
#[derive(Debug, Error)]
pub enum SummarizeMeetingError {
    /// The meeting id is unknown to the store.
    #[error("meeting {0} not found")]
    NotFound(MeetingId),

    /// The meeting exists but has no transcribed text yet — there's
    /// nothing for the LLM to summarise. Surfaced explicitly so the UI
    /// can show "wait until the recording finishes" instead of a
    /// generic error.
    #[error("meeting {0} has no transcript text to summarize")]
    EmptyTranscript(MeetingId),

    /// The LLM adapter failed (load, decode, OOM, …). Wrapped so the
    /// caller doesn't need to import `DomainError`.
    #[error("llm failed: {0}")]
    Llm(DomainError),

    /// Storage layer failure (disk full, schema mismatch, …).
    #[error(transparent)]
    Storage(DomainError),
}

/// Use-case handler. Holding both ports as `Arc<dyn …>` keeps the
/// IPC and CLI layers decoupled from concrete adapters.
pub struct SummarizeMeeting {
    llm: Arc<dyn LlmModel>,
    store: Arc<dyn MeetingStore>,
}

impl SummarizeMeeting {
    /// Wire the use case against concrete adapters.
    #[must_use]
    pub fn new(llm: Arc<dyn LlmModel>, store: Arc<dyn MeetingStore>) -> Self {
        Self { llm, store }
    }

    /// Generate (and persist) the General-template summary for a
    /// meeting. Returns the freshly persisted [`Summary`] so the
    /// caller can render it without an extra round-trip.
    #[instrument(skip(self), fields(meeting_id = %meeting_id))]
    pub async fn execute(&self, meeting_id: MeetingId) -> Result<Summary, SummarizeMeetingError> {
        let meeting = self
            .store
            .get(meeting_id)
            .await
            .map_err(SummarizeMeetingError::Storage)?
            .ok_or(SummarizeMeetingError::NotFound(meeting_id))?;

        let transcript = render_transcript(&meeting);
        if transcript.trim().is_empty() {
            return Err(SummarizeMeetingError::EmptyTranscript(meeting_id));
        }

        let language = meeting.summary.language.clone();
        let language_instruction = language_instruction(language.as_deref());

        // ---- first attempt -------------------------------------------------
        let first_prompt = build_general_prompt(&transcript, &language_instruction, None);
        let first_response = self
            .llm
            .generate(&first_prompt, &generate_opts())
            .await
            .map_err(SummarizeMeetingError::Llm)?;

        let content = match parse_general_payload(&first_response) {
            Ok(c) => c,
            Err(first_err) => {
                warn!(error = %first_err, "summary JSON parse failed on first try, retrying");

                // ---- retry with parser feedback -----------------------------
                let retry_prompt =
                    build_general_prompt(&transcript, &language_instruction, Some(&first_err));
                let retry_response = self
                    .llm
                    .generate(&retry_prompt, &generate_opts())
                    .await
                    .map_err(SummarizeMeetingError::Llm)?;

                match parse_general_payload(&retry_response) {
                    Ok(c) => c,
                    Err(retry_err) => {
                        warn!(
                            error = %retry_err,
                            "summary JSON parse failed twice; falling back to free text"
                        );
                        SummaryContent::FreeText {
                            text: retry_response,
                        }
                    }
                }
            }
        };

        let summary = Summary {
            id: SummaryId::new(),
            meeting_id,
            model: self.llm.model_id().to_string(),
            language,
            created_at: OffsetDateTime::now_utc(),
            content,
        };

        self.store
            .upsert_summary(&summary)
            .await
            .map_err(SummarizeMeetingError::Storage)?;

        info!(
            template = summary.template(),
            model = %summary.model,
            "summary persisted"
        );
        Ok(summary)
    }
}

/// Default generation knobs for summaries. Low temperature + nucleus
/// sampling biases the model toward valid JSON; the explicit stop
/// sequences cover the most common Qwen / Llama chat-template
/// terminators so we don't keep decoding past the assistant turn.
fn generate_opts() -> GenerateOptions {
    GenerateOptions {
        max_tokens: SUMMARY_MAX_TOKENS,
        temperature: 0.2,
        top_p: 0.95,
        seed: None,
        stop: vec!["<|im_end|>".into(), "<|eot_id|>".into(), "</s>".into()],
    }
}

/// Render the meeting's transcript into a single block of text the
/// LLM can ingest. Speakers are inlined as `Speaker N:` (or the user
/// label, if assigned) so the model can attribute action items.
///
/// When the transcript is longer than [`MAX_TRANSCRIPT_CHARS`] we keep
/// the first 2/3 from the head and the last 1/3 from the tail with
/// an explicit `[…elision…]` marker. This is good enough for the MVP
/// — long-meeting chunked summarisation lands in Sprint 2 with the
/// chat use case.
fn render_transcript(meeting: &Meeting) -> String {
    let mut buf = String::new();
    for seg in &meeting.segments {
        let label = seg
            .speaker_id
            .and_then(|id| meeting.speakers.iter().find(|s| s.id == id))
            .map(speaker_display_name)
            .unwrap_or_else(|| "Speaker".to_string());
        let text = seg.text.trim();
        if text.is_empty() {
            continue;
        }
        if !buf.is_empty() {
            buf.push('\n');
        }
        buf.push_str(&label);
        buf.push_str(": ");
        buf.push_str(text);
    }

    if buf.chars().count() <= MAX_TRANSCRIPT_CHARS {
        return buf;
    }

    // Head (4/6) + tail (2/6) split with an explicit elision marker.
    let head_chars = MAX_TRANSCRIPT_CHARS * 4 / 6;
    let tail_chars = MAX_TRANSCRIPT_CHARS - head_chars;
    let head: String = buf.chars().take(head_chars).collect();
    let tail: String = buf
        .chars()
        .rev()
        .take(tail_chars)
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    format!("{head}\n[… (transcript truncated for length) …]\n{tail}")
}

fn speaker_display_name(s: &Speaker) -> String {
    s.label
        .clone()
        .unwrap_or_else(|| format!("Speaker {}", s.slot + 1))
}

fn language_instruction(language: Option<&str>) -> String {
    match language.unwrap_or("").to_ascii_lowercase().as_str() {
        // Spanish + Latin-American Spanish variants.
        "es" | "es-mx" | "es-es" | "es-419" => "Responde SIEMPRE en español neutro.".to_string(),
        // English. Defaulted to when the meeting has no language tag.
        "" | "en" | "en-us" | "en-gb" => "Always respond in English.".to_string(),
        // Anything else: instruct the model to mirror the source.
        other => format!(
            "Always respond in the same language as the meeting transcript (ISO code: {other})."
        ),
    }
}

/// Build the chat-template-wrapped prompt. The Qwen 3 family is the
/// default model for the MVP (Qwen 2.5 is a legacy fallback), and both
/// share the same `<|im_start|>` template, so a single prompt covers
/// both. Most other modern instruct GGUFs (Llama 3, Mistral, Phi 3)
/// tolerate it as a literal user message and still produce reasonable
/// output, which is good enough for the day-9 single-template scope.
fn build_general_prompt(
    transcript: &str,
    language_instruction: &str,
    parser_feedback: Option<&str>,
) -> String {
    // The schema lives in its own raw string so we can keep the
    // double quotes verbatim (Rust string escaping inside `format!`
    // would make the JSON unreadable). Lines are deliberately kept
    // short so the model sees one field per line.
    const SCHEMA: &str = r#"Schema:
{
  "summary": string,                        // 2-3 sentences
  "keyPoints": string[],                    // bulleted highlights
  "decisions": string[],                    // decisions taken
  "actionItems": [                           // owners/dues optional
    { "task": string, "owner": string|null, "due": string|null }
  ],
  "openQuestions": string[]                 // unanswered questions
}"#;

    let system = format!(
        "You are a meeting summarizer. {language_instruction}\n\
         Output ONLY a single JSON object that matches the schema below — no prose, \
         no markdown fences, no commentary. All string values must be valid UTF-8 and \
         all arrays may be empty when the transcript does not contain that information.\n\n\
         {SCHEMA}"
    );

    let mut user = format!("Transcript:\n---\n{transcript}\n---\n\nReturn the JSON object only.");

    if let Some(err) = parser_feedback {
        // Self-correction turn: surface the parser's complaint so the
        // model knows what to fix.
        user.push_str(&format!(
            "\n\nYour previous response could not be parsed as JSON. \
             Parser error: {err}\n\
             Return ONLY a valid JSON object that matches the schema. \
             Do not include any text before or after the JSON."
        ));
    }

    format!(
        "<|im_start|>system\n{system}<|im_end|>\n\
         <|im_start|>user\n{user}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

/// Pull the JSON payload out of the model's response and parse it
/// into [`SummaryContent::General`].
///
/// The matcher is permissive on purpose — local quantized models love
/// to wrap JSON in ```json fences or prefix it with "Here is the
/// summary:". We grab the substring between the first `{` and the
/// matching `}` (counting braces, ignoring those inside strings) so
/// markdown wrappers don't break parsing.
fn parse_general_payload(raw: &str) -> Result<SummaryContent, String> {
    let block =
        extract_json_object(raw).ok_or_else(|| "no JSON object found in response".to_string())?;

    // The on-the-wire shape uses camelCase already (matching our
    // `rename_all_fields = "camelCase"` enum), but we still need to
    // hand-pluck the fields rather than deserializing into
    // `SummaryContent` directly because the LLM doesn't know about the
    // `template` discriminator. Decode into an untyped value first,
    // then map.
    let value: serde_json::Value =
        serde_json::from_str(block).map_err(|e| format!("invalid JSON: {e}"))?;
    let obj = value
        .as_object()
        .ok_or_else(|| "expected a JSON object".to_string())?;

    let summary = obj
        .get("summary")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "missing required field `summary`".to_string())?
        .to_string();

    let key_points = string_array(obj.get("keyPoints"));
    let decisions = string_array(obj.get("decisions"));
    let open_questions = string_array(obj.get("openQuestions"));
    let action_items = action_items_array(obj.get("actionItems"));

    Ok(SummaryContent::General {
        summary,
        key_points,
        decisions,
        action_items,
        open_questions,
    })
}

fn string_array(v: Option<&serde_json::Value>) -> Vec<String> {
    v.and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| item.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

fn action_items_array(v: Option<&serde_json::Value>) -> Vec<echo_domain::ActionItem> {
    v.and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| {
                    let obj = item.as_object()?;
                    let task = obj.get("task").and_then(|v| v.as_str())?.to_string();
                    let owner = obj
                        .get("owner")
                        .and_then(|v| v.as_str().map(str::to_string));
                    let due = obj.get("due").and_then(|v| v.as_str().map(str::to_string));
                    Some(echo_domain::ActionItem { task, owner, due })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Find the first balanced JSON object in `raw`, ignoring braces that
/// appear inside string literals. Returns the substring including the
/// outer braces, or `None` when no balanced object is found.
fn extract_json_object(raw: &str) -> Option<&str> {
    let bytes = raw.as_bytes();
    let start = bytes.iter().position(|&b| b == b'{')?;

    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;

    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&raw[start..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use pretty_assertions::assert_eq;
    use time::OffsetDateTime;

    use echo_domain::{
        ActionItem, AudioFormat, CreateMeeting, FinalizeMeeting, MeetingSearchHit, MeetingSummary,
        Segment, SegmentId, SpeakerId,
    };

    /// In-memory store with summary support. Other methods panic so
    /// accidental new dependencies fail loudly during tests.
    #[derive(Default)]
    struct FakeStore {
        meetings: Mutex<Vec<Meeting>>,
        summaries: Mutex<Vec<Summary>>,
    }

    #[async_trait]
    impl MeetingStore for FakeStore {
        async fn create(&self, _: CreateMeeting) -> Result<MeetingSummary, DomainError> {
            unreachable!()
        }
        async fn append_segments(&self, _: MeetingId, _: &[Segment]) -> Result<(), DomainError> {
            unreachable!()
        }
        async fn upsert_speaker(&self, _: MeetingId, _: &Speaker) -> Result<(), DomainError> {
            unreachable!()
        }
        async fn list_speakers(&self, _: MeetingId) -> Result<Vec<Speaker>, DomainError> {
            unreachable!()
        }
        async fn rename_speaker(
            &self,
            _: MeetingId,
            _: SpeakerId,
            _: Option<&str>,
        ) -> Result<bool, DomainError> {
            unreachable!()
        }
        async fn finalize(
            &self,
            _: MeetingId,
            _: FinalizeMeeting,
        ) -> Result<MeetingSummary, DomainError> {
            unreachable!()
        }
        async fn list(&self, _: u32) -> Result<Vec<MeetingSummary>, DomainError> {
            unreachable!()
        }
        async fn get(&self, meeting_id: MeetingId) -> Result<Option<Meeting>, DomainError> {
            Ok(self
                .meetings
                .lock()
                .unwrap()
                .iter()
                .find(|m| m.summary.id == meeting_id)
                .cloned())
        }
        async fn delete(&self, _: MeetingId) -> Result<bool, DomainError> {
            unreachable!()
        }
        async fn search(&self, _: &str, _: u32) -> Result<Vec<MeetingSearchHit>, DomainError> {
            unreachable!()
        }
        async fn upsert_summary(&self, summary: &Summary) -> Result<(), DomainError> {
            let mut guard = self.summaries.lock().unwrap();
            // Replace existing summary for the same meeting.
            guard.retain(|s| s.meeting_id != summary.meeting_id);
            guard.push(summary.clone());
            Ok(())
        }
        async fn get_summary(&self, meeting_id: MeetingId) -> Result<Option<Summary>, DomainError> {
            Ok(self
                .summaries
                .lock()
                .unwrap()
                .iter()
                .find(|s| s.meeting_id == meeting_id)
                .cloned())
        }
    }

    /// Scripted LLM. Each `generate` call returns the next response
    /// from a queue; if the queue is exhausted the test panics so
    /// runaway retries are visible.
    struct ScriptedLlm {
        responses: Mutex<Vec<Result<String, DomainError>>>,
        prompts: Mutex<Vec<String>>,
        model_id: String,
    }

    impl ScriptedLlm {
        fn new(responses: Vec<Result<String, DomainError>>) -> Self {
            Self {
                responses: Mutex::new(responses.into_iter().rev().collect()),
                prompts: Mutex::new(Vec::new()),
                model_id: "fake-llm".into(),
            }
        }

        fn calls(&self) -> usize {
            self.prompts.lock().unwrap().len()
        }
    }

    #[async_trait]
    impl LlmModel for ScriptedLlm {
        fn model_id(&self) -> &str {
            &self.model_id
        }
        async fn generate(&self, prompt: &str, _: &GenerateOptions) -> Result<String, DomainError> {
            self.prompts.lock().unwrap().push(prompt.to_string());
            self.responses
                .lock()
                .unwrap()
                .pop()
                .expect("ScriptedLlm: no scripted response left")
        }
    }

    fn seed_meeting(store: &FakeStore, segments: Vec<&str>) -> MeetingId {
        let id = MeetingId::new();
        let mut segs = Vec::new();
        for (i, text) in segments.iter().enumerate() {
            segs.push(Segment {
                id: SegmentId::new(),
                start_ms: (i * 1_000) as u32,
                end_ms: ((i + 1) * 1_000) as u32,
                text: (*text).to_string(),
                speaker_id: None,
                confidence: None,
            });
        }
        store.meetings.lock().unwrap().push(Meeting {
            summary: MeetingSummary {
                id,
                title: "T".into(),
                started_at: OffsetDateTime::now_utc(),
                ended_at: None,
                duration_ms: segments.len() as u32 * 1_000,
                language: Some("es".into()),
                segment_count: segments.len() as u32,
            },
            input_format: AudioFormat::WHISPER,
            segments: segs,
            speakers: vec![],
        });
        id
    }

    #[tokio::test]
    async fn happy_path_persists_general_summary() {
        let store = Arc::new(FakeStore::default());
        let id = seed_meeting(&store, vec!["Hola equipo", "Decidimos lanzar el lunes"]);
        let llm = Arc::new(ScriptedLlm::new(vec![Ok(r#"
            {
              "summary": "Reunión de planeación.",
              "keyPoints": ["Lanzamos lunes"],
              "decisions": ["Aprobar release v1"],
              "actionItems": [
                { "task": "Revisar QA", "owner": "Ana", "due": "domingo" }
              ],
              "openQuestions": []
            }
        "#
        .into())]));

        let uc = SummarizeMeeting::new(llm.clone(), store.clone());
        let summary = uc.execute(id).await.unwrap();

        assert_eq!(llm.calls(), 1);
        assert_eq!(summary.template(), "general");
        assert_eq!(summary.meeting_id, id);
        assert_eq!(summary.model, "fake-llm");
        match &summary.content {
            SummaryContent::General {
                summary: text,
                key_points,
                decisions,
                action_items,
                open_questions,
            } => {
                assert_eq!(text, "Reunión de planeación.");
                assert_eq!(key_points, &vec!["Lanzamos lunes".to_string()]);
                assert_eq!(decisions, &vec!["Aprobar release v1".to_string()]);
                assert_eq!(
                    action_items,
                    &vec![ActionItem {
                        task: "Revisar QA".into(),
                        owner: Some("Ana".into()),
                        due: Some("domingo".into()),
                    }]
                );
                assert!(open_questions.is_empty());
            }
            other => panic!("expected General, got {other:?}"),
        }

        // Round-trip through the store.
        let fetched = store.get_summary(id).await.unwrap().unwrap();
        assert_eq!(fetched.id, summary.id);
    }

    #[tokio::test]
    async fn malformed_first_then_valid_retry_succeeds() {
        let store = Arc::new(FakeStore::default());
        let id = seed_meeting(&store, vec!["Hola"]);
        let llm = Arc::new(ScriptedLlm::new(vec![
            Ok("not json at all, the model went sideways".into()),
            Ok(r#"```json
            { "summary": "Saludo breve.", "keyPoints": [], "decisions": [],
              "actionItems": [], "openQuestions": [] }
            ```"#
                .into()),
        ]));

        let uc = SummarizeMeeting::new(llm.clone(), store.clone());
        let summary = uc.execute(id).await.unwrap();

        assert_eq!(llm.calls(), 2, "expected one retry");
        assert_eq!(summary.template(), "general");
        // The retry prompt should have included the parser error so
        // the model could self-correct.
        let prompts = llm.prompts.lock().unwrap();
        assert!(
            prompts[1].contains("could not be parsed"),
            "retry prompt should include parser feedback: {prompts:?}"
        );
    }

    #[tokio::test]
    async fn double_failure_falls_back_to_free_text() {
        let store = Arc::new(FakeStore::default());
        let id = seed_meeting(&store, vec!["Hola"]);
        let llm = Arc::new(ScriptedLlm::new(vec![
            Ok("garbage 1".into()),
            Ok("garbage 2 — still no JSON here".into()),
        ]));

        let uc = SummarizeMeeting::new(llm.clone(), store.clone());
        let summary = uc.execute(id).await.unwrap();

        assert_eq!(llm.calls(), 2);
        match summary.content {
            SummaryContent::FreeText { text } => {
                assert_eq!(text, "garbage 2 — still no JSON here");
            }
            other => panic!("expected FreeText fallback, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn empty_meeting_short_circuits_without_calling_llm() {
        let store = Arc::new(FakeStore::default());
        // Meeting with whitespace-only segments.
        let id = seed_meeting(&store, vec!["   ", "\n"]);
        let llm = Arc::new(ScriptedLlm::new(vec![]));

        let uc = SummarizeMeeting::new(llm.clone(), store.clone());
        let err = uc.execute(id).await.unwrap_err();
        assert!(
            matches!(err, SummarizeMeetingError::EmptyTranscript(mid) if mid == id),
            "got {err:?}"
        );
        assert_eq!(llm.calls(), 0, "LLM must not be invoked for empty input");
    }

    #[tokio::test]
    async fn unknown_meeting_returns_not_found() {
        let store = Arc::new(FakeStore::default());
        let llm = Arc::new(ScriptedLlm::new(vec![]));

        let uc = SummarizeMeeting::new(llm.clone(), store.clone());
        let err = uc.execute(MeetingId::new()).await.unwrap_err();
        assert!(matches!(err, SummarizeMeetingError::NotFound(_)), "{err:?}");
    }

    #[tokio::test]
    async fn llm_failure_propagates_as_llm_error() {
        let store = Arc::new(FakeStore::default());
        let id = seed_meeting(&store, vec!["Hola"]);
        let llm = Arc::new(ScriptedLlm::new(vec![Err(DomainError::LlmFailed(
            "OOM".into(),
        ))]));

        let uc = SummarizeMeeting::new(llm.clone(), store.clone());
        let err = uc.execute(id).await.unwrap_err();
        assert!(matches!(err, SummarizeMeetingError::Llm(_)), "{err:?}");
    }

    #[test]
    fn extract_json_object_handles_markdown_fences() {
        let raw = "Here is the summary:\n\n```json\n{\n  \"a\": 1\n}\n```\nbye";
        let block = extract_json_object(raw).unwrap();
        assert_eq!(block, "{\n  \"a\": 1\n}");
    }

    #[test]
    fn extract_json_object_ignores_braces_inside_strings() {
        // The closing brace inside the string must not terminate the
        // outer object.
        let raw = r#"{ "a": "hello } world", "b": 1 }"#;
        let block = extract_json_object(raw).unwrap();
        assert_eq!(block, raw);
    }

    #[test]
    fn extract_json_object_returns_none_for_no_object() {
        assert!(extract_json_object("nothing here").is_none());
        assert!(extract_json_object("{ unbalanced").is_none());
    }

    #[test]
    fn render_transcript_uses_speaker_labels() {
        let mut meeting = Meeting {
            summary: MeetingSummary {
                id: MeetingId::new(),
                title: "T".into(),
                started_at: OffsetDateTime::now_utc(),
                ended_at: None,
                duration_ms: 0,
                language: None,
                segment_count: 0,
            },
            input_format: AudioFormat::WHISPER,
            segments: vec![],
            speakers: vec![],
        };
        let s0 = Speaker::anonymous(0);
        let s1 = Speaker::anonymous(1).renamed("Ana");
        meeting.segments.push(Segment {
            id: SegmentId::new(),
            start_ms: 0,
            end_ms: 1_000,
            text: "Hola".into(),
            speaker_id: Some(s0.id),
            confidence: None,
        });
        meeting.segments.push(Segment {
            id: SegmentId::new(),
            start_ms: 1_000,
            end_ms: 2_000,
            text: "Buenos días".into(),
            speaker_id: Some(s1.id),
            confidence: None,
        });
        meeting.speakers = vec![s0, s1];

        let rendered = render_transcript(&meeting);
        assert!(rendered.contains("Speaker 1: Hola"), "{rendered}");
        assert!(rendered.contains("Ana: Buenos días"), "{rendered}");
    }

    #[test]
    fn render_transcript_truncates_long_input_with_marker() {
        let mut meeting = Meeting {
            summary: MeetingSummary {
                id: MeetingId::new(),
                title: "T".into(),
                started_at: OffsetDateTime::now_utc(),
                ended_at: None,
                duration_ms: 0,
                language: None,
                segment_count: 0,
            },
            input_format: AudioFormat::WHISPER,
            segments: vec![],
            speakers: vec![],
        };
        let chunk = "x".repeat(MAX_TRANSCRIPT_CHARS);
        meeting.segments.push(Segment {
            id: SegmentId::new(),
            start_ms: 0,
            end_ms: 1_000,
            text: chunk.clone(),
            speaker_id: None,
            confidence: None,
        });
        meeting.segments.push(Segment {
            id: SegmentId::new(),
            start_ms: 1_000,
            end_ms: 2_000,
            text: "TAIL".into(),
            speaker_id: None,
            confidence: None,
        });
        let rendered = render_transcript(&meeting);
        assert!(rendered.contains("transcript truncated"), "{rendered}");
        assert!(rendered.ends_with("TAIL"), "tail should be preserved");
    }

    #[test]
    fn language_instruction_picks_spanish_for_es_codes() {
        assert!(language_instruction(Some("es")).contains("español"));
        assert!(language_instruction(Some("ES-MX")).contains("español"));
        assert!(language_instruction(Some("en")).contains("English"));
        assert!(language_instruction(None).contains("English"));
        let other = language_instruction(Some("fr"));
        assert!(other.contains("fr"), "{other}");
    }
}
