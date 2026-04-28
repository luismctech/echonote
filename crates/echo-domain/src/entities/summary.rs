//! Summary entity.
//!
//! A [`Summary`] is a structured projection over a [`crate::Meeting`]
//! produced by the local LLM (CU-04 in the development plan). Six
//! templates are implemented: General (default), OneOnOne, SprintReview,
//! Interview, SalesCall, and Lecture — each as a variant of
//! [`SummaryContent`].
//!
//! ## Why "content" is a discriminated union
//!
//! The header (`id`, `meeting_id`, `template`, `model`, `created_at`,
//! `language`) is identical across templates, but the payload differs
//! per template (1:1 has `wins`/`blockers`, sales call has
//! `objections`/`deal_stage_indicator`, etc.). Modelling that as a
//! single flat struct with `Option<Vec<…>>` everywhere would push the
//! validation burden onto every consumer; using a tagged enum on the
//! `template` field keeps each template's shape exhaustive at the type
//! level.
//!
//! `serde(rename_all = "camelCase")` keeps the wire format aligned
//! with the rest of the IPC surface (the React app consumes
//! `meetingId`, not `meeting_id`).

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use uuid::Uuid;

use crate::entities::meeting::MeetingId;

/// Strongly-typed identifier for a [`Summary`]. UUIDv7 keeps lexical
/// ordering aligned with creation time, the same convention every
/// other id in this crate uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, specta::Type)]
#[serde(transparent)]
pub struct SummaryId(pub Uuid);

impl SummaryId {
    /// Generate a new UUIDv7 identifier.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }
}

impl Default for SummaryId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for SummaryId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// One actionable item extracted by the summarizer.
///
/// `owner` and `due` are best-effort; the LLM may leave either as
/// `None` when the transcript doesn't pin them down. The UI renders
/// them with sensible placeholders ("unassigned", "no due date") so
/// the summary remains useful even when partial.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct ActionItem {
    /// What needs to be done. Required — an action item without a
    /// task is meaningless.
    pub task: String,
    /// Who is on the hook. `None` when the transcript doesn't say.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner: Option<String>,
    /// Free-form deadline as the LLM extracted it ("Friday", "next
    /// sprint", "2026-05-01"). Kept as a string on purpose — parsing
    /// natural-language dates reliably is a separate problem and the
    /// UI can render the raw text well enough for the MVP.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub due: Option<String>,
}

/// A notable quote attributed to a speaker, used by the Interview
/// template.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct InterviewQuote {
    /// Who said it.
    pub speaker: String,
    /// Verbatim or near-verbatim quote.
    pub quote: String,
    /// Situational context around the quote.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
}

/// A term/definition pair extracted from a lecture or class

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct Definition {
    /// The term being defined.
    pub term: String,
    /// The definition provided.
    pub definition: String,
}

/// Discriminated union over the body of a summary, keyed on the
/// template that produced it.
///
/// Six content templates are implemented.
/// The `#[non_exhaustive]` attribute means consumers must `_ => …`
/// when matching, so adding new variants later is non-breaking.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, specta::Type)]
#[serde(tag = "template", rename_all_fields = "camelCase")]
#[non_exhaustive]
pub enum SummaryContent {
    /// Default template — works for any meeting type (§3.2.1).
    #[serde(rename = "general")]
    General {
        /// 2-3 sentence overview.
        summary: String,
        /// Bulleted highlights.
        #[serde(default)]
        key_points: Vec<String>,
        /// Decisions taken.
        #[serde(default)]
        decisions: Vec<String>,
        /// Action items with optional owner + due.
        #[serde(default)]
        action_items: Vec<ActionItem>,
        /// Questions raised but not answered.
        #[serde(default)]
        open_questions: Vec<String>,
    },

    /// 1:1 meetings between manager and report (§3.2.2).
    #[serde(rename = "oneOnOne")]
    OneOnOne {
        /// 2-3 sentence overview.
        summary: String,
        /// Achievements mentioned.
        #[serde(default)]
        wins: Vec<String>,
        /// Obstacles / blockers.
        #[serde(default)]
        blockers: Vec<String>,
        /// Growth / development feedback.
        #[serde(default)]
        growth_feedback: Vec<String>,
        /// Follow-up action items.
        #[serde(default)]
        next_steps: Vec<ActionItem>,
        /// Topics for the next 1:1.
        #[serde(default)]
        follow_up_topics: Vec<String>,
    },

    /// Sprint review / retrospective (§3.2.3).
    #[serde(rename = "sprintReview")]
    SprintReview {
        /// 2-3 sentence overview.
        summary: String,
        /// Items completed this sprint.
        #[serde(default)]
        completed_items: Vec<String>,
        /// Items carried to next sprint.
        #[serde(default)]
        carry_over: Vec<String>,
        /// Risks identified.
        #[serde(default)]
        risks: Vec<String>,
        /// Priorities for next sprint.
        #[serde(default)]
        next_sprint_priorities: Vec<String>,
    },

    /// User research or hiring interview (§3.2.4).
    #[serde(rename = "interview")]
    Interview {
        /// 2-3 sentence overview.
        summary: String,
        /// Notable quotes from participants.
        #[serde(default)]
        quotes: Vec<InterviewQuote>,
        /// Recurring themes.
        #[serde(default)]
        themes: Vec<String>,
        /// Pain points mentioned.
        #[serde(default)]
        pain_points: Vec<String>,
        /// Opportunities identified.
        #[serde(default)]
        opportunities: Vec<String>,
    },

    /// Sales / discovery call (§3.2.5).
    #[serde(rename = "salesCall")]
    SalesCall {
        /// 2-3 sentence overview.
        summary: String,
        /// Background on the customer.
        #[serde(default)]
        customer_context: Option<String>,
        /// Customer pain points.
        #[serde(default)]
        pain_points: Vec<String>,
        /// Positive interest signals.
        #[serde(default)]
        interest_signals: Vec<String>,
        /// Objections raised.
        #[serde(default)]
        objections: Vec<String>,
        /// Follow-up actions.
        #[serde(default)]
        next_steps: Vec<ActionItem>,
        /// Pipeline stage hint (discovery / evaluation / proposal / negotiation).
        #[serde(default)]
        deal_stage_indicator: Option<String>,
    },

    /// Lecture, class, or workshop (§3.2.6).
    #[serde(rename = "lecture")]
    Lecture {
        /// 2-3 sentence overview.
        summary: String,
        /// Key concepts taught.
        #[serde(default)]
        concepts_covered: Vec<String>,
        /// Term/definition pairs.
        #[serde(default)]
        definitions: Vec<Definition>,
        /// Illustrative examples.
        #[serde(default)]
        examples: Vec<String>,
        /// Homework or next-session topics.
        #[serde(default)]
        homework_or_next: Vec<String>,
    },

    /// Graceful degradation when JSON parsing fails twice.
    #[serde(rename = "freeText")]
    FreeText {
        /// Whatever the model produced, verbatim.
        text: String,
    },

    /// User-defined custom template. The `template_name` identifies
    /// which [`crate::CustomTemplate`] produced this summary.
    #[serde(rename = "custom")]
    Custom {
        /// Display name of the custom template (e.g. "Product Standup").
        template_name: String,
        /// The LLM output, stored verbatim.
        text: String,
    },
}

impl SummaryContent {
    /// Short identifier matching the SQLite `summaries.template`
    /// column — used by the storage adapter, the IPC payloads and the
    /// `echo-proto summarize` CLI.
    #[must_use]
    pub fn template_id(&self) -> &'static str {
        match self {
            SummaryContent::General { .. } => "general",
            SummaryContent::OneOnOne { .. } => "oneOnOne",
            SummaryContent::SprintReview { .. } => "sprintReview",
            SummaryContent::Interview { .. } => "interview",
            SummaryContent::SalesCall { .. } => "salesCall",
            SummaryContent::Lecture { .. } => "lecture",
            SummaryContent::FreeText { .. } => "freeText",
            SummaryContent::Custom { .. } => "custom",
        }
    }
}

/// The six user-facing template identifiers (excludes `freeText`
/// which is a fallback, not something the user selects).
pub const TEMPLATE_IDS: &[&str] = &[
    "general",
    "oneOnOne",
    "sprintReview",
    "interview",
    "salesCall",
    "lecture",
];

/// A persisted LLM summary attached to a [`crate::Meeting`].
///
/// `serde(rename_all = "camelCase")` so the React layer can consume
/// `meetingId` / `createdAt` directly. The `content` field flattens
/// the [`SummaryContent`] enum so JSON readers see one combined
/// document instead of `{ "content": { … } }`, matching how the LLM
/// itself emits the structured output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct Summary {
    /// Stable identifier.
    pub id: SummaryId,
    /// Owning meeting. Each meeting may carry at most one summary
    /// per template at a time — re-running the summarizer
    /// overwrites the previous one (the SQLite adapter enforces
    /// this with a unique index).
    pub meeting_id: MeetingId,
    /// LLM model identifier (e.g. `"qwen2.5-7b-instruct-q4_k_m"`).
    /// Stored alongside the content so a future "regenerate with
    /// model X" flow can compare provenance.
    pub model: String,
    /// ISO-639-1 language the summary was produced in. Echoes the
    /// transcript's dominant language; the LLM is instructed to
    /// answer in that language.
    pub language: Option<String>,
    /// RFC 3339 instant the summary finished generating.
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
    /// Structured payload — discriminated on `template`.
    #[serde(flatten)]
    pub content: SummaryContent,
}

impl Summary {
    /// Convenience accessor mirroring [`SummaryContent::template_id`].
    #[must_use]
    pub fn template(&self) -> &'static str {
        self.content.template_id()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn fixture_general() -> Summary {
        Summary {
            id: SummaryId::new(),
            meeting_id: MeetingId::new(),
            model: "qwen2.5-7b-instruct-q4_k_m".into(),
            language: Some("es".into()),
            created_at: OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap(),
            content: SummaryContent::General {
                summary: "Equipo discutió el roadmap de Q3.".into(),
                key_points: vec!["Migración a SQLite-vec".into(), "Métricas WER".into()],
                decisions: vec!["Postergar v1.0 dos semanas".into()],
                action_items: vec![ActionItem {
                    task: "Preparar bench LLM".into(),
                    owner: Some("Ana".into()),
                    due: Some("viernes".into()),
                }],
                open_questions: vec!["¿Mantenemos Tauri 2?".into()],
            },
        }
    }

    #[test]
    fn summary_id_default_uses_uuid_v7() {
        let id = SummaryId::default();
        assert_eq!(id.0.as_bytes()[6] >> 4, 7, "expected uuid v7");
    }

    #[test]
    fn general_summary_round_trips_with_camelcase_and_template_tag() {
        let s = fixture_general();
        let json = serde_json::to_string(&s).unwrap();

        assert!(json.contains("\"meetingId\""), "got: {json}");
        assert!(json.contains("\"createdAt\""), "got: {json}");
        assert!(json.contains("\"actionItems\""), "got: {json}");
        assert!(json.contains("\"keyPoints\""), "got: {json}");
        assert!(json.contains("\"openQuestions\""), "got: {json}");
        assert!(json.contains("\"template\":\"general\""), "got: {json}");

        let back: Summary = serde_json::from_str(&json).unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn free_text_summary_round_trips() {
        let s = Summary {
            id: SummaryId::new(),
            meeting_id: MeetingId::new(),
            model: "qwen2.5-7b-instruct-q4_k_m".into(),
            language: None,
            created_at: OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap(),
            content: SummaryContent::FreeText {
                text: "El modelo no produjo JSON válido.".into(),
            },
        };
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("\"template\":\"freeText\""), "got: {json}");
        let back: Summary = serde_json::from_str(&json).unwrap();
        assert_eq!(back, s);
        assert_eq!(back.template(), "freeText");
    }

    #[test]
    fn template_id_matches_serde_tag() {
        let cases: Vec<(SummaryContent, &str)> = vec![
            (
                SummaryContent::General {
                    summary: String::new(),
                    key_points: vec![],
                    decisions: vec![],
                    action_items: vec![],
                    open_questions: vec![],
                },
                "general",
            ),
            (
                SummaryContent::OneOnOne {
                    summary: String::new(),
                    wins: vec![],
                    blockers: vec![],
                    growth_feedback: vec![],
                    next_steps: vec![],
                    follow_up_topics: vec![],
                },
                "oneOnOne",
            ),
            (
                SummaryContent::SprintReview {
                    summary: String::new(),
                    completed_items: vec![],
                    carry_over: vec![],
                    risks: vec![],
                    next_sprint_priorities: vec![],
                },
                "sprintReview",
            ),
            (
                SummaryContent::Interview {
                    summary: String::new(),
                    quotes: vec![],
                    themes: vec![],
                    pain_points: vec![],
                    opportunities: vec![],
                },
                "interview",
            ),
            (
                SummaryContent::SalesCall {
                    summary: String::new(),
                    customer_context: None,
                    pain_points: vec![],
                    interest_signals: vec![],
                    objections: vec![],
                    next_steps: vec![],
                    deal_stage_indicator: None,
                },
                "salesCall",
            ),
            (
                SummaryContent::Lecture {
                    summary: String::new(),
                    concepts_covered: vec![],
                    definitions: vec![],
                    examples: vec![],
                    homework_or_next: vec![],
                },
                "lecture",
            ),
            (
                SummaryContent::FreeText {
                    text: String::new(),
                },
                "freeText",
            ),
        ];
        for (content, expected_id) in &cases {
            assert_eq!(content.template_id(), *expected_id);
            let json = serde_json::to_string(content).unwrap();
            assert!(
                json.contains(&format!("\"template\":\"{expected_id}\"")),
                "template_id '{expected_id}' must match serde tag in JSON: {json}"
            );
        }
    }

    #[test]
    fn missing_optional_fields_default_to_empty() {
        // Mirrors what we'll see when the LLM produces a minimal valid
        // output — only `summary` is filled in. The deserializer should
        // accept this and default the rest to empty vectors.
        let json = serde_json::json!({
            "id": SummaryId::new(),
            "meetingId": MeetingId::new(),
            "model": "qwen",
            "language": null,
            "createdAt": "2026-04-18T00:00:00Z",
            "template": "general",
            "summary": "Solo resumen."
        });
        let parsed: Summary = serde_json::from_value(json).unwrap();
        match parsed.content {
            SummaryContent::General {
                key_points,
                decisions,
                action_items,
                open_questions,
                ..
            } => {
                assert!(key_points.is_empty());
                assert!(decisions.is_empty());
                assert!(action_items.is_empty());
                assert!(open_questions.is_empty());
            }
            _ => panic!("expected General"),
        }
    }
}
