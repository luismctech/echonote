//! User-defined summary prompt templates.
//!
//! A [`CustomTemplate`] lets the user define their own system prompt
//! (role + instructions) and optionally a JSON schema hint. The LLM
//! output is stored as [`crate::SummaryContent::Custom`] — free-form
//! text keyed by the template's id.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Strongly-typed identifier for a [`CustomTemplate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, specta::Type)]
#[serde(transparent)]
pub struct CustomTemplateId(pub Uuid);

impl CustomTemplateId {
    /// Generate a new UUIDv7 identifier.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }
}

impl Default for CustomTemplateId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for CustomTemplateId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A user-created summary prompt template.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct CustomTemplate {
    /// Stable identifier.
    pub id: CustomTemplateId,
    /// Short display name shown in the template selector (e.g.
    /// "Product Standup", "Board Meeting").
    pub name: String,
    /// The system-prompt text sent to the LLM. Should describe the
    /// role, desired output format, and any constraints.
    /// Example: "You are a product standup summarizer. Output a JSON
    /// object with keys: blockers, updates, askForHelp."
    pub prompt: String,
}
