/**
 * LLM summary types — mirror of `echo_domain::Summary` and the
 * `SummaryContent` discriminated union exposed by the
 * `summarize_meeting` / `get_summary` Tauri commands.
 *
 * Wire format note: the Rust enum uses
 * `#[serde(tag = "template", rename_all_fields = "camelCase")]`,
 * so each variant arrives as a `template`-tagged object with the
 * variant's fields flattened into the top-level Summary alongside
 * `id`, `meetingId`, `model`, `language`, `createdAt`. Add a new
 * variant here whenever a new template ships in the domain layer
 * (kept in lockstep manually until `tauri-specta` lands — see ADR
 * note in `docs/adr/0002-rust-plus-react-stack.md`).
 */

import type { MeetingId } from "./meeting";

/** UUIDv7 string identifying a persisted summary. */
export type SummaryId = string;

/**
 * One concrete to-do extracted from the transcript. `owner` and
 * `due` are best-effort: the LLM omits them when the meeting did not
 * specify either, so the UI must tolerate `null`.
 */
export type ActionItem = {
  task: string;
  owner: string | null;
  due: string | null;
};

/** General-purpose meeting summary (the only template shipping in v1). */
export type GeneralSummary = {
  template: "general";
  /** 2–3 sentence narrative recap. */
  summary: string;
  /** Bulleted highlights — empty when the meeting was light. */
  keyPoints: string[];
  /** Decisions taken during the meeting. */
  decisions: string[];
  /** Action items with optional owner/due metadata. */
  actionItems: ActionItem[];
  /** Questions raised but not answered. */
  openQuestions: string[];
};

/**
 * Fallback variant: used by the use case when the LLM keeps
 * returning malformed JSON after one corrective retry. The frontend
 * renders it as a single text block with a "Could not parse —
 * regenerate?" affordance.
 */
export type FreeTextSummary = {
  template: "freeText";
  text: string;
};

/** Discriminated union of every supported summary template. */
export type SummaryContent = GeneralSummary | FreeTextSummary;

/**
 * The persisted summary as the backend returns it. The
 * `SummaryContent` fields are flattened in via `#[serde(flatten)]`
 * on the Rust side — that's why this type intersects with the
 * union rather than nesting it under a `content` key.
 */
export type Summary = SummaryContent & {
  id: SummaryId;
  meetingId: MeetingId;
  /** LLM identifier the summary was produced with (e.g. `qwen2.5-7b…`). */
  model: string;
  /** ISO-639-1 language hint that fed the prompt; `null` when unknown. */
  language: string | null;
  /** RFC 3339 timestamp of generation. */
  createdAt: string;
};
