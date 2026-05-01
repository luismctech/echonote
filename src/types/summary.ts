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
 * note in the architecture docs).
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

export type InterviewQuote = {
  speaker: string;
  quote: string;
  context: string | null;
};

export type Definition = {
  term: string;
  definition: string;
};

/** General-purpose meeting summary (§3.2.1). */
export type GeneralSummary = {
  template: "general";
  summary: string;
  keyPoints: string[];
  decisions: string[];
  actionItems: ActionItem[];
  openQuestions: string[];
};

/** 1:1 manager/report meeting (§3.2.2). */
export type OneOnOneSummary = {
  template: "oneOnOne";
  summary: string;
  wins: string[];
  blockers: string[];
  growthFeedback: string[];
  nextSteps: ActionItem[];
  followUpTopics: string[];
};

/** Sprint review / retrospective (§3.2.3). */
export type SprintReviewSummary = {
  template: "sprintReview";
  summary: string;
  completedItems: string[];
  carryOver: string[];
  risks: string[];
  nextSprintPriorities: string[];
};

/** User research or hiring interview (§3.2.4). */
export type InterviewSummary = {
  template: "interview";
  summary: string;
  quotes: InterviewQuote[];
  themes: string[];
  painPoints: string[];
  opportunities: string[];
};

/** Sales / discovery call (§3.2.5). */
export type SalesCallSummary = {
  template: "salesCall";
  summary: string;
  customerContext: string | null;
  painPoints: string[];
  interestSignals: string[];
  objections: string[];
  nextSteps: ActionItem[];
  dealStageIndicator: string | null;
};

/** Lecture, class, or workshop (§3.2.6). */
export type LectureSummary = {
  template: "lecture";
  summary: string;
  conceptsCovered: string[];
  definitions: Definition[];
  examples: string[];
  homeworkOrNext: string[];
};

/** Fallback when JSON parsing fails twice. */
export type FreeTextSummary = {
  template: "freeText";
  text: string;
};

/** User-defined custom template output. */
export type CustomSummary = {
  template: "custom";
  templateName: string;
  text: string;
};

/** Discriminated union of every supported summary template. */
export type SummaryContent =
  | GeneralSummary
  | OneOnOneSummary
  | SprintReviewSummary
  | InterviewSummary
  | SalesCallSummary
  | LectureSummary
  | FreeTextSummary
  | CustomSummary;

/** User-facing template identifiers (excludes freeText). */
export const TEMPLATE_IDS = [
  "general",
  "oneOnOne",
  "sprintReview",
  "interview",
  "salesCall",
  "lecture",
] as const;

export type TemplateId = (typeof TEMPLATE_IDS)[number];

export const TEMPLATE_LABELS: Record<TemplateId, string> = {
  general: "General",
  oneOnOne: "1:1",
  sprintReview: "Sprint Review",
  interview: "Interview",
  salesCall: "Sales Call",
  lecture: "Lecture",
};

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

// ---------------------------------------------------------------------------
// Streaming summary events (mirrors `echo_app::SummarizeEvent`)
// ---------------------------------------------------------------------------

export type SummarizeEventStarted = { kind: "started"; model: string };
export type SummarizeEventToken = { kind: "token"; delta: string };
export type SummarizeEventCompleted = { kind: "completed"; summary: Summary };
export type SummarizeEventFailed = { kind: "failed"; error: string };

/** Discriminated union of events emitted during streaming summary generation. */
export type SummarizeEvent =
  | SummarizeEventStarted
  | SummarizeEventToken
  | SummarizeEventCompleted
  | SummarizeEventFailed;
