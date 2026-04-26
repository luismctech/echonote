/**
 * Meeting + segment domain types.
 *
 * Mirrors the `Meeting` aggregate exposed by `get_meeting` /
 * `list_meetings` / `search_meetings`. `Segment` lives here (not in
 * `streaming.ts`) because both the persisted aggregate and the live
 * `TranscriptEvent.chunk` events carry segments — keeping a single
 * definition avoids drift between the live and replay views.
 */

import type { AudioFormat } from "./streaming";
import type { Speaker } from "./speaker";

/** UUIDv7 string identifying a persisted meeting. */
export type MeetingId = string;

/** A single transcribed span within a meeting. */
export type Segment = {
  id: string;
  startMs: number;
  endMs: number;
  text: string;
  speakerId: string | null;
  confidence: number | null;
};

/** Lightweight projection used by the sidebar listing. */
export type MeetingSummary = {
  id: MeetingId;
  title: string;
  startedAt: string;
  endedAt: string | null;
  durationMs: number;
  language: string | null;
  segmentCount: number;
};

/** Full meeting aggregate (header + segments + speakers). */
export type Meeting = MeetingSummary & {
  inputFormat: AudioFormat;
  segments: Segment[];
  /** Diarized speakers, ordered by `slot` ascending. May be empty. */
  speakers: Speaker[];
};

/**
 * One hit returned by the FTS5 search.
 *
 * The backend collapses results to one row per meeting and sorts by
 * BM25 rank ascending — *smaller is better* (negative numbers are
 * the strongest matches). The UI must preserve that ordering.
 *
 * `snippet` is pre-rendered with `<mark>...</mark>` markers around
 * the matched terms. Render with `dangerouslySetInnerHTML` — the
 * markers are emitted by SQLite over text we ourselves indexed, so
 * the XSS surface is the same as showing a raw segment body.
 */
export type MeetingSearchHit = {
  meeting: MeetingSummary;
  snippet: string;
  rank: number;
};
