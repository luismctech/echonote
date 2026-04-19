-- EchoNote · LLM summaries (Sprint 1 day 9)
--
-- Stores the structured summary produced by the local LLM (CU-04 in
-- DEVELOPMENT_PLAN.md §3.1). One row per meeting per template, with a
-- unique index that lets the use case "regenerate" a summary by
-- upserting on (meeting_id, template) instead of carrying the previous
-- SummaryId across the wire.
--
-- ## Storage choices
--
-- * `payload` is a JSON blob holding the full
--   `echo_domain::SummaryContent` discriminated union. Keeping it
--   opaque means new templates can be added without a schema change —
--   the column was tagged `TEXT` rather than the SQLite-native `JSON`
--   on purpose: STRICT tables only allow the five primitive type
--   names, and we already round-trip through `serde_json` in the
--   application layer so SQLite-side validation buys us nothing.
-- * `template` mirrors the discriminator inside the JSON payload so
--   we can index it without parsing JSON. Cheap denormalisation that
--   pays for itself the first time the UI needs to filter "show me
--   meetings with a sales-call summary".
-- * `model` is the LLM identifier the summary was produced with
--   (e.g. `qwen2.5-7b-instruct-q4_k_m`). Stored alongside the
--   payload as provenance so a future "regenerate with model X"
--   flow can highlight stale summaries.
-- * `created_at` is RFC 3339 — same convention every other timestamp
--   in this schema uses.
-- * `language` mirrors the meeting's dominant language at summary
--   time so the UI can hint "Generate again in <lang>" when the
--   user changes it.
-- * `ON DELETE CASCADE` against `meetings(id)` keeps the schema
--   self-cleaning: deleting a meeting drops its summaries too.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS summaries (
    id          TEXT    PRIMARY KEY NOT NULL,             -- UUIDv7
    meeting_id  TEXT    NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    template    TEXT    NOT NULL,                         -- "general", "freeText", …
    model       TEXT    NOT NULL,                         -- LLM identifier
    language    TEXT,                                     -- ISO-639-1, NULL when unknown
    created_at  TEXT    NOT NULL,                         -- RFC 3339
    payload     TEXT    NOT NULL                          -- serde_json::to_string(SummaryContent)
) STRICT;

-- One *current* summary per (meeting, template). Upserts on this pair
-- replace the previous row so re-generating a summary doesn't leave
-- stale rows around. If we ever need a history of past summaries this
-- is the constraint to drop and replace with a versioning column.
CREATE UNIQUE INDEX IF NOT EXISTS summaries_meeting_template_unique
    ON summaries(meeting_id, template);

-- Most reads are "give me the most recent summary for this meeting".
-- Index on `meeting_id` so the lookup avoids a full table scan; the
-- unique index above already covers the (meeting_id, template) read
-- path but a separate index on just `meeting_id` keeps the query
-- planner from having to use a partial covering index.
CREATE INDEX IF NOT EXISTS summaries_by_meeting
    ON summaries(meeting_id, created_at DESC);
