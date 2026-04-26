-- EchoNote · speaker slot + nullable label (Sprint 1 day 7 — Fase B)
--
-- The 0001 schema scaffolded `speakers` with a NOT NULL `label`,
-- assuming labels would be set on insert. Sprint 1's diarizer creates
-- speakers anonymously (label is `NULL` until the user renames them)
-- and needs a stable arrival-order `slot` so the UI palette colour
-- survives renames and reloads.
--
-- SQLite STRICT tables don't allow ALTER COLUMN to drop NOT NULL or
-- add a UNIQUE constraint, so we rebuild the table — the canonical
-- SQLite migration pattern. Existing rows (if any) get sequential
-- slots derived from `id` ordering, which is consistent with UUIDv7's
-- creation-time ordering.

PRAGMA foreign_keys = OFF;

CREATE TABLE speakers_new (
    id          TEXT    PRIMARY KEY NOT NULL,             -- UUIDv7
    meeting_id  TEXT    NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    slot        INTEGER NOT NULL,                         -- 0-based arrival order within the meeting
    label       TEXT,                                     -- user-assigned name; NULL ⇒ "Speaker {slot+1}"
    UNIQUE (meeting_id, slot)
) STRICT;

INSERT INTO speakers_new (id, meeting_id, slot, label)
SELECT
    id,
    meeting_id,
    -- Replay arrival order from the only stable signal we have on
    -- legacy rows: the UUIDv7 sort. New inserts will set slot
    -- explicitly via upsert_speaker.
    ROW_NUMBER() OVER (PARTITION BY meeting_id ORDER BY id) - 1 AS slot,
    label
FROM speakers;

DROP TABLE speakers;
ALTER TABLE speakers_new RENAME TO speakers;

CREATE INDEX IF NOT EXISTS speakers_by_meeting ON speakers(meeting_id);

PRAGMA foreign_keys = ON;
