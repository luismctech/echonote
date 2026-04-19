-- EchoNote · initial schema (Sprint 0 day 8)
--
-- Stores meeting headers + segments. Speakers are scaffolded so the
-- Sprint 2 diarizer can populate them without another migration.
-- FTS5 over segments.text lands in Sprint 1 alongside the chat use case.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meetings (
    id              TEXT    PRIMARY KEY NOT NULL,         -- UUIDv7 (lower-case hex with dashes)
    title           TEXT    NOT NULL,
    started_at      TEXT    NOT NULL,                     -- RFC 3339
    ended_at        TEXT,                                 -- NULL while recording
    duration_ms     INTEGER NOT NULL DEFAULT 0,
    language        TEXT,                                 -- dominant detected language (ISO-639-1)
    segment_count   INTEGER NOT NULL DEFAULT 0,
    sample_rate_hz  INTEGER NOT NULL,
    channels        INTEGER NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS meetings_started_at_desc
    ON meetings(started_at DESC);

CREATE TABLE IF NOT EXISTS speakers (
    id          TEXT    PRIMARY KEY NOT NULL,             -- UUIDv7
    meeting_id  TEXT    NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    label       TEXT    NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS speakers_by_meeting ON speakers(meeting_id);

CREATE TABLE IF NOT EXISTS segments (
    id          TEXT    PRIMARY KEY NOT NULL,             -- UUIDv7
    meeting_id  TEXT    NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    start_ms    INTEGER NOT NULL,
    end_ms      INTEGER NOT NULL,
    text        TEXT    NOT NULL,
    speaker_id  TEXT             REFERENCES speakers(id),
    confidence  REAL
) STRICT;

CREATE INDEX IF NOT EXISTS segments_meeting_start
    ON segments(meeting_id, start_ms);
