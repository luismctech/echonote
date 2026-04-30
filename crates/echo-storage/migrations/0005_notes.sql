-- EchoNote · User notes during recording (Sprint 2)
--
-- Stores timestamped text annotations the user creates while a
-- recording session is active. `timestamp_ms` is relative to the
-- meeting start (same timeline as segments.start_ms), enabling notes
-- and transcript segments to be rendered on a single unified timeline.
--
-- Cascade delete ensures notes are cleaned up when a meeting is
-- removed.

CREATE TABLE IF NOT EXISTS notes (
    id          TEXT    NOT NULL PRIMARY KEY,
    meeting_id  TEXT    NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    text        TEXT    NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    created_at  TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_notes_meeting_ts
    ON notes(meeting_id, timestamp_ms);
