-- EchoNote · FTS5 search over segment text (Sprint 1 day 8)
--
-- A flat scan of `meetings` is fine for ~30 rows but degrades fast,
-- and segment text was unsearchable until now (no `LIKE` index either).
-- This migration introduces an external-content FTS5 virtual table
-- mirroring `segments.text`, kept in sync via three triggers, plus a
-- one-shot backfill so existing meetings are searchable from day one.
--
-- Tokenizer: `unicode61 remove_diacritics 2` is the modern choice for
-- Spanish (and Latin-based languages in general):
--
--   * `unicode61` lowercases via NFKC + Unicode case folding, so
--     "García" matches "garcia".
--   * `remove_diacritics 2` is the post-Unicode-9 variant that handles
--     codepoints introduced after the original `1` table; the older
--     `1` is buggy on some accents. SQLite doesn't degrade gracefully
--     when both sides disagree on the diacritics setting, so it's
--     pinned here and the implementation queries with the same.
--
-- Storage choice: `content='segments'` makes this an external-content
-- FTS5 table — SQLite stores only the inverted index, and reads the
-- actual text back from `segments` when computing snippets. This is
-- simpler than a contentless table because triggers only need to pass
-- `(rowid, text)`; we don't have to mirror UNINDEXED columns. The
-- `meeting_id` join happens at query time via `segments.rowid`, which
-- is fine because `segments_meeting_start` already indexes that join.
--
-- Ranking is left to FTS5's built-in BM25 (see `bm25(segments_fts)`
-- in queries). Single-column FTS so no per-column weighting needed.

CREATE VIRTUAL TABLE segments_fts USING fts5(
    text,
    content='segments',
    content_rowid='rowid',
    tokenize="unicode61 remove_diacritics 2"
);

-- One-shot backfill so meetings recorded before this migration are
-- searchable. `rowid` of the FTS row mirrors `segments.rowid` so we
-- can join back without an extra id mapping.
INSERT INTO segments_fts (rowid, text)
SELECT rowid, text FROM segments;

-- Sync triggers. FTS5 has no foreign-key support, so we mirror
-- segment changes manually. All three are AFTER triggers because
-- FTS5 row updates must reference an existing rowid in the base
-- table. The 'delete' / 'insert' magic command on the table-name
-- column is the documented idiom for keeping external-content tables
-- in sync (see "External Content and Contentless Tables" in the
-- SQLite FTS5 docs).
CREATE TRIGGER segments_ai AFTER INSERT ON segments BEGIN
    INSERT INTO segments_fts (rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TRIGGER segments_ad AFTER DELETE ON segments BEGIN
    INSERT INTO segments_fts (segments_fts, rowid, text)
    VALUES ('delete', old.rowid, old.text);
END;

CREATE TRIGGER segments_au AFTER UPDATE ON segments BEGIN
    INSERT INTO segments_fts (segments_fts, rowid, text)
    VALUES ('delete', old.rowid, old.text);
    INSERT INTO segments_fts (rowid, text)
    VALUES (new.rowid, new.text);
END;
