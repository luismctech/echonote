//! SQLite implementation of [`MeetingStore`].

use std::path::Path;
use std::str::FromStr;

use async_trait::async_trait;
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteSynchronous};
use sqlx::{Row, SqlitePool};
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;
use tracing::{debug, info};
use uuid::Uuid;

use echo_domain::{
    AudioFormat, CreateMeeting, DomainError, FinalizeMeeting, Meeting, MeetingId, MeetingSearchHit,
    MeetingStore, MeetingSummary, Note, NoteId, Segment, SegmentId, Speaker, SpeakerId, Summary,
    SummaryContent, SummaryId,
};

/// Embedded migrations (`crates/echo-storage/migrations/`).
static MIGRATIONS: sqlx::migrate::Migrator = sqlx::migrate!("./migrations");

/// Wrap a free-form storage failure (parse, I/O, …) as a `DomainError`.
///
/// Use [`DomainError::NotFound`] directly for missing-row cases instead
/// of routing them through here — callers (use cases, IPC commands)
/// distinguish "missing" from "storage failed" and the noisy
/// `Storage("...")` envelope obscures that.
fn db_err(msg: impl Into<String>) -> DomainError {
    DomainError::Storage(msg.into())
}

fn map_sqlx<E: std::fmt::Display>(prefix: &'static str) -> impl Fn(E) -> DomainError {
    move |e| DomainError::Storage(format!("{prefix}: {e}"))
}

/// SQLite-backed [`MeetingStore`]. Cheap to clone (`SqlitePool` wraps an
/// `Arc` internally).
#[derive(Debug, Clone)]
pub struct SqliteMeetingStore {
    pool: SqlitePool,
}

impl SqliteMeetingStore {
    /// Open (creating if missing) the SQLite database at `path` and run
    /// pending migrations.
    pub async fn open(path: impl AsRef<Path>) -> Result<Self, DomainError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| db_err(format!("create db parent dir: {e}")))?;
            }
        }
        let url = format!("sqlite://{}", path.display());
        Self::connect(&url).await
    }

    /// Open an in-memory database. Useful for tests; data does not
    /// survive process exit.
    pub async fn open_in_memory() -> Result<Self, DomainError> {
        Self::connect("sqlite::memory:").await
    }

    async fn connect(url: &str) -> Result<Self, DomainError> {
        let opts = SqliteConnectOptions::from_str(url)
            .map_err(map_sqlx("parse sqlite url"))?
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .synchronous(SqliteSynchronous::Normal)
            .foreign_keys(true);

        let pool = SqlitePoolOptions::new()
            .max_connections(8)
            .acquire_timeout(std::time::Duration::from_secs(5))
            .connect_with(opts)
            .await
            .map_err(map_sqlx("open sqlite pool"))?;

        MIGRATIONS
            .run(&pool)
            .await
            .map_err(map_sqlx("apply migrations"))?;

        info!(target: "echo_storage", url, "sqlite store ready");
        Ok(Self { pool })
    }

    /// Drop the underlying pool. Mostly useful in tests so the temp dir
    /// can be unlinked on Windows.
    pub async fn close(&self) {
        self.pool.close().await;
    }

    /// Force a WAL checkpoint so all data is flushed to the main database
    /// file. Must be called before any hard-exit (`_exit`, `process::exit`)
    /// to prevent data loss.
    pub async fn checkpoint(&self) -> Result<(), DomainError> {
        sqlx::query("PRAGMA wal_checkpoint(TRUNCATE)")
            .execute(&self.pool)
            .await
            .map_err(|e| db_err(format!("WAL checkpoint failed: {e}")))?;
        Ok(())
    }
}

fn parse_uuid(s: &str, what: &'static str) -> Result<Uuid, DomainError> {
    Uuid::parse_str(s).map_err(|e| db_err(format!("invalid {what} uuid {s:?}: {e}")))
}

fn parse_rfc3339(s: &str, what: &'static str) -> Result<OffsetDateTime, DomainError> {
    OffsetDateTime::parse(s, &Rfc3339)
        .map_err(|e| db_err(format!("invalid {what} timestamp {s:?}: {e}")))
}

#[async_trait]
impl MeetingStore for SqliteMeetingStore {
    async fn create(&self, params: CreateMeeting) -> Result<MeetingSummary, DomainError> {
        let started_at = OffsetDateTime::now_utc();
        let started_str = started_at
            .format(&Rfc3339)
            .map_err(|e| db_err(format!("format started_at: {e}")))?;
        let id_str = params.id.to_string();

        sqlx::query(
            r#"INSERT INTO meetings
                 (id, title, started_at, ended_at, duration_ms,
                  language, segment_count, sample_rate_hz, channels)
               VALUES (?, ?, ?, NULL, 0, NULL, 0, ?, ?)"#,
        )
        .bind(&id_str)
        .bind(&params.title)
        .bind(&started_str)
        .bind(i64::from(params.input_format.sample_rate_hz))
        .bind(i64::from(params.input_format.channels))
        .execute(&self.pool)
        .await
        .map_err(map_sqlx("insert meeting"))?;

        debug!(meeting.id = %params.id, "meeting row created");

        Ok(MeetingSummary {
            id: params.id,
            title: params.title,
            started_at,
            ended_at: None,
            duration_ms: 0,
            language: None,
            segment_count: 0,
        })
    }

    async fn upsert_speaker(
        &self,
        meeting_id: MeetingId,
        speaker: &Speaker,
    ) -> Result<(), DomainError> {
        // Upsert key is (meeting_id, slot) — the diarizer's stable
        // arrival-order index. The first INSERT wins the `id`;
        // subsequent calls only refresh the `label` *if* the caller
        // passed a non-null one. The streaming recorder always
        // upserts with `label = None`, so it cannot clobber a name
        // the user picked via the rename use case (Sprint 1 day 8).
        let meeting_str = meeting_id.to_string();
        let id_str = speaker.id.0.to_string();
        sqlx::query(
            r#"INSERT INTO speakers (id, meeting_id, slot, label)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(meeting_id, slot) DO UPDATE
                       SET label = COALESCE(excluded.label, speakers.label)"#,
        )
        .bind(&id_str)
        .bind(&meeting_str)
        .bind(i64::from(speaker.slot))
        .bind(speaker.label.as_deref())
        .execute(&self.pool)
        .await
        .map_err(map_sqlx("upsert speaker"))?;
        Ok(())
    }

    async fn rename_speaker(
        &self,
        meeting_id: MeetingId,
        speaker_id: SpeakerId,
        label: Option<&str>,
    ) -> Result<bool, DomainError> {
        // Targeted UPDATE WHERE id=?: lets the caller clear the
        // label back to NULL (impossible via the COALESCE upsert) and
        // identifies the row by its stable SpeakerId, so the UI can
        // rename without knowing the arrival-order slot.
        let result = sqlx::query("UPDATE speakers SET label = ? WHERE id = ? AND meeting_id = ?")
            .bind(label)
            .bind(speaker_id.0.to_string())
            .bind(meeting_id.to_string())
            .execute(&self.pool)
            .await
            .map_err(map_sqlx("rename speaker"))?;
        Ok(result.rows_affected() > 0)
    }

    async fn rename_meeting(
        &self,
        meeting_id: MeetingId,
        title: &str,
    ) -> Result<bool, DomainError> {
        let result = sqlx::query("UPDATE meetings SET title = ? WHERE id = ?")
            .bind(title)
            .bind(meeting_id.to_string())
            .execute(&self.pool)
            .await
            .map_err(map_sqlx("rename meeting"))?;
        Ok(result.rows_affected() > 0)
    }

    async fn list_speakers(&self, meeting_id: MeetingId) -> Result<Vec<Speaker>, DomainError> {
        let id_str = meeting_id.to_string();
        let rows = sqlx::query(
            r#"SELECT id, slot, label
                   FROM speakers
                  WHERE meeting_id = ?
               ORDER BY slot ASC"#,
        )
        .bind(&id_str)
        .fetch_all(&self.pool)
        .await
        .map_err(map_sqlx("list speakers"))?;
        rows.iter().map(row_to_speaker).collect()
    }

    async fn append_segments(
        &self,
        meeting_id: MeetingId,
        segments: &[Segment],
    ) -> Result<(), DomainError> {
        if segments.is_empty() {
            return Ok(());
        }
        let mut tx = self.pool.begin().await.map_err(map_sqlx("begin tx"))?;

        let meeting_str = meeting_id.to_string();
        for s in segments {
            sqlx::query(
                r#"INSERT INTO segments
                       (id, meeting_id, start_ms, end_ms, text, speaker_id, confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO NOTHING"#,
            )
            .bind(s.id.0.to_string())
            .bind(&meeting_str)
            .bind(i64::from(s.start_ms))
            .bind(i64::from(s.end_ms))
            .bind(&s.text)
            .bind(s.speaker_id.map(|sp| sp.0.to_string()))
            .bind(s.confidence.map(f64::from))
            .execute(&mut *tx)
            .await
            .map_err(map_sqlx("insert segment"))?;
        }

        // Recompute the segment count from the source of truth so we
        // stay accurate even if duplicate segment ids were de-duped above.
        sqlx::query(
            r#"UPDATE meetings
                  SET segment_count = (SELECT COUNT(*) FROM segments WHERE meeting_id = ?)
                WHERE id = ?"#,
        )
        .bind(&meeting_str)
        .bind(&meeting_str)
        .execute(&mut *tx)
        .await
        .map_err(map_sqlx("update segment_count"))?;

        tx.commit().await.map_err(map_sqlx("commit tx"))?;
        Ok(())
    }

    async fn finalize(
        &self,
        meeting_id: MeetingId,
        patch: FinalizeMeeting,
    ) -> Result<MeetingSummary, DomainError> {
        let id_str = meeting_id.to_string();

        // COALESCE with the new value so callers can patch a single
        // column without nuking the others.
        let result = sqlx::query(
            r#"UPDATE meetings
                  SET ended_at      = COALESCE(?, ended_at),
                      duration_ms   = COALESCE(?, duration_ms),
                      language      = COALESCE(?, language),
                      segment_count = COALESCE(?, segment_count)
                WHERE id = ?"#,
        )
        .bind(patch.ended_at_rfc3339.as_deref())
        .bind(patch.duration_ms.map(i64::from))
        .bind(patch.language.as_deref())
        .bind(patch.segment_count.map(i64::from))
        .bind(&id_str)
        .execute(&self.pool)
        .await
        .map_err(map_sqlx("finalize meeting"))?;

        if result.rows_affected() == 0 {
            return Err(db_err(format!("meeting {meeting_id} not found")));
        }

        let summary = self
            .fetch_summary(&id_str)
            .await?
            .ok_or_else(|| db_err(format!("meeting {meeting_id} disappeared after finalize")))?;
        Ok(summary)
    }

    async fn list(&self, limit: u32) -> Result<Vec<MeetingSummary>, DomainError> {
        let limit_i64 = if limit == 0 { -1_i64 } else { i64::from(limit) };
        let rows = sqlx::query(
            r#"SELECT id, title, started_at, ended_at, duration_ms,
                       language, segment_count
                  FROM meetings
                 WHERE ended_at IS NOT NULL
              ORDER BY started_at DESC
                 LIMIT ?"#,
        )
        .bind(limit_i64)
        .fetch_all(&self.pool)
        .await
        .map_err(map_sqlx("list meetings"))?;

        rows.iter().map(row_to_summary).collect()
    }

    async fn get(&self, meeting_id: MeetingId) -> Result<Option<Meeting>, DomainError> {
        let id_str = meeting_id.to_string();
        let Some(summary) = self.fetch_summary(&id_str).await? else {
            return Ok(None);
        };
        let format_row = sqlx::query("SELECT sample_rate_hz, channels FROM meetings WHERE id = ?")
            .bind(&id_str)
            .fetch_one(&self.pool)
            .await
            .map_err(map_sqlx("fetch meeting format"))?;
        let input_format = AudioFormat {
            sample_rate_hz: u32::try_from(format_row.get::<i64, _>(0))
                .map_err(|e| db_err(format!("sample_rate_hz overflow: {e}")))?,
            channels: u16::try_from(format_row.get::<i64, _>(1))
                .map_err(|e| db_err(format!("channels overflow: {e}")))?,
        };

        let seg_rows = sqlx::query(
            r#"SELECT id, start_ms, end_ms, text, speaker_id, confidence
                  FROM segments
                 WHERE meeting_id = ?
              ORDER BY start_ms ASC"#,
        )
        .bind(&id_str)
        .fetch_all(&self.pool)
        .await
        .map_err(map_sqlx("fetch segments"))?;

        let mut segments = Vec::with_capacity(seg_rows.len());
        for row in seg_rows {
            let id_text: String = row.get(0);
            let speaker_id_text: Option<String> = row.get(4);
            segments.push(Segment {
                id: SegmentId(parse_uuid(&id_text, "segment")?),
                start_ms: u32::try_from(row.get::<i64, _>(1))
                    .map_err(|e| db_err(format!("start_ms overflow: {e}")))?,
                end_ms: u32::try_from(row.get::<i64, _>(2))
                    .map_err(|e| db_err(format!("end_ms overflow: {e}")))?,
                text: row.get(3),
                speaker_id: speaker_id_text
                    .as_deref()
                    .map(|s| parse_uuid(s, "speaker").map(SpeakerId))
                    .transpose()?,
                confidence: row.get::<Option<f64>, _>(5).map(|v| v as f32),
            });
        }

        let speakers = self.list_speakers(meeting_id).await?;
        let notes = self.list_notes(meeting_id).await?;

        Ok(Some(Meeting {
            summary,
            input_format,
            segments,
            speakers,
            notes,
        }))
    }

    async fn delete(&self, meeting_id: MeetingId) -> Result<bool, DomainError> {
        let id_str = meeting_id.to_string();
        let result = sqlx::query("DELETE FROM meetings WHERE id = ?")
            .bind(&id_str)
            .execute(&self.pool)
            .await
            .map_err(map_sqlx("delete meeting"))?;
        Ok(result.rows_affected() > 0)
    }

    async fn search(&self, query: &str, limit: u32) -> Result<Vec<MeetingSearchHit>, DomainError> {
        // Empty input ⇒ empty result. The UI binds this to `onChange`
        // so the initial empty string would otherwise hit FTS with a
        // syntax error ("no terms to match").
        let Some(match_expr) = sanitize_fts_query(query) else {
            return Ok(Vec::new());
        };

        // FTS5 auxiliary functions (`bm25`, `snippet`) only resolve
        // when the FTS table appears directly in the FROM clause of
        // the *same* SELECT that holds the MATCH — they do NOT work
        // through CTEs or subqueries. So we run a single flat query
        // and dedupe per meeting in Rust.
        //
        // `snippet(segments_fts, 0, '<mark>', '</mark>', '…', 16)`
        // requests 16 tokens of context with `<mark>` markers — the
        // React side renders them via `dangerouslySetInnerHTML` (the
        // markers are HTML-safe and the index is built off our own
        // text so XSS surface is the same as showing the segment
        // body raw, which the rest of the UI already does).
        //
        // We over-fetch by a small factor before dedupe so a single
        // long meeting can't crowd out other hits below `limit`.
        // 4× is enough in practice; this stays bounded because we
        // also pass `LIMIT raw_cap` to SQLite.
        let raw_cap: i64 = if limit == 0 {
            -1
        } else {
            (i64::from(limit)).saturating_mul(4)
        };
        let rows = sqlx::query(
            r#"
            SELECT  m.id, m.title, m.started_at, m.ended_at, m.duration_ms,
                    m.language, m.segment_count,
                    snippet(segments_fts, 0, '<mark>', '</mark>', '…', 16) AS snippet,
                    bm25(segments_fts) AS rank,
                    s.meeting_id AS meeting_id
              FROM segments_fts
              JOIN segments s ON s.rowid = segments_fts.rowid
              JOIN meetings m ON m.id = s.meeting_id
             WHERE segments_fts MATCH ?
          ORDER BY rank ASC
             LIMIT ?
            "#,
        )
        .bind(&match_expr)
        .bind(raw_cap)
        .fetch_all(&self.pool)
        .await
        .map_err(map_sqlx("search meetings"))?;

        // Dedupe per meeting, keeping the strongest match (rows are
        // already ordered ASC by rank so the first occurrence wins).
        // `seen` is a hash set keyed on meeting id text — no need to
        // re-parse the UUID for the lookup.
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let cap = if limit == 0 {
            usize::MAX
        } else {
            limit as usize
        };
        let mut hits = Vec::new();
        for row in &rows {
            let mid_text: String = row.get(9);
            if !seen.insert(mid_text) {
                continue;
            }
            let snippet: String = row.get(7);
            let snippet = sanitize_fts_snippet(&snippet);
            let rank: f64 = row.get(8);
            hits.push(MeetingSearchHit {
                meeting: row_to_summary(row)?,
                snippet,
                rank,
            });
            if hits.len() >= cap {
                break;
            }
        }
        Ok(hits)
    }

    async fn checkpoint(&self) -> Result<(), DomainError> {
        SqliteMeetingStore::checkpoint(self).await
    }

    async fn close(&self) {
        // Forward to the inherent helper, which awaits any in-flight
        // queries and flushes the WAL before the pool drops. Called
        // from the shell's shutdown hook so SQLite has a chance to
        // checkpoint before `_exit` skips destructors.
        SqliteMeetingStore::close(self).await;
    }

    async fn upsert_summary(&self, summary: &Summary) -> Result<(), DomainError> {
        // Persist a freshly-generated `Summary` to SQLite. The unique
        // index on (meeting_id, template) means re-running the
        // `SummarizeMeeting` use case on the same meeting REPLACES the
        // previous row instead of accumulating drafts — that matches
        // the use case contract (one current summary per template).
        //
        // We store the full `SummaryContent` as JSON in `payload`
        // (with the `template` tag included) and denormalise the
        // discriminator into its own column so we can index it. The
        // round-trip in `get_summary` ignores the column and parses
        // the payload back, which keeps the JSON the source of truth
        // and the column purely a routing aid.
        // Serialise the variant first, then read its `template`
        // discriminator out of the resulting JSON. That way the
        // column we index on is guaranteed to match what serde wrote
        // — and the storage adapter does not have to be updated
        // every time a new `SummaryContent` variant ships (the enum
        // is `#[non_exhaustive]` for exactly this reason).
        let payload_value = serde_json::to_value(&summary.content)
            .map_err(|e| db_err(format!("encode summary payload: {e}")))?;
        let template = payload_value
            .get("template")
            .and_then(|v| v.as_str())
            .ok_or_else(|| db_err("summary payload missing `template` discriminator"))?
            .to_owned();
        let payload = serde_json::to_string(&payload_value)
            .map_err(|e| db_err(format!("re-encode summary payload: {e}")))?;
        let created_at = summary
            .created_at
            .format(&Rfc3339)
            .map_err(|e| db_err(format!("format summary created_at: {e}")))?;

        sqlx::query(
            r#"INSERT INTO summaries
                   (id, meeting_id, template, model, language, created_at, payload)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(meeting_id, template) DO UPDATE SET
                   id         = excluded.id,
                   model      = excluded.model,
                   language   = excluded.language,
                   created_at = excluded.created_at,
                   payload    = excluded.payload"#,
        )
        .bind(summary.id.0.to_string())
        .bind(summary.meeting_id.to_string())
        .bind(&template)
        .bind(&summary.model)
        .bind(summary.language.as_deref())
        .bind(&created_at)
        .bind(&payload)
        .execute(&self.pool)
        .await
        .map_err(map_sqlx("upsert summary"))?;

        debug!(
            meeting.id = %summary.meeting_id,
            summary.id = %summary.id.0,
            template = %template,
            "summary upserted"
        );
        Ok(())
    }

    async fn get_summary(&self, meeting_id: MeetingId) -> Result<Option<Summary>, DomainError> {
        // `LIMIT 1` ordered by created_at DESC: the UI only ever asks
        // for the *current* summary of a meeting (the use case
        // upserts on (meeting_id, template), so today there is at
        // most one row per template anyway, but ordering keeps us
        // honest the day a second template ships).
        let row = sqlx::query(
            r#"SELECT id, model, language, created_at, payload
                  FROM summaries
                 WHERE meeting_id = ?
              ORDER BY created_at DESC
                 LIMIT 1"#,
        )
        .bind(meeting_id.to_string())
        .fetch_optional(&self.pool)
        .await
        .map_err(map_sqlx("fetch summary"))?;

        let Some(row) = row else { return Ok(None) };

        let id_text: String = row.get(0);
        let model: String = row.get(1);
        let language: Option<String> = row.get(2);
        let created_text: String = row.get(3);
        let payload: String = row.get(4);

        let content: SummaryContent = serde_json::from_str(&payload)
            .map_err(|e| db_err(format!("decode summary payload: {e}")))?;

        Ok(Some(Summary {
            id: SummaryId(parse_uuid(&id_text, "summary")?),
            meeting_id,
            model,
            language,
            created_at: parse_rfc3339(&created_text, "summary created_at")?,
            content,
        }))
    }

    async fn add_note(
        &self,
        meeting_id: MeetingId,
        text: &str,
        timestamp_ms: u32,
    ) -> Result<Note, DomainError> {
        let id = NoteId::new();
        let created_at = OffsetDateTime::now_utc()
            .format(&Rfc3339)
            .map_err(|e| db_err(format!("format note created_at: {e}")))?;

        sqlx::query(
            r#"INSERT INTO notes (id, meeting_id, text, timestamp_ms, created_at)
               VALUES (?, ?, ?, ?, ?)"#,
        )
        .bind(id.0.to_string())
        .bind(meeting_id.to_string())
        .bind(text)
        .bind(i64::from(timestamp_ms))
        .bind(&created_at)
        .execute(&self.pool)
        .await
        .map_err(map_sqlx("add note"))?;

        debug!(meeting.id = %meeting_id, note.id = %id, "note added");

        Ok(Note {
            id,
            meeting_id,
            text: text.to_owned(),
            timestamp_ms,
            created_at,
        })
    }

    async fn list_notes(&self, meeting_id: MeetingId) -> Result<Vec<Note>, DomainError> {
        let rows = sqlx::query(
            r#"SELECT id, text, timestamp_ms, created_at
                 FROM notes
                WHERE meeting_id = ?
             ORDER BY timestamp_ms ASC"#,
        )
        .bind(meeting_id.to_string())
        .fetch_all(&self.pool)
        .await
        .map_err(map_sqlx("list notes"))?;

        let mut notes = Vec::with_capacity(rows.len());
        for row in rows {
            let id_text: String = row.get(0);
            notes.push(Note {
                id: NoteId(parse_uuid(&id_text, "note")?),
                meeting_id,
                text: row.get(1),
                timestamp_ms: u32::try_from(row.get::<i64, _>(2))
                    .map_err(|e| db_err(format!("note timestamp_ms overflow: {e}")))?,
                created_at: row.get(3),
            });
        }
        Ok(notes)
    }

    async fn delete_note(&self, note_id: NoteId) -> Result<bool, DomainError> {
        let result = sqlx::query("DELETE FROM notes WHERE id = ?")
            .bind(note_id.0.to_string())
            .execute(&self.pool)
            .await
            .map_err(map_sqlx("delete note"))?;
        Ok(result.rows_affected() > 0)
    }
}

impl SqliteMeetingStore {
    async fn fetch_summary(&self, id_str: &str) -> Result<Option<MeetingSummary>, DomainError> {
        let row = sqlx::query(
            r#"SELECT id, title, started_at, ended_at, duration_ms,
                       language, segment_count
                  FROM meetings
                 WHERE id = ?"#,
        )
        .bind(id_str)
        .fetch_optional(&self.pool)
        .await
        .map_err(map_sqlx("fetch meeting"))?;
        row.as_ref().map(row_to_summary).transpose()
    }
}

fn row_to_speaker(row: &sqlx::sqlite::SqliteRow) -> Result<Speaker, DomainError> {
    let id_text: String = row.get(0);
    let slot = u32::try_from(row.get::<i64, _>(1))
        .map_err(|e| db_err(format!("speaker slot overflow: {e}")))?;
    let label: Option<String> = row.get(2);
    Ok(Speaker {
        id: SpeakerId(parse_uuid(&id_text, "speaker")?),
        slot,
        label,
    })
}

fn row_to_summary(row: &sqlx::sqlite::SqliteRow) -> Result<MeetingSummary, DomainError> {
    let id_text: String = row.get(0);
    let started_text: String = row.get(2);
    let ended_text: Option<String> = row.get(3);
    Ok(MeetingSummary {
        id: MeetingId(parse_uuid(&id_text, "meeting")?),
        title: row.get(1),
        started_at: parse_rfc3339(&started_text, "started_at")?,
        ended_at: ended_text
            .as_deref()
            .map(|s| parse_rfc3339(s, "ended_at"))
            .transpose()?,
        duration_ms: u32::try_from(row.get::<i64, _>(4))
            .map_err(|e| db_err(format!("duration_ms overflow: {e}")))?,
        language: row.get(5),
        segment_count: u32::try_from(row.get::<i64, _>(6))
            .map_err(|e| db_err(format!("segment_count overflow: {e}")))?,
    })
}

/// Convert raw user input into a safe FTS5 `MATCH` expression.
///
/// FTS5 has its own micro-grammar (`"phrase"`, `term1 AND term2`,
/// `column:foo`, `^bar`, `*` prefix, etc.) and a stray `"` from the
/// user can either error out or — depending on what comes next —
/// silently change the meaning of the query. Defense: we strip
/// FTS5 syntax characters from each whitespace-delimited token and
/// then wrap the survivor in double quotes, turning the input into
/// the safe shape `"tok1" "tok2" "tok3"` (an implicit AND).
///
/// HTML-escape a snippet from FTS5, preserving only the `<mark>` /
/// `</mark>` tags that **FTS5 itself** injected as highlight markers.
///
/// Problem with the naïve escape-then-restore approach: if the
/// user-stored text literally contains the string `<mark>`, the old
/// code would restore it as a real HTML tag — a stored-XSS vector.
///
/// Fix: replace the FTS5-injected markers with unique placeholders
/// **before** HTML-escaping, then swap the placeholders with the real
/// `<mark>` tags afterwards. The placeholders cannot appear in the
/// original text because they contain NUL bytes.
fn sanitize_fts_snippet(raw: &str) -> String {
    const OPEN_PH: &str = "\x00MARK_OPEN\x00";
    const CLOSE_PH: &str = "\x00MARK_CLOSE\x00";

    // Step 1: replace the real FTS5 markers with placeholders.
    let with_placeholders = raw.replace("<mark>", OPEN_PH).replace("</mark>", CLOSE_PH);

    // Step 2: HTML-escape everything (including any literal "<mark>"
    // that was NOT an FTS5 marker — those were already consumed in
    // step 1, so only user-authored `<mark>` survives to be escaped).
    let escaped = with_placeholders
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;");

    // Step 3: restore placeholders → real HTML.
    escaped
        .replace(OPEN_PH, "<mark>")
        .replace(CLOSE_PH, "</mark>")
}

/// Returns `None` for empty / whitespace-only / all-stripped inputs
/// so the caller can short-circuit to "no hits" without ever asking
/// SQLite. That matches the search port contract.
fn sanitize_fts_query(raw: &str) -> Option<String> {
    // FTS5's tokenizer treats these as syntax / operator chars.
    // Stripping is safer than trying to escape them — the user is
    // looking for natural words, not composing a search expression.
    const STRIP: &[char] = &['"', '*', '(', ')', '^', ':', '+', '-', '~'];

    let mut out = String::new();
    for token in raw.split_whitespace() {
        let cleaned: String = token.chars().filter(|c| !STRIP.contains(c)).collect();
        if cleaned.is_empty() {
            continue;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        out.push('"');
        out.push_str(&cleaned);
        out.push('"');
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn fmt() -> AudioFormat {
        AudioFormat::WHISPER
    }

    fn segment(start_ms: u32, end_ms: u32, text: &str) -> Segment {
        Segment {
            id: SegmentId::new(),
            start_ms,
            end_ms,
            text: text.into(),
            speaker_id: None,
            confidence: Some(0.9),
        }
    }

    #[tokio::test]
    async fn round_trip_create_append_get_list_delete() {
        let store = SqliteMeetingStore::open_in_memory().await.unwrap();
        let id = MeetingId::new();
        let summary = store
            .create(CreateMeeting {
                id,
                title: "Standup".into(),
                input_format: fmt(),
            })
            .await
            .unwrap();
        assert_eq!(summary.id, id);
        assert_eq!(summary.segment_count, 0);

        store
            .append_segments(
                id,
                &[segment(0, 1_000, "hello"), segment(1_000, 2_000, "world")],
            )
            .await
            .unwrap();

        let listed = store.list(0).await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].segment_count, 2);

        let full = store.get(id).await.unwrap().unwrap();
        assert_eq!(full.segments.len(), 2);
        assert_eq!(full.segments[0].text, "hello");
        assert_eq!(full.input_format, fmt());
        assert!(full.speakers.is_empty());

        let deleted = store.delete(id).await.unwrap();
        assert!(deleted);
        assert!(store.get(id).await.unwrap().is_none());
        assert!(!store.delete(id).await.unwrap());
    }

    #[tokio::test]
    async fn append_is_idempotent_on_segment_id() {
        let store = SqliteMeetingStore::open_in_memory().await.unwrap();
        let id = MeetingId::new();
        store
            .create(CreateMeeting {
                id,
                title: "T".into(),
                input_format: fmt(),
            })
            .await
            .unwrap();
        let s = segment(0, 500, "once");
        store.append_segments(id, &[s.clone()]).await.unwrap();
        store.append_segments(id, &[s.clone()]).await.unwrap();
        let listed = store.list(0).await.unwrap();
        assert_eq!(listed[0].segment_count, 1);
    }

    #[tokio::test]
    async fn finalize_patches_only_provided_fields() {
        let store = SqliteMeetingStore::open_in_memory().await.unwrap();
        let id = MeetingId::new();
        store
            .create(CreateMeeting {
                id,
                title: "T".into(),
                input_format: fmt(),
            })
            .await
            .unwrap();

        let updated = store
            .finalize(
                id,
                FinalizeMeeting {
                    ended_at_rfc3339: Some(OffsetDateTime::now_utc().format(&Rfc3339).unwrap()),
                    duration_ms: Some(12_000),
                    language: Some("en".into()),
                    segment_count: None,
                },
            )
            .await
            .unwrap();
        assert!(updated.ended_at.is_some());
        assert_eq!(updated.duration_ms, 12_000);
        assert_eq!(updated.language.as_deref(), Some("en"));
        assert_eq!(updated.segment_count, 0); // untouched
    }

    #[tokio::test]
    async fn list_is_ordered_by_started_at_desc() {
        let store = SqliteMeetingStore::open_in_memory().await.unwrap();
        let a = MeetingId::new();
        store
            .create(CreateMeeting {
                id: a,
                title: "first".into(),
                input_format: fmt(),
            })
            .await
            .unwrap();
        // Sleep enough that started_at differs at second granularity.
        tokio::time::sleep(std::time::Duration::from_millis(1_100)).await;
        let b = MeetingId::new();
        store
            .create(CreateMeeting {
                id: b,
                title: "second".into(),
                input_format: fmt(),
            })
            .await
            .unwrap();
        let listed = store.list(0).await.unwrap();
        assert_eq!(listed.len(), 2);
        assert_eq!(listed[0].id, b, "newest first");
        assert_eq!(listed[1].id, a);
    }

    #[tokio::test]
    async fn upsert_speaker_is_idempotent_on_meeting_slot() {
        let store = SqliteMeetingStore::open_in_memory().await.unwrap();
        let id = MeetingId::new();
        store
            .create(CreateMeeting {
                id,
                title: "diarized".into(),
                input_format: fmt(),
            })
            .await
            .unwrap();

        // First insert: anonymous speaker at slot 0.
        let s0 = Speaker::anonymous(0);
        store.upsert_speaker(id, &s0).await.unwrap();

        // Recorder-style re-upsert with label=None must keep the
        // original id and label (preserving the row exactly).
        let s0_again = Speaker {
            id: SpeakerId::new(),
            slot: 0,
            label: None,
        };
        store.upsert_speaker(id, &s0_again).await.unwrap();

        let listed = store.list_speakers(id).await.unwrap();
        assert_eq!(listed.len(), 1, "second upsert must not spawn a new row");
        assert_eq!(listed[0].id, s0.id, "original SpeakerId must survive");
        assert!(listed[0].label.is_none(), "label still anonymous");

        // Rename-style upsert with a label sets it.
        let renamed = Speaker {
            id: SpeakerId::new(),
            slot: 0,
            label: Some("Alice".into()),
        };
        store.upsert_speaker(id, &renamed).await.unwrap();
        let listed = store.list_speakers(id).await.unwrap();
        assert_eq!(listed[0].id, s0.id, "rename must NOT replace the SpeakerId");
        assert_eq!(listed[0].label.as_deref(), Some("Alice"));

        // Subsequent recorder upsert (label None) must NOT clobber
        // the user-provided "Alice" — that is the whole point of the
        // COALESCE in the upsert SQL.
        store.upsert_speaker(id, &s0_again).await.unwrap();
        let listed = store.list_speakers(id).await.unwrap();
        assert_eq!(
            listed[0].label.as_deref(),
            Some("Alice"),
            "recorder re-upsert must not erase the user's label"
        );

        // Insert a second speaker at slot 1; both are returned in slot order.
        let s1 = Speaker::anonymous(1);
        store.upsert_speaker(id, &s1).await.unwrap();
        let listed = store.list_speakers(id).await.unwrap();
        assert_eq!(listed.len(), 2);
        assert_eq!(listed[0].slot, 0);
        assert_eq!(listed[1].slot, 1);
    }

    #[tokio::test]
    async fn get_returns_segments_with_their_speaker_ids_and_speaker_rows() {
        let store = SqliteMeetingStore::open_in_memory().await.unwrap();
        let id = MeetingId::new();
        store
            .create(CreateMeeting {
                id,
                title: "diarized".into(),
                input_format: fmt(),
            })
            .await
            .unwrap();

        // Speaker first, segment referencing it second — matches the
        // ordering MeetingRecorder uses on every Chunk event.
        let alice = Speaker::anonymous(0);
        store.upsert_speaker(id, &alice).await.unwrap();
        let mut seg = segment(0, 1_000, "hello");
        seg.speaker_id = Some(alice.id);
        store.append_segments(id, &[seg.clone()]).await.unwrap();

        let full = store.get(id).await.unwrap().unwrap();
        assert_eq!(full.segments.len(), 1);
        assert_eq!(full.segments[0].speaker_id, Some(alice.id));
        assert_eq!(full.speakers.len(), 1);
        assert_eq!(full.speakers[0].id, alice.id);
        assert_eq!(full.speakers[0].slot, 0);
        assert!(full.speakers[0].label.is_none());
    }

    #[tokio::test]
    async fn rename_speaker_updates_label_and_can_clear_back_to_null() {
        let store = SqliteMeetingStore::open_in_memory().await.unwrap();
        let id = MeetingId::new();
        store
            .create(CreateMeeting {
                id,
                title: "T".into(),
                input_format: fmt(),
            })
            .await
            .unwrap();
        let s = Speaker::anonymous(0);
        store.upsert_speaker(id, &s).await.unwrap();

        // Set label.
        let renamed = store.rename_speaker(id, s.id, Some("Alice")).await.unwrap();
        assert!(renamed);
        assert_eq!(
            store.list_speakers(id).await.unwrap()[0].label.as_deref(),
            Some("Alice")
        );

        // Clear it — impossible via the COALESCE upsert, but the
        // rename method must be able to express this.
        let cleared = store.rename_speaker(id, s.id, None).await.unwrap();
        assert!(cleared);
        assert!(store.list_speakers(id).await.unwrap()[0].label.is_none());

        // Unknown speaker id → false (so the use case can 404 the UI).
        let missing = store
            .rename_speaker(id, SpeakerId::new(), Some("ghost"))
            .await
            .unwrap();
        assert!(!missing);

        // Wrong meeting id → also false (scoping check).
        let other_meeting = MeetingId::new();
        let wrong_meeting = store
            .rename_speaker(other_meeting, s.id, Some("nope"))
            .await
            .unwrap();
        assert!(!wrong_meeting);
    }

    #[tokio::test]
    async fn deleting_meeting_cascades_to_speakers() {
        let store = SqliteMeetingStore::open_in_memory().await.unwrap();
        let id = MeetingId::new();
        store
            .create(CreateMeeting {
                id,
                title: "T".into(),
                input_format: fmt(),
            })
            .await
            .unwrap();
        store
            .upsert_speaker(id, &Speaker::anonymous(0))
            .await
            .unwrap();
        assert_eq!(store.list_speakers(id).await.unwrap().len(), 1);

        store.delete(id).await.unwrap();
        assert!(store.list_speakers(id).await.unwrap().is_empty());
    }

    // ---------- FTS5 search (Sprint 1 day 8) -------------------------------

    /// Convenience: create a meeting and append a single segment with
    /// the given text. Returns the id so tests can search for it.
    async fn seed_meeting_with_text(
        store: &SqliteMeetingStore,
        title: &str,
        text: &str,
    ) -> MeetingId {
        let id = MeetingId::new();
        store
            .create(CreateMeeting {
                id,
                title: title.into(),
                input_format: fmt(),
            })
            .await
            .unwrap();
        store
            .append_segments(id, &[segment(0, 1_000, text)])
            .await
            .unwrap();
        id
    }

    #[tokio::test]
    async fn search_finds_matching_segment_and_returns_summary_with_snippet() {
        let store = SqliteMeetingStore::open_in_memory().await.unwrap();
        let id = seed_meeting_with_text(&store, "Roadmap", "we ship the alpha next quarter").await;

        let hits = store.search("alpha", 10).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].meeting.id, id);
        assert_eq!(hits[0].meeting.title, "Roadmap");
        // snippet() wraps the matched term with our chosen markers.
        assert!(
            hits[0].snippet.contains("<mark>alpha</mark>"),
            "got: {}",
            hits[0].snippet
        );
    }

    #[tokio::test]
    async fn search_orders_by_bm25_rank_ascending() {
        // Two meetings, one with the term repeated → lower (better)
        // bm25 rank, so it must come first.
        let store = SqliteMeetingStore::open_in_memory().await.unwrap();
        let _weak = seed_meeting_with_text(&store, "weak", "we mention budget once").await;
        let strong = seed_meeting_with_text(
            &store,
            "strong",
            "budget budget budget — budget review meeting",
        )
        .await;

        let hits = store.search("budget", 10).await.unwrap();
        assert_eq!(hits.len(), 2);
        // BM25 negative-scaled: smaller is better.
        assert!(hits[0].rank <= hits[1].rank, "ranks: {hits:?}");
        assert_eq!(hits[0].meeting.id, strong);
    }

    #[tokio::test]
    async fn search_collapses_per_meeting_to_one_hit() {
        // Three matching segments in the same meeting → still one hit.
        let store = SqliteMeetingStore::open_in_memory().await.unwrap();
        let id = MeetingId::new();
        store
            .create(CreateMeeting {
                id,
                title: "Long".into(),
                input_format: fmt(),
            })
            .await
            .unwrap();
        store
            .append_segments(
                id,
                &[
                    segment(0, 1_000, "alpha release notes"),
                    segment(1_000, 2_000, "alpha milestones"),
                    segment(2_000, 3_000, "alpha retro"),
                ],
            )
            .await
            .unwrap();

        let hits = store.search("alpha", 10).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].meeting.id, id);
    }

    #[tokio::test]
    async fn search_is_diacritic_insensitive_for_spanish() {
        // Tokenizer is `unicode61 remove_diacritics 2` so "Garcia"
        // must hit a segment containing "García". This is the whole
        // reason we picked that tokenizer in the migration.
        let store = SqliteMeetingStore::open_in_memory().await.unwrap();
        let id = seed_meeting_with_text(&store, "ES", "la propuesta de García fue aprobada").await;

        let hits = store.search("garcia", 10).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].meeting.id, id);
    }

    #[tokio::test]
    async fn search_handles_special_chars_without_erroring() {
        // A user typing `"` or `*` in the search box must not crash
        // the FTS parser. Sanitiser strips them before MATCH.
        let store = SqliteMeetingStore::open_in_memory().await.unwrap();
        let id =
            seed_meeting_with_text(&store, "Quoted", "we discuss the design doc tomorrow").await;

        // Quote in the middle of the term: should still match "design".
        let hits = store.search(r#"design""#, 10).await.unwrap();
        assert_eq!(hits.len(), 1, "got: {:?}", hits);
        assert_eq!(hits[0].meeting.id, id);

        // All-syntax input collapses to None → empty result, no error.
        assert!(store.search(r#"*()"^"#, 10).await.unwrap().is_empty());

        // Empty / whitespace-only inputs return empty without hitting FTS.
        assert!(store.search("", 10).await.unwrap().is_empty());
        assert!(store.search("   \t\n  ", 10).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn search_respects_limit_and_zero_means_no_cap() {
        let store = SqliteMeetingStore::open_in_memory().await.unwrap();
        for i in 0..5 {
            seed_meeting_with_text(&store, &format!("m{i}"), "shared term").await;
        }
        let capped = store.search("shared", 2).await.unwrap();
        assert_eq!(capped.len(), 2);

        let uncapped = store.search("shared", 0).await.unwrap();
        assert_eq!(uncapped.len(), 5);
    }

    #[tokio::test]
    async fn deleting_meeting_removes_its_text_from_search_index() {
        // Trigger `segments_ad` must keep the inverted index in sync
        // when ON DELETE CASCADE wipes the segments out from under us.
        let store = SqliteMeetingStore::open_in_memory().await.unwrap();
        let id = seed_meeting_with_text(&store, "ToDelete", "ephemeral phrase here").await;

        assert_eq!(store.search("ephemeral", 10).await.unwrap().len(), 1);
        store.delete(id).await.unwrap();
        assert!(
            store.search("ephemeral", 10).await.unwrap().is_empty(),
            "FTS index leaked after meeting delete"
        );
    }

    #[test]
    fn sanitize_fts_query_wraps_tokens_in_quotes_and_strips_syntax() {
        assert_eq!(sanitize_fts_query("hello"), Some(r#""hello""#.into()));
        assert_eq!(
            sanitize_fts_query("hello world"),
            Some(r#""hello" "world""#.into())
        );
        // Quote inside the token is stripped, not escaped — simpler
        // and equivalent for natural-language search.
        assert_eq!(sanitize_fts_query(r#"he"llo"#), Some(r#""hello""#.into()));
        // All operators stripped → nothing to search → None.
        assert_eq!(sanitize_fts_query(r#"*()^"#), None);
        assert_eq!(sanitize_fts_query(""), None);
        assert_eq!(sanitize_fts_query("   "), None);
    }

    // ---------- LLM summaries (Sprint 1 day 9) -----------------------------

    fn general_summary(meeting: MeetingId, model: &str, language: Option<&str>) -> Summary {
        Summary {
            id: SummaryId::new(),
            meeting_id: meeting,
            model: model.into(),
            language: language.map(str::to_owned),
            created_at: OffsetDateTime::now_utc(),
            content: SummaryContent::General {
                summary: "Discussed Q4 launch.".into(),
                key_points: vec!["Ship beta".into(), "Hire eng".into()],
                decisions: vec!["Move launch to Nov".into()],
                action_items: vec![echo_domain::ActionItem {
                    task: "Draft press release".into(),
                    owner: Some("Ana".into()),
                    due: Some("2025-10-30".into()),
                }],
                open_questions: vec!["Who owns translation?".into()],
            },
        }
    }

    async fn make_meeting(store: &SqliteMeetingStore) -> MeetingId {
        let id = MeetingId::new();
        store
            .create(CreateMeeting {
                id,
                title: "Standup".into(),
                input_format: fmt(),
            })
            .await
            .unwrap();
        id
    }

    #[tokio::test]
    async fn upsert_summary_round_trips_through_get_summary() {
        let store = SqliteMeetingStore::open_in_memory().await.unwrap();
        let mid = make_meeting(&store).await;

        // Empty meeting → no summary yet.
        assert!(store.get_summary(mid).await.unwrap().is_none());

        let s = general_summary(mid, "qwen2.5-7b-instruct-q4_k_m", Some("es"));
        store.upsert_summary(&s).await.unwrap();

        let fetched = store.get_summary(mid).await.unwrap().expect("row exists");
        assert_eq!(fetched.id, s.id);
        assert_eq!(fetched.meeting_id, s.meeting_id);
        assert_eq!(fetched.model, s.model);
        assert_eq!(fetched.language.as_deref(), Some("es"));
        // Same variant + payload survived the JSON round-trip.
        match (&fetched.content, &s.content) {
            (
                SummaryContent::General {
                    summary: a,
                    key_points: ak,
                    decisions: ad,
                    action_items: aa,
                    open_questions: aq,
                },
                SummaryContent::General {
                    summary: b,
                    key_points: bk,
                    decisions: bd,
                    action_items: ba,
                    open_questions: bq,
                },
            ) => {
                assert_eq!(a, b);
                assert_eq!(ak, bk);
                assert_eq!(ad, bd);
                assert_eq!(aa, ba);
                assert_eq!(aq, bq);
            }
            _ => panic!("variant mismatch"),
        }
    }

    #[tokio::test]
    async fn upsert_summary_replaces_previous_row_for_same_template() {
        let store = SqliteMeetingStore::open_in_memory().await.unwrap();
        let mid = make_meeting(&store).await;

        let s1 = general_summary(mid, "qwen2.5-3b", None);
        store.upsert_summary(&s1).await.unwrap();

        // Re-generate with a different model + content. The unique
        // index on (meeting_id, template) MUST replace, not duplicate.
        let mut s2 = general_summary(mid, "qwen2.5-7b", Some("en"));
        if let SummaryContent::General {
            ref mut summary, ..
        } = s2.content
        {
            *summary = "Replaced summary text".into();
        }
        store.upsert_summary(&s2).await.unwrap();

        let row_count: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM summaries WHERE meeting_id = ?")
                .bind(mid.to_string())
                .fetch_one(&store.pool)
                .await
                .unwrap();
        assert_eq!(row_count, 1, "second upsert must replace, not append");

        let fetched = store.get_summary(mid).await.unwrap().unwrap();
        assert_eq!(fetched.id, s2.id, "row id is the new one");
        assert_eq!(fetched.model, "qwen2.5-7b");
        assert_eq!(fetched.language.as_deref(), Some("en"));
        if let SummaryContent::General { summary, .. } = fetched.content {
            assert_eq!(summary, "Replaced summary text");
        } else {
            panic!("expected General after replace");
        }
    }

    #[tokio::test]
    async fn freetext_summary_round_trips() {
        // The fallback path the use case takes when the LLM keeps
        // returning malformed JSON: a `FreeText` payload must
        // serialise/deserialise without losing its discriminator.
        let store = SqliteMeetingStore::open_in_memory().await.unwrap();
        let mid = make_meeting(&store).await;

        let s = Summary {
            id: SummaryId::new(),
            meeting_id: mid,
            model: "qwen2.5-7b".into(),
            language: None,
            created_at: OffsetDateTime::now_utc(),
            content: SummaryContent::FreeText {
                text: "raw model output that was not valid JSON".into(),
            },
        };
        store.upsert_summary(&s).await.unwrap();

        let fetched = store.get_summary(mid).await.unwrap().unwrap();
        match fetched.content {
            SummaryContent::FreeText { text } => {
                assert_eq!(text, "raw model output that was not valid JSON");
            }
            _ => panic!("expected FreeText variant"),
        }
    }

    #[tokio::test]
    async fn deleting_meeting_cascades_to_summary() {
        let store = SqliteMeetingStore::open_in_memory().await.unwrap();
        let mid = make_meeting(&store).await;
        store
            .upsert_summary(&general_summary(mid, "qwen2.5-7b", Some("es")))
            .await
            .unwrap();
        assert!(store.get_summary(mid).await.unwrap().is_some());

        store.delete(mid).await.unwrap();
        assert!(
            store.get_summary(mid).await.unwrap().is_none(),
            "ON DELETE CASCADE failed: stale summary leaked"
        );
    }

    #[tokio::test]
    async fn open_creates_file_and_persists_across_pools() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested/echo.db");
        let id = MeetingId::new();
        {
            let store = SqliteMeetingStore::open(&path).await.unwrap();
            store
                .create(CreateMeeting {
                    id,
                    title: "persisted".into(),
                    input_format: fmt(),
                })
                .await
                .unwrap();
            store.close().await;
        }
        let store = SqliteMeetingStore::open(&path).await.unwrap();
        let listed = store.list(0).await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, id);
    }
}
