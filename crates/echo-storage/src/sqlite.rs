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
    AudioFormat, CreateMeeting, DomainError, FinalizeMeeting, Meeting, MeetingId, MeetingStore,
    MeetingSummary, Segment, SegmentId, SpeakerId,
};

/// Embedded migrations (`crates/echo-storage/migrations/`).
static MIGRATIONS: sqlx::migrate::Migrator = sqlx::migrate!("./migrations");

fn db_err(msg: impl Into<String>) -> DomainError {
    DomainError::Invariant(msg.into())
}

fn map_sqlx<E: std::fmt::Display>(prefix: &'static str) -> impl Fn(E) -> DomainError {
    move |e| DomainError::Invariant(format!("{prefix}: {e}"))
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

        Ok(Some(Meeting {
            summary,
            input_format,
            segments,
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
