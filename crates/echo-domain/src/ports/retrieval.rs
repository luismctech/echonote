//! Segment retrieval port.
//!
//! The chat use case grounds its answers on the whole transcript when it
//! fits the model's context window. When it does not, it asks a
//! [`SegmentRetriever`] for the segments most relevant to the question and
//! fills the window with those.
//!
//! This trait is the seam for retrieval strategies: the first adapter is
//! lexical (SQLite FTS5 + BM25), and a semantic or hybrid retriever
//! (embeddings, reciprocal-rank fusion, ...) can replace or wrap it without
//! touching the use case.

use async_trait::async_trait;

use crate::entities::meeting::MeetingId;
use crate::entities::segment::SegmentId;
use crate::errors::DomainError;

/// One segment returned by a [`SegmentRetriever`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SegmentHit {
    /// The matching segment.
    pub segment_id: SegmentId,
    /// Adapter-specific relevance score, **lower is better** (SQLite's
    /// `bm25()` convention). Informational: callers must rely on the order
    /// of the returned list, never compare scores across adapters.
    pub rank: f64,
}

/// Finds the segments of one meeting that are relevant to a free-text query.
#[async_trait]
pub trait SegmentRetriever: Send + Sync {
    /// Return at most `limit` segments of `meeting_id` relevant to `query`,
    /// best match first.
    ///
    /// `query` is natural language typed by a user, not a search
    /// expression: implementations must treat it as plain text. A query with
    /// nothing searchable (empty, only stopwords or punctuation) yields an
    /// empty list, not an error. Only segments of `meeting_id` may be
    /// returned.
    ///
    /// # Errors
    ///
    /// [`DomainError::Storage`] (or the adapter's equivalent) when the
    /// underlying index cannot be queried.
    async fn search_segments(
        &self,
        meeting_id: MeetingId,
        query: &str,
        limit: u32,
    ) -> Result<Vec<SegmentHit>, DomainError>;
}
