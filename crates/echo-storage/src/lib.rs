//! # echo-storage
//!
//! SQLite-backed persistence adapter. Implements [`MeetingStore`] using
//! `sqlx` with the bundled SQLite driver. Migrations are embedded at
//! compile time from `migrations/`.
//!
//! Concurrency model:
//!
//! * The pool is `WAL` + `synchronous = NORMAL`, which gives us
//!   concurrent readers and a single writer with crash-safe durability.
//! * Per-meeting writes are serialized inside one transaction in
//!   [`SqliteMeetingStore::append_segments`] so the streaming pipeline
//!   never observes a half-written chunk.
//!
//! At-rest encryption (SQLCipher) lands in Sprint 3 behind a cargo
//! feature flag — the API surface is already future-proof.

#![forbid(unsafe_code)]
#![warn(missing_docs, rust_2018_idioms, clippy::all)]

mod sqlite;

pub use sqlite::SqliteMeetingStore;
