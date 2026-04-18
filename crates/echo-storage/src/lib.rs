//! # echo-storage
//!
//! SQLite-backed persistence adapter. Implements the `Storage` port of
//! `echo-domain` using `sqlx` with SQLite + FTS5, optionally wrapped in
//! SQLCipher for at-rest encryption.
//!
//! Schema and migrations live under `migrations/` (to be created Sprint 3)
//! and follow the structure described in `docs/ARCHITECTURE.md` §8.2.

#![forbid(unsafe_code)]
#![warn(rust_2018_idioms, clippy::all)]
