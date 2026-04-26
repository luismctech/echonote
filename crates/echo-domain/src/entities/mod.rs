//! Domain entities.
//!
//! Each submodule owns the definition of an entity plus its value objects.
//! All entities are serializable so they can be persisted by the storage
//! adapter and emitted to the frontend across the Tauri IPC boundary.
//!
//! Concrete types will be fleshed out during Sprint 0 days 5-9 (see
//! `docs/DEVELOPMENT_PLAN.md` §6.2, epic E2 onwards).

pub mod meeting;
pub mod segment;
pub mod speaker;
pub mod streaming;
pub mod summary;
