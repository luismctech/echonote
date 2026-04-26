//! Cross-cutting application services.
//!
//! Today this is just `meeting_recorder` (the streaming → SQLite
//! persistence service) and `wer` (the offline benchmark helper used
//! by `echo-proto bench wer`). Earlier plans staked out modules for a
//! generic session registry and an in-process event bus, but both
//! remained empty and were dropped — the streaming pipeline owns its
//! own session lifecycle inside `src-tauri/src/commands.rs`, and there
//! are no cross-component subscribers that would benefit from a bus.
//! Reintroduce as named modules when a second consumer appears.

pub mod meeting_recorder;
pub mod wer;
