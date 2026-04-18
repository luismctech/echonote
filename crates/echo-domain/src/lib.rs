//! # echo-domain
//!
//! The pure domain layer of EchoNote. Holds the language of the business:
//! entities, value objects, domain errors and the **ports** (trait
//! abstractions) that outer layers implement.
//!
//! Rules enforced by this crate:
//!
//! 1. No dependency on I/O, filesystem, network or OS APIs.
//! 2. No dependency on frameworks (Tauri, tokio runtime, sqlx, whisper).
//! 3. Every public item is `Send + Sync` where it makes sense, to ease use
//!    across async boundaries in the outer layers.
//!
//! See `docs/ARCHITECTURE.md` §4 for the layering contract.

#![forbid(unsafe_code)]
#![warn(missing_docs, rust_2018_idioms, clippy::all)]

pub mod entities;
pub mod errors;
pub mod ports;

pub use errors::DomainError;
