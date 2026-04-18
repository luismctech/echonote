//! # echo-telemetry
//!
//! Shared observability primitives for EchoNote: a [`init`] helper that
//! configures `tracing` with a sensible default subscriber (env filter +
//! human-readable fmt layer), plus reusable span/metric helpers to be
//! added as the application layer matures.
//!
//! Telemetry transmission to remote services is strictly opt-in and is
//! gated at the Tauri layer, not here. See `docs/ARCHITECTURE.md` §10.

#![forbid(unsafe_code)]
#![warn(missing_docs, rust_2018_idioms, clippy::all)]

use tracing_subscriber::{fmt, prelude::*, EnvFilter};

/// Initialize a default tracing subscriber for CLI and library callers.
///
/// Reads the log level from the `ECHO_LOG` env var (fallback `RUST_LOG`,
/// fallback `info`). Safe to call more than once — the second call is a
/// no-op because `tracing_subscriber::registry` uses a global default.
pub fn init() {
    let filter = EnvFilter::try_from_env("ECHO_LOG")
        .or_else(|_| EnvFilter::try_from_default_env())
        .unwrap_or_else(|_| EnvFilter::new("info"));

    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().with_target(false).compact())
        .try_init();
}
