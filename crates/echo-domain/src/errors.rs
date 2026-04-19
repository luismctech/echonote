//! Domain-level error taxonomy.
//!
//! Errors in this module never reference concrete I/O or library errors.
//! Infrastructure adapters are responsible for mapping their own errors onto
//! a `DomainError` variant when crossing the layer boundary.

use thiserror::Error;

/// The root error returned by any domain port.
///
/// The variants intentionally remain coarse during scaffolding and will be
/// refined as concrete use cases land in later sprints.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DomainError {
    /// A required audio device is not available.
    #[error("audio device unavailable: {0}")]
    AudioDeviceUnavailable(String),

    /// The host exposes the device but the requested format is not
    /// supported (sample rate, channel count, sample format).
    #[error("audio format unsupported: {0}")]
    AudioFormatUnsupported(String),

    /// The capture engine itself reported a runtime error.
    #[error("audio capture failed: {0}")]
    AudioCaptureFailed(String),

    /// A model required for transcription or summarization is not loaded.
    #[error("model not loaded: {0}")]
    ModelNotLoaded(String),

    /// Voice activity detection failed at runtime (model inference,
    /// state corruption, sample-rate mismatch).
    #[error("voice activity detection failed: {0}")]
    VadFailed(String),

    /// Speaker diarization failed (embedding inference, clustering
    /// degeneration, or sample-rate mismatch).
    #[error("speaker diarization failed: {0}")]
    DiarizationFailed(String),

    /// The requested session does not exist or is in an invalid state.
    #[error("invalid session state: {0}")]
    InvalidSessionState(String),

    /// Generic invariant violation. Prefer adding a specific variant when
    /// the error recurs in multiple places.
    #[error("domain invariant violated: {0}")]
    Invariant(String),
}
