//! Structured IPC error type for the Tauri command boundary.
//!
//! Every `#[tauri::command]` returns `Result<T, IpcError>` so the
//! frontend receives a machine-readable `code` alongside the
//! human-readable `message`, enabling precise error handling (retry
//! vs fatal vs user-action-needed) without string parsing.

use echo_app::{AskAboutMeetingError, RenameSpeakerError, SummarizeMeetingError};
use echo_domain::DomainError;
use serde::Serialize;

/// Machine-readable error codes the frontend can match on.
///
/// Kept intentionally coarse — add variants when the frontend needs
/// to distinguish a new failure mode, not for every conceivable
/// backend error.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, specta::Type)]
#[serde(rename_all = "camelCase")]
pub enum ErrorCode {
    /// A required record (meeting, speaker, segment, …) was not found.
    NotFound,
    /// Persistent storage failed (SQLite I/O, migration, …).
    Storage,
    /// The local LLM could not load or inference failed.
    Llm,
    /// The ASR engine could not load or transcription failed.
    Asr,
    /// Input validation failed (empty question, bad template, …).
    InvalidInput,
    /// A streaming session conflict (poisoned map, duplicate id, …).
    SessionConflict,
    /// A required model is not downloaded yet.
    ModelNotReady,
    /// Audio capture or device error.
    Audio,
    /// Voice activity detection failed.
    Vad,
    /// Speaker diarization failed.
    Diarization,
    /// An HTTP / network operation failed (model download, …).
    Network,
    /// Catch-all for unexpected / unclassified errors.
    Internal,
}

impl ErrorCode {
    /// Whether the frontend should offer a retry affordance.
    pub const fn retriable(self) -> bool {
        matches!(
            self,
            ErrorCode::Storage
                | ErrorCode::Llm
                | ErrorCode::Asr
                | ErrorCode::Network
                | ErrorCode::Vad
                | ErrorCode::Diarization
                | ErrorCode::Internal
        )
    }
}

/// Structured error returned by every Tauri command.
///
/// Serializes to `{ "code": "notFound", "message": "…", "retriable": false }`
/// on the IPC wire, which the frontend's `useIpcAction` hook can
/// destructure for precise UX.
#[derive(Debug, Clone, Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct IpcError {
    pub code: ErrorCode,
    pub message: String,
    pub retriable: bool,
}

impl IpcError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            retriable: code.retriable(),
            code,
            message: message.into(),
        }
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::NotFound, message)
    }

    pub fn storage(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::Storage, message)
    }

    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::InvalidInput, message)
    }

    pub fn model_not_ready(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::ModelNotReady, message)
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::Internal, message)
    }

    pub fn network(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::Network, message)
    }
}

// ---------------------------------------------------------------------------
// From impls for domain / use-case errors
// ---------------------------------------------------------------------------

impl From<DomainError> for IpcError {
    fn from(e: DomainError) -> Self {
        match &e {
            DomainError::AudioDeviceUnavailable(_)
            | DomainError::AudioFormatUnsupported(_)
            | DomainError::AudioCaptureFailed(_) => IpcError::new(ErrorCode::Audio, e.to_string()),
            DomainError::ModelNotLoaded(_) => IpcError::model_not_ready(e.to_string()),
            DomainError::VadFailed(_) => IpcError::new(ErrorCode::Vad, e.to_string()),
            DomainError::DiarizationFailed(_) => {
                IpcError::new(ErrorCode::Diarization, e.to_string())
            }
            DomainError::InvalidSessionState(_) => {
                IpcError::new(ErrorCode::SessionConflict, e.to_string())
            }
            DomainError::NotFound { .. } => IpcError::not_found(e.to_string()),
            DomainError::Storage(_) => IpcError::storage(e.to_string()),
            DomainError::LlmFailed(_) => IpcError::new(ErrorCode::Llm, e.to_string()),
            DomainError::Invariant(_) => IpcError::internal(e.to_string()),
            _ => IpcError::internal(e.to_string()),
        }
    }
}

impl From<SummarizeMeetingError> for IpcError {
    fn from(e: SummarizeMeetingError) -> Self {
        match &e {
            SummarizeMeetingError::NotFound(_) => IpcError::not_found(e.to_string()),
            SummarizeMeetingError::EmptyTranscript(_) => IpcError::invalid_input(e.to_string()),
            SummarizeMeetingError::InvalidTemplate(_) => IpcError::invalid_input(e.to_string()),
            SummarizeMeetingError::Llm(_) => IpcError::new(ErrorCode::Llm, e.to_string()),
            SummarizeMeetingError::Storage(_) => IpcError::storage(e.to_string()),
        }
    }
}

impl From<AskAboutMeetingError> for IpcError {
    fn from(e: AskAboutMeetingError) -> Self {
        match &e {
            AskAboutMeetingError::NotFound(_) => IpcError::not_found(e.to_string()),
            AskAboutMeetingError::EmptyQuestion => IpcError::invalid_input(e.to_string()),
            AskAboutMeetingError::EmptyTranscript(_) => IpcError::invalid_input(e.to_string()),
            AskAboutMeetingError::Chat(_) => IpcError::new(ErrorCode::Llm, e.to_string()),
            AskAboutMeetingError::Storage(_) => IpcError::storage(e.to_string()),
        }
    }
}

impl From<RenameSpeakerError> for IpcError {
    fn from(e: RenameSpeakerError) -> Self {
        match &e {
            RenameSpeakerError::NotFound { .. } => IpcError::not_found(e.to_string()),
            RenameSpeakerError::Invalid(_) => IpcError::invalid_input(e.to_string()),
            RenameSpeakerError::Storage(_) => IpcError::storage(e.to_string()),
        }
    }
}
