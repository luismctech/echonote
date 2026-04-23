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
pub use ports::audio::{
    AudioCapture, AudioFormat, AudioFrame, AudioSource, AudioStream, CaptureSpec, DeviceInfo,
    Sample,
};
pub use ports::chat::{ChatAssistant, ChatMessage, ChatOptions, ChatRequest, ChatRole, ChatToken};
pub use ports::diarizer::Diarizer;
pub use ports::llm::{GenerateOptions, LlmModel};
pub use ports::resampler::Resampler;
pub use ports::transcriber::{TranscribeOptions, Transcriber, Transcript};
pub use ports::vad::{Vad, VoiceState};

pub use entities::meeting::{Meeting, MeetingId, MeetingSearchHit, MeetingSummary};
pub use entities::segment::{Segment, SegmentId};
pub use entities::speaker::{Speaker, SpeakerId};
pub use entities::streaming::{StreamingOptions, StreamingSessionId, TranscriptEvent};
pub use entities::summary::{ActionItem, Summary, SummaryContent, SummaryId};
pub use ports::storage::{CreateMeeting, FinalizeMeeting, MeetingStore};
