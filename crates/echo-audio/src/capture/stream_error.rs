//! Severity classification for [`cpal::StreamError`].
//!
//! cpal reports every hiccup its audio worker sees through the stream
//! error callback, but only some of them mean the stream is dead. The
//! ALSA host in particular calls the error callback and then **keeps
//! polling** — its worker maps the error to `PollDescriptorsFlow::
//! Continue`, so the stream is still live once the callback returns.
//!
//! Treating those as fatal is what broke Linux capture: on PipeWire's
//! ALSA plugin `poll()` returns 0 (timeout) several times per second,
//! cpal surfaces it as ``BackendSpecific("`alsa::poll()` spuriously
//! returned")``, and a caller that tears the stream down on any error
//! ends up in a rebuild loop that never delivers a frame.
//!
//! Callers should reconnect only on [`Severity::Fatal`]. Genuinely
//! stalled-but-error-free devices are covered by the starvation
//! watchdog in [`super::cpal_microphone`], so misclassifying a real
//! failure as transient costs at most one watchdog period.

/// How a caller should react to a [`cpal::StreamError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// The stream is still alive; log and keep going.
    Transient,
    /// The stream is unusable; drop it and reconnect.
    Fatal,
}

/// Backend error descriptions known to be recoverable.
///
/// Matched case-insensitively as substrings, because the surrounding
/// text differs between cpal versions and ALSA builds.
const TRANSIENT_BACKEND_MARKERS: &[&str] = &[
    // ALSA `poll()` returned 0 descriptors (timeout). Routine on the
    // PipeWire and PulseAudio ALSA plugins; cpal's worker continues.
    "spuriously returned",
    // ALSA xrun recovery paths surface EAGAIN while the ring buffer
    // refills; the next poll cycle delivers audio again.
    "resource temporarily unavailable",
];

/// Classify a stream error.
#[must_use]
pub fn classify(err: &cpal::StreamError) -> Severity {
    match err {
        // The device is gone (unplugged, taken by an exclusive-mode
        // client, profile switched). Only a rebuild can recover.
        cpal::StreamError::DeviceNotAvailable | cpal::StreamError::StreamInvalidated => {
            Severity::Fatal
        }
        // A glitch in the audio, not a dead stream — cpal keeps going.
        cpal::StreamError::BufferUnderrun => Severity::Transient,
        cpal::StreamError::BackendSpecific { err } => {
            let description = err.description.to_lowercase();
            if TRANSIENT_BACKEND_MARKERS
                .iter()
                .any(|marker| description.contains(marker))
            {
                Severity::Transient
            } else {
                Severity::Fatal
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cpal::{BackendSpecificError, StreamError};
    use pretty_assertions::assert_eq;

    fn backend(description: &str) -> StreamError {
        StreamError::BackendSpecific {
            err: BackendSpecificError {
                description: description.to_string(),
            },
        }
    }

    #[test]
    fn alsa_spurious_poll_is_transient() {
        assert_eq!(
            classify(&backend("`alsa::poll()` spuriously returned")),
            Severity::Transient
        );
    }

    #[test]
    fn marker_match_is_case_insensitive() {
        assert_eq!(
            classify(&backend("ALSA::POLL() SPURIOUSLY RETURNED")),
            Severity::Transient
        );
    }

    #[test]
    fn buffer_underrun_is_transient() {
        assert_eq!(classify(&StreamError::BufferUnderrun), Severity::Transient);
    }

    #[test]
    fn device_loss_is_fatal() {
        assert_eq!(classify(&StreamError::DeviceNotAvailable), Severity::Fatal);
        assert_eq!(classify(&StreamError::StreamInvalidated), Severity::Fatal);
    }

    #[test]
    fn unknown_backend_error_is_fatal() {
        assert_eq!(classify(&backend("no such device")), Severity::Fatal);
    }
}
