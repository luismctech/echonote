//! Severity classification for stream errors reported by cpal.
//!
//! cpal reports every hiccup its audio worker sees through the stream
//! error callback as a [`cpal::Error`], but only some of them mean the
//! stream is dead. The ALSA host in particular calls the error callback
//! and then **keeps polling** — its worker maps the error to
//! `PollDescriptorsFlow::Continue`, so the stream is still live once the
//! callback returns.
//!
//! Treating those as fatal is what broke Linux capture: on PipeWire's
//! ALSA plugin `poll()` returns 0 (timeout) several times per second,
//! cpal surfaces it as a backend error whose message reads
//! ``"`alsa::poll()` spuriously returned"``, and a caller that tears the
//! stream down on any error ends up in a rebuild loop that never
//! delivers a frame.
//!
//! The mapping onto [`cpal::ErrorKind`] (cpal 0.18):
//!
//! | Kind | Severity | Why |
//! |------|----------|-----|
//! | `Xrun` | Transient | A glitch in the audio; cpal keeps the stream running. |
//! | `DeviceChanged` | Transient | The stream was rerouted to the new default device and stays active. |
//! | `RealtimeDenied` | Transient | Audio still flows, only without real-time priority. |
//! | `BackendError`, `Other` | Transient only when the message matches a known recoverable marker | ALSA poll timeouts and EAGAIN arrive this way. |
//! | everything else | Fatal | Device gone, busy, invalidated, denied, or an unknown condition. |
//!
//! Callers should reconnect only on [`Severity::Fatal`]. Genuinely
//! stalled-but-error-free devices are covered by the starvation
//! watchdog in [`super::cpal_microphone`], so misclassifying a real
//! failure as transient costs at most one watchdog period.

/// How a caller should react to a stream [`cpal::Error`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// The stream is still alive; log and keep going.
    Transient,
    /// The stream is unusable; drop it and reconnect.
    Fatal,
}

/// Backend error messages known to be recoverable.
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

/// Classify a stream error reported through cpal's error callback.
#[must_use]
pub fn classify(err: &cpal::Error) -> Severity {
    use cpal::ErrorKind;

    match err.kind() {
        // A glitch in the audio, not a dead stream — cpal keeps going.
        ErrorKind::Xrun => Severity::Transient,
        // cpal already moved the stream to the new default route; a
        // rebuild would only add a gap.
        ErrorKind::DeviceChanged => Severity::Transient,
        // Audio still flows, just without real-time scheduling.
        ErrorKind::RealtimeDenied => Severity::Transient,
        // Unmapped backend errors: only the ones we know the backend
        // survives are transient (see `TRANSIENT_BACKEND_MARKERS`).
        ErrorKind::BackendError | ErrorKind::Other => {
            if has_transient_marker(err.message()) {
                Severity::Transient
            } else {
                Severity::Fatal
            }
        }
        // DeviceNotAvailable, StreamInvalidated, DeviceBusy,
        // HostUnavailable, PermissionDenied, ResourceExhausted and the
        // rest: only a rebuild can recover. `ErrorKind` is
        // `#[non_exhaustive]`, so kinds added by future cpal releases
        // also land here: a needless reconnect is cheaper than a stream
        // that silently stopped.
        _ => Severity::Fatal,
    }
}

fn has_transient_marker(message: Option<&str>) -> bool {
    let Some(message) = message else {
        return false;
    };
    let message = message.to_lowercase();
    TRANSIENT_BACKEND_MARKERS
        .iter()
        .any(|marker| message.contains(marker))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cpal::{Error, ErrorKind};
    use pretty_assertions::assert_eq;

    fn err(kind: ErrorKind, message: &'static str) -> Error {
        Error::with_message(kind, message)
    }

    #[test]
    fn alsa_spurious_poll_is_transient() {
        assert_eq!(
            classify(&err(
                ErrorKind::BackendError,
                "`alsa::poll()` spuriously returned"
            )),
            Severity::Transient
        );
    }

    #[test]
    fn alsa_eagain_is_transient() {
        assert_eq!(
            classify(&err(
                ErrorKind::BackendError,
                "Resource temporarily unavailable"
            )),
            Severity::Transient
        );
    }

    #[test]
    fn marker_match_is_case_insensitive() {
        assert_eq!(
            classify(&err(
                ErrorKind::BackendError,
                "ALSA::POLL() SPURIOUSLY RETURNED"
            )),
            Severity::Transient
        );
    }

    #[test]
    fn xrun_is_transient() {
        assert_eq!(classify(&Error::new(ErrorKind::Xrun)), Severity::Transient);
    }

    #[test]
    fn rerouted_stream_is_transient() {
        assert_eq!(
            classify(&Error::new(ErrorKind::DeviceChanged)),
            Severity::Transient
        );
    }

    #[test]
    fn realtime_denied_is_transient() {
        assert_eq!(
            classify(&Error::new(ErrorKind::RealtimeDenied)),
            Severity::Transient
        );
    }

    #[test]
    fn device_loss_is_fatal() {
        assert_eq!(
            classify(&Error::new(ErrorKind::DeviceNotAvailable)),
            Severity::Fatal
        );
        assert_eq!(
            classify(&Error::new(ErrorKind::StreamInvalidated)),
            Severity::Fatal
        );
    }

    #[test]
    fn busy_device_is_fatal() {
        assert_eq!(
            classify(&Error::new(ErrorKind::DeviceBusy)),
            Severity::Fatal
        );
    }

    #[test]
    fn unknown_backend_error_is_fatal() {
        assert_eq!(
            classify(&err(ErrorKind::BackendError, "no such device")),
            Severity::Fatal
        );
        assert_eq!(
            classify(&Error::new(ErrorKind::BackendError)),
            Severity::Fatal
        );
    }

    #[test]
    fn other_with_marker_is_transient() {
        assert_eq!(
            classify(&err(ErrorKind::Other, "`alsa::poll()` spuriously returned")),
            Severity::Transient
        );
    }

    #[test]
    fn other_without_marker_is_fatal() {
        assert_eq!(
            classify(&err(ErrorKind::Other, "something broke")),
            Severity::Fatal
        );
        assert_eq!(classify(&Error::new(ErrorKind::Other)), Severity::Fatal);
    }
}
