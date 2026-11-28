//! [`AudioCapture`] composite that tries a preferred adapter first.
//!
//! On macOS the Core Audio process tap is preferred for system audio and
//! ScreenCaptureKit takes over when the tap cannot list or start (macOS
//! older than 14.2, no output device, a Core Audio error).

use std::sync::Arc;

use async_trait::async_trait;
use tracing::warn;

use echo_domain::{AudioCapture, AudioSource, AudioStream, CaptureSpec, DeviceInfo, DomainError};

/// Uses `primary`, and `fallback` whenever `primary` fails.
pub struct FallbackAudioCapture {
    primary: Arc<dyn AudioCapture>,
    fallback: Arc<dyn AudioCapture>,
}

impl FallbackAudioCapture {
    #[must_use]
    pub fn new(primary: Arc<dyn AudioCapture>, fallback: Arc<dyn AudioCapture>) -> Self {
        Self { primary, fallback }
    }
}

#[async_trait]
impl AudioCapture for FallbackAudioCapture {
    async fn list_devices(&self, source: AudioSource) -> Result<Vec<DeviceInfo>, DomainError> {
        match self.primary.list_devices(source).await {
            Ok(devices) => Ok(devices),
            Err(e) => {
                warn!(error = %e, "preferred capture cannot list devices; using the fallback");
                self.fallback.list_devices(source).await
            }
        }
    }

    async fn start(&self, spec: CaptureSpec) -> Result<Box<dyn AudioStream>, DomainError> {
        match self.primary.start(spec.clone()).await {
            Ok(stream) => Ok(stream),
            Err(e) => {
                warn!(error = %e, "preferred capture failed to start; using the fallback");
                self.fallback.start(spec).await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use echo_domain::{AudioFormat, AudioFrame};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Adapter that either works (yielding a stream in `format`) or
    /// fails, counting calls.
    struct Fake {
        format: Option<AudioFormat>,
        starts: AtomicUsize,
        lists: AtomicUsize,
    }

    impl Fake {
        fn working(rate: u32) -> Arc<Self> {
            Arc::new(Self {
                format: Some(AudioFormat {
                    sample_rate_hz: rate,
                    channels: 2,
                }),
                starts: AtomicUsize::new(0),
                lists: AtomicUsize::new(0),
            })
        }
        fn broken() -> Arc<Self> {
            Arc::new(Self {
                format: None,
                starts: AtomicUsize::new(0),
                lists: AtomicUsize::new(0),
            })
        }
        fn err() -> DomainError {
            DomainError::AudioDeviceUnavailable("broken".into())
        }
    }

    struct Stream(AudioFormat);

    #[async_trait]
    impl AudioStream for Stream {
        fn format(&self) -> AudioFormat {
            self.0
        }
        async fn next_frame(&mut self) -> Option<AudioFrame> {
            None
        }
        async fn stop(&mut self) -> Result<(), DomainError> {
            Ok(())
        }
    }

    #[async_trait]
    impl AudioCapture for Fake {
        async fn list_devices(&self, _source: AudioSource) -> Result<Vec<DeviceInfo>, DomainError> {
            self.lists.fetch_add(1, Ordering::SeqCst);
            let format = self.format.ok_or_else(Self::err)?;
            Ok(vec![DeviceInfo {
                id: format.sample_rate_hz.to_string(),
                name: "fake".into(),
                is_default: true,
            }])
        }
        async fn start(&self, _spec: CaptureSpec) -> Result<Box<dyn AudioStream>, DomainError> {
            self.starts.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(Stream(self.format.ok_or_else(Self::err)?)))
        }
    }

    fn system() -> CaptureSpec {
        CaptureSpec {
            source: AudioSource::SystemOutput,
            device_id: None,
            preferred_format: AudioFormat::WHISPER,
        }
    }

    #[tokio::test]
    async fn a_working_primary_is_used_alone() {
        let (primary, fallback) = (Fake::working(48_000), Fake::working(44_100));
        let capture = FallbackAudioCapture::new(primary.clone(), fallback.clone());
        let stream = capture.start(system()).await.unwrap();
        assert_eq!(stream.format().sample_rate_hz, 48_000);
        let devices = capture
            .list_devices(AudioSource::SystemOutput)
            .await
            .unwrap();
        assert_eq!(devices[0].id, "48000");
        assert_eq!(fallback.starts.load(Ordering::SeqCst), 0);
        assert_eq!(fallback.lists.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn a_failing_primary_hands_over_to_the_fallback() {
        let (primary, fallback) = (Fake::broken(), Fake::working(44_100));
        let capture = FallbackAudioCapture::new(primary.clone(), fallback.clone());
        let stream = capture.start(system()).await.unwrap();
        assert_eq!(stream.format().sample_rate_hz, 44_100);
        let devices = capture
            .list_devices(AudioSource::SystemOutput)
            .await
            .unwrap();
        assert_eq!(devices[0].id, "44100");
        assert_eq!(primary.starts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn when_both_fail_the_fallback_error_is_returned() {
        let capture = FallbackAudioCapture::new(Fake::broken(), Fake::broken());
        let outcome = capture.start(system()).await;
        assert!(matches!(
            outcome,
            Err(DomainError::AudioDeviceUnavailable(_))
        ));
    }
}
