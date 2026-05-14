//! Source-routing [`AudioCapture`] composite.
//!
//! `RoutingAudioCapture` wraps the per-source adapters so callers can
//! work against a single `Arc<dyn AudioCapture>` regardless of whether
//! they intend to capture the microphone, the system audio mix, or both
//! in the same session.
//!
//! ## Layout
//!
//! ```text
//! ┌───────────────────────┐    Microphone     ┌─────────────────────┐
//! │ RoutingAudioCapture   ├──────────────────▶│ CpalMicrophoneCapture│
//! │ (Arc<dyn …> facade)   │                   └─────────────────────┘
//! │                       │  SystemOutput      ┌─────────────────────┐
//! │                       ├──────────────────▶│ ScreenCaptureKit-   │
//! │                       │                   │ Capture (macOS)     │
//! │                       │  Mixed             └─────────────────────┘
//! │                       ├──────────────────▶ Both adapters → MixedStream
//! └───────────────────────┘
//! ```
//!
//! On unsupported targets the system-output slot is `None` and any
//! request for `AudioSource::SystemOutput` or `AudioSource::Mixed`
//! returns [`DomainError::AudioDeviceUnavailable`] with a clear message.

use std::sync::Arc;

use async_trait::async_trait;
use echo_domain::{
    AudioCapture, AudioFormat, AudioSource, AudioStream, CaptureSpec, DeviceInfo, DomainError,
};

use super::mixed::{MixControls, MixedStream};

use crate::CpalMicrophoneCapture;

#[cfg(target_os = "macos")]
use super::ScreenCaptureKitCapture;

#[cfg(target_os = "windows")]
use super::WasapiLoopbackCapture;

#[cfg(target_os = "linux")]
use super::PulseMonitorCapture;

/// Composite [`AudioCapture`] that delegates to a per-source adapter.
///
/// Construct with [`RoutingAudioCapture::with_default_adapters`] for
/// the standard EchoNote setup (cpal mic + ScreenCaptureKit system on
/// macOS), or with [`RoutingAudioCapture::new`] when injecting custom
/// adapters from a test.
#[derive(Clone)]
pub struct RoutingAudioCapture {
    microphone: Arc<dyn AudioCapture>,
    system_output: Option<Arc<dyn AudioCapture>>,
}

impl std::fmt::Debug for RoutingAudioCapture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RoutingAudioCapture")
            .field("microphone", &"<Arc<dyn AudioCapture>>")
            .field(
                "system_output",
                &self
                    .system_output
                    .as_ref()
                    .map(|_| "<Arc<dyn AudioCapture>>"),
            )
            .finish()
    }
}

impl RoutingAudioCapture {
    /// Build with explicit adapters. Pass `system_output = None` to
    /// reject [`AudioSource::SystemOutput`] requests with a clean
    /// error (used in tests and on non-macOS targets).
    #[must_use]
    pub fn new(
        microphone: Arc<dyn AudioCapture>,
        system_output: Option<Arc<dyn AudioCapture>>,
    ) -> Self {
        Self {
            microphone,
            system_output,
        }
    }

    /// Build with the default per-OS adapters:
    ///
    /// - microphone: [`CpalMicrophoneCapture`] on every platform.
    /// - system output: [`ScreenCaptureKitCapture`] on macOS,
    ///   [`WasapiLoopbackCapture`] on Windows,
    ///   [`PulseMonitorCapture`] on Linux.
    #[must_use]
    pub fn with_default_adapters() -> Self {
        let microphone: Arc<dyn AudioCapture> = Arc::new(CpalMicrophoneCapture::new());

        #[cfg(target_os = "macos")]
        let system_output: Option<Arc<dyn AudioCapture>> =
            Some(Arc::new(ScreenCaptureKitCapture::new()));

        #[cfg(target_os = "windows")]
        let system_output: Option<Arc<dyn AudioCapture>> =
            Some(Arc::new(WasapiLoopbackCapture::new()));

        #[cfg(target_os = "linux")]
        let system_output: Option<Arc<dyn AudioCapture>> =
            Some(Arc::new(PulseMonitorCapture::new()));

        #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
        let system_output: Option<Arc<dyn AudioCapture>> = None;

        Self::new(microphone, system_output)
    }

    fn pick(&self, source: AudioSource) -> Result<&Arc<dyn AudioCapture>, DomainError> {
        match source {
            AudioSource::Microphone => Ok(&self.microphone),
            AudioSource::SystemOutput | AudioSource::Mixed => {
                self.system_output.as_ref().ok_or_else(|| {
                    DomainError::AudioDeviceUnavailable(
                        "system audio capture is not available on this platform".into(),
                    )
                })
            }
        }
    }

    /// Start both the microphone and system-audio streams and merge them
    /// into a single [`MixedStream`]. Returns the stream and a
    /// [`MixControls`] handle that can mute each source independently
    /// during the live session.
    ///
    /// The mixer normalizes each source independently (channel downmix +
    /// sample-rate conversion) to [`AudioFormat::WHISPER`] (16 kHz mono)
    /// before combining them. This makes the mix correct even when the
    /// mic and system streams negotiate different native formats — the
    /// common case on macOS, where built-in mics expose mono 48 kHz while
    /// ScreenCaptureKit delivers stereo 48 kHz.
    pub async fn start_mixed(
        &self,
        spec: CaptureSpec,
    ) -> Result<(Box<dyn AudioStream>, MixControls), DomainError> {
        let sys_adapter = self.system_output.as_ref().ok_or_else(|| {
            DomainError::AudioDeviceUnavailable(
                "system audio capture is not available on this platform; \
                 Mixed mode requires both microphone and system-audio support"
                    .into(),
            )
        })?;

        // Each source is started at its native preferred rate (48 kHz
        // stereo is a reasonable hint for both cpal and ScreenCaptureKit;
        // adapters fall back to whatever the device actually supports).
        // The MixedStream then downmixes + resamples each stream to the
        // canonical Whisper target so the per-sample mix is time-aligned.
        let hint = AudioFormat {
            sample_rate_hz: 48_000,
            channels: 2,
        };

        let mic_spec = CaptureSpec {
            source: AudioSource::Microphone,
            device_id: spec.device_id.clone(),
            preferred_format: hint,
        };
        let sys_spec = CaptureSpec {
            source: AudioSource::SystemOutput,
            device_id: None,
            preferred_format: hint,
        };

        let mic_stream = self.microphone.start(mic_spec).await?;
        let sys_stream = sys_adapter.clone().start(sys_spec).await?;

        let (stream, controls) = MixedStream::new(mic_stream, sys_stream, AudioFormat::WHISPER)?;
        Ok((Box::new(stream), controls))
    }
}

impl Default for RoutingAudioCapture {
    fn default() -> Self {
        Self::with_default_adapters()
    }
}

#[async_trait]
impl AudioCapture for RoutingAudioCapture {
    async fn list_devices(&self, source: AudioSource) -> Result<Vec<DeviceInfo>, DomainError> {
        if source == AudioSource::Mixed {
            // Synthesise a single logical "mixed" device.
            return Ok(vec![DeviceInfo {
                id: "mixed:default".to_string(),
                name: "Mixed (Microphone + System Audio)".to_string(),
                is_default: true,
            }]);
        }
        self.pick(source)?.list_devices(source).await
    }

    async fn start(&self, spec: CaptureSpec) -> Result<Box<dyn AudioStream>, DomainError> {
        if spec.source == AudioSource::Mixed {
            // Discard mix controls — callers that need live-toggle must use
            // `start_mixed` directly so they can retain the control handles.
            let (stream, _controls) = self.start_mixed(spec).await?;
            return Ok(stream);
        }
        let adapter = self.pick(spec.source)?.clone();
        adapter.start(spec).await
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use echo_domain::{AudioFormat, AudioFrame};
    use pretty_assertions::assert_eq;

    /// Spy adapter that records every call so tests can assert the
    /// router dispatched to the right delegate without touching real
    /// audio hardware.
    struct SpyCapture {
        name: &'static str,
        list_calls: Arc<AtomicUsize>,
        start_calls: Arc<AtomicUsize>,
    }

    impl SpyCapture {
        fn new(name: &'static str) -> (Arc<Self>, Arc<AtomicUsize>, Arc<AtomicUsize>) {
            let list_calls = Arc::new(AtomicUsize::new(0));
            let start_calls = Arc::new(AtomicUsize::new(0));
            let arc = Arc::new(Self {
                name,
                list_calls: list_calls.clone(),
                start_calls: start_calls.clone(),
            });
            (arc, list_calls, start_calls)
        }
    }

    #[async_trait]
    impl AudioCapture for SpyCapture {
        async fn list_devices(&self, _source: AudioSource) -> Result<Vec<DeviceInfo>, DomainError> {
            self.list_calls.fetch_add(1, Ordering::SeqCst);
            Ok(vec![DeviceInfo {
                id: format!("{}:default", self.name),
                name: format!("{} default", self.name),
                is_default: true,
            }])
        }

        async fn start(&self, _spec: CaptureSpec) -> Result<Box<dyn AudioStream>, DomainError> {
            self.start_calls.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(SilentStream))
        }
    }

    /// Minimal `AudioStream` impl that yields no frames — just enough
    /// for the router test to receive a `Box<dyn AudioStream>`.
    struct SilentStream;

    #[async_trait]
    impl AudioStream for SilentStream {
        fn format(&self) -> AudioFormat {
            AudioFormat::WHISPER
        }

        async fn next_frame(&mut self) -> Option<AudioFrame> {
            None
        }

        async fn stop(&mut self) -> Result<(), DomainError> {
            Ok(())
        }
    }

    fn router_with(mic: Arc<SpyCapture>, sys: Option<Arc<SpyCapture>>) -> RoutingAudioCapture {
        let mic_dyn: Arc<dyn AudioCapture> = mic;
        let sys_dyn = sys.map(|a| a as Arc<dyn AudioCapture>);
        RoutingAudioCapture::new(mic_dyn, sys_dyn)
    }

    #[tokio::test]
    async fn microphone_request_dispatches_to_mic_adapter() {
        let (mic, mic_list, mic_start) = SpyCapture::new("mic");
        let (sys, sys_list, sys_start) = SpyCapture::new("sys");
        let router = router_with(mic, Some(sys));

        let devices = router.list_devices(AudioSource::Microphone).await.unwrap();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].id, "mic:default");

        router
            .start(CaptureSpec::default_microphone())
            .await
            .unwrap();

        assert_eq!(mic_list.load(Ordering::SeqCst), 1);
        assert_eq!(mic_start.load(Ordering::SeqCst), 1);
        assert_eq!(sys_list.load(Ordering::SeqCst), 0);
        assert_eq!(sys_start.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn system_output_request_dispatches_to_sys_adapter() {
        let (mic, mic_list, mic_start) = SpyCapture::new("mic");
        let (sys, sys_list, sys_start) = SpyCapture::new("sys");
        let router = router_with(mic, Some(sys));

        let devices = router
            .list_devices(AudioSource::SystemOutput)
            .await
            .unwrap();
        assert_eq!(devices[0].id, "sys:default");

        router
            .start(CaptureSpec {
                source: AudioSource::SystemOutput,
                device_id: None,
                preferred_format: AudioFormat::WHISPER,
            })
            .await
            .unwrap();

        assert_eq!(sys_list.load(Ordering::SeqCst), 1);
        assert_eq!(sys_start.load(Ordering::SeqCst), 1);
        assert_eq!(mic_list.load(Ordering::SeqCst), 0);
        assert_eq!(mic_start.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn system_output_without_adapter_returns_unavailable() {
        let (mic, _, _) = SpyCapture::new("mic");
        let router = router_with(mic, None);

        let err = router
            .list_devices(AudioSource::SystemOutput)
            .await
            .expect_err("must reject when no system adapter is wired");
        match err {
            DomainError::AudioDeviceUnavailable(msg) => {
                assert!(
                    msg.contains("system audio") && msg.contains("not available"),
                    "expected message about missing system adapter, got: {msg}"
                );
            }
            other => panic!("unexpected error variant: {other:?}"),
        }

        let start_outcome = router
            .start(CaptureSpec {
                source: AudioSource::SystemOutput,
                device_id: None,
                preferred_format: AudioFormat::WHISPER,
            })
            .await;
        match start_outcome {
            Ok(_) => panic!("must reject start when no system adapter is wired"),
            Err(DomainError::AudioDeviceUnavailable(_)) => {}
            Err(other) => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn default_adapters_constructor_is_total() {
        // The "default" router must always be constructible; on non-macOS
        // targets the system slot is None but Microphone still works.
        let router = RoutingAudioCapture::with_default_adapters();
        let cfg = format!("{router:?}");
        assert!(cfg.contains("RoutingAudioCapture"));
    }
}
