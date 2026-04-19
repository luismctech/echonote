//! `start_recording` — first cut.
//!
//! Sprint 0 day 5 ships a deliberately small slice: capture from the
//! microphone for a bounded duration and forward every frame to a
//! caller-provided sink. Diarization, persistence and event publishing
//! land later (Sprint 1+).

use std::sync::Arc;
use std::time::Duration;

use tokio::time::Instant;
use tracing::{debug, info, warn};

use echo_domain::{AudioCapture, AudioFormat, AudioFrame, CaptureSpec, DomainError};

/// Anything that consumes captured frames. Intentionally minimal so the
/// CLI, tests and eventually the storage layer can implement it.
pub trait FrameSink: Send {
    /// Push a frame downstream. Returning an error stops the recording.
    fn accept(&mut self, frame: &AudioFrame) -> Result<(), DomainError>;
    /// Called once after the last frame. Implementations may flush
    /// headers, close files, signal observers, etc.
    fn finish(&mut self) -> Result<(), DomainError> {
        Ok(())
    }
}

/// Outcome of a recording run.
#[derive(Debug, Clone, Copy)]
pub struct RecordingReport {
    /// Format actually negotiated with the device.
    pub format: AudioFormat,
    /// Number of frames the sink accepted.
    pub frames: u64,
    /// Number of PCM samples (per channel × channel count) written.
    pub samples: u64,
    /// Wall-clock duration of the capture.
    pub elapsed: Duration,
}

/// Bounded-duration recording use case.
pub struct RecordToSink<S: FrameSink> {
    capture: Arc<dyn AudioCapture>,
    sink: S,
}

impl<S: FrameSink> RecordToSink<S> {
    /// Wire the use case with concrete adapters.
    pub fn new(capture: Arc<dyn AudioCapture>, sink: S) -> Self {
        Self { capture, sink }
    }

    /// Drives the capture for `duration`, forwarding every frame to the
    /// sink. Returns once the deadline is reached or the stream ends.
    pub async fn execute(
        mut self,
        spec: CaptureSpec,
        duration: Duration,
    ) -> Result<RecordingReport, DomainError> {
        info!(
            duration_ms = duration.as_millis() as u64,
            ?spec,
            "starting recording"
        );

        let mut stream = self.capture.start(spec).await?;
        let format = stream.format();
        let started = Instant::now();
        let deadline = started + duration;

        let mut frames: u64 = 0;
        let mut samples: u64 = 0;

        loop {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            let remaining = deadline - now;
            let frame = match tokio::time::timeout(remaining, stream.next_frame()).await {
                Ok(Some(frame)) => frame,
                Ok(None) => {
                    debug!("capture stream ended before deadline");
                    break;
                }
                Err(_) => {
                    debug!("recording deadline reached");
                    break;
                }
            };
            samples += frame.samples.len() as u64;
            if let Err(e) = self.sink.accept(&frame) {
                warn!(error = %e, "sink rejected frame; stopping recording");
                let _ = stream.stop().await;
                self.sink.finish().ok();
                return Err(e);
            }
            frames += 1;
        }

        stream.stop().await?;
        self.sink.finish()?;

        let report = RecordingReport {
            format,
            frames,
            samples,
            elapsed: started.elapsed(),
        };
        info!(
            frames = report.frames,
            samples = report.samples,
            elapsed_ms = report.elapsed.as_millis() as u64,
            sample_rate_hz = report.format.sample_rate_hz,
            channels = report.format.channels,
            "recording finished"
        );
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    use async_trait::async_trait;
    use echo_domain::{
        AudioCapture, AudioFormat, AudioFrame, AudioSource, AudioStream, CaptureSpec, DeviceInfo,
    };
    use pretty_assertions::assert_eq;

    /// Synthetic stream that yields N frames then ends.
    struct FakeStream {
        format: AudioFormat,
        remaining: usize,
        chunk: Vec<f32>,
    }

    #[async_trait]
    impl AudioStream for FakeStream {
        fn format(&self) -> AudioFormat {
            self.format
        }
        async fn next_frame(&mut self) -> Option<AudioFrame> {
            if self.remaining == 0 {
                return None;
            }
            self.remaining -= 1;
            // Yield to the runtime so timeouts work as expected.
            tokio::task::yield_now().await;
            Some(AudioFrame {
                samples: self.chunk.clone(),
                format: self.format,
                captured_at_ns: 0,
            })
        }
        async fn stop(&mut self) -> Result<(), DomainError> {
            self.remaining = 0;
            Ok(())
        }
    }

    struct FakeCapture {
        format: AudioFormat,
        chunk: Vec<f32>,
        frames: usize,
    }

    #[async_trait]
    impl AudioCapture for FakeCapture {
        async fn list_devices(&self, _source: AudioSource) -> Result<Vec<DeviceInfo>, DomainError> {
            Ok(vec![])
        }
        async fn start(&self, _spec: CaptureSpec) -> Result<Box<dyn AudioStream>, DomainError> {
            Ok(Box::new(FakeStream {
                format: self.format,
                remaining: self.frames,
                chunk: self.chunk.clone(),
            }))
        }
    }

    #[derive(Default, Clone)]
    struct CountingSink(Arc<Mutex<(u64, u64, bool)>>);
    impl FrameSink for CountingSink {
        fn accept(&mut self, frame: &AudioFrame) -> Result<(), DomainError> {
            let mut g = self.0.lock().unwrap();
            g.0 += 1;
            g.1 += frame.samples.len() as u64;
            Ok(())
        }
        fn finish(&mut self) -> Result<(), DomainError> {
            self.0.lock().unwrap().2 = true;
            Ok(())
        }
    }

    #[tokio::test]
    async fn drives_fake_capture_until_stream_ends() {
        let capture = Arc::new(FakeCapture {
            format: AudioFormat::WHISPER,
            chunk: vec![0.0; 1_600], // 100 ms at 16 kHz mono
            frames: 5,
        });
        let sink = CountingSink::default();
        let report = RecordToSink::new(capture.clone(), sink.clone())
            .execute(CaptureSpec::default_microphone(), Duration::from_secs(10))
            .await
            .unwrap();

        assert_eq!(report.frames, 5);
        assert_eq!(report.samples, 5 * 1_600);
        let g = sink.0.lock().unwrap();
        assert_eq!(g.0, 5);
        assert_eq!(g.1, 5 * 1_600);
        assert!(g.2, "sink.finish was not called");
    }

    #[tokio::test]
    async fn deadline_terminates_long_capture() {
        let capture = Arc::new(FakeCapture {
            format: AudioFormat::WHISPER,
            chunk: vec![0.0; 16],
            frames: usize::MAX,
        });
        let sink = CountingSink::default();
        let report = RecordToSink::new(capture, sink.clone())
            .execute(CaptureSpec::default_microphone(), Duration::from_millis(50))
            .await
            .unwrap();
        assert!(report.elapsed >= Duration::from_millis(50));
        assert!(sink.0.lock().unwrap().2);
    }
}
