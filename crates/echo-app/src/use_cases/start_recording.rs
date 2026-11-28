//! `start_recording` — first cut.
//!
//! Sprint 0 day 5 ships a deliberately small slice: capture from the
//! microphone for a bounded duration and forward every frame to a
//! caller-provided sink. Diarization, persistence and event publishing
//! land later (Sprint 1+).

use std::sync::Arc;
use std::time::Duration;

use tokio::time::Instant;
use tracing::{debug, error, info, warn};

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
    /// Format negotiated with the device at start.
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
    ///
    /// Frames reach the sink in their own format: a device that
    /// reconnects mid-recording may change it (the report keeps the
    /// format at start).
    ///
    /// # Errors
    ///
    /// Fails if the capture cannot start or the sink rejects a frame;
    /// the sink is finished first, so the audio written until then is
    /// kept (a failure to finish is logged).
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
        let mut current = format;

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
            // A device that reconnected may come back in another format.
            // Frames keep their own format; the sink decides how to store
            // them (a WAV file converts them to its own).
            if frame.format != current {
                info!(
                    old = ?current,
                    new = ?frame.format,
                    "capture format changed mid-recording; the sink receives the new format"
                );
                current = frame.format;
            }
            samples += frame.samples.len() as u64;
            if let Err(e) = self.sink.accept(&frame) {
                warn!(error = %e, "sink rejected frame; stopping recording");
                let _ = stream.stop().await;
                if let Err(finish) = self.sink.finish() {
                    // The rejection is the error the caller gets; still
                    // report that what was written may be unusable too.
                    error!(error = %finish, "finishing the sink after a rejected frame failed");
                }
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

    /// Synthetic stream that yields N frames then ends; with `switch`,
    /// frames after the first `.0` ones carry format `.1`, like a device
    /// that reconnected in another configuration.
    struct FakeStream {
        format: AudioFormat,
        remaining: usize,
        chunk: Vec<f32>,
        switch: Option<(usize, AudioFormat)>,
        sent: usize,
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
            let format = match self.switch {
                Some((after, later)) if self.sent >= after => later,
                _ => self.format,
            };
            self.sent += 1;
            Some(AudioFrame {
                samples: self.chunk.clone(),
                format,
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
        switch: Option<(usize, AudioFormat)>,
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
                switch: self.switch,
                sent: 0,
            }))
        }
    }

    /// What a [`CountingSink`] received.
    #[derive(Default)]
    struct SinkLog {
        frames: u64,
        samples: u64,
        finished: bool,
        formats: Vec<AudioFormat>,
    }

    #[derive(Default, Clone)]
    struct CountingSink(Arc<Mutex<SinkLog>>);
    impl FrameSink for CountingSink {
        fn accept(&mut self, frame: &AudioFrame) -> Result<(), DomainError> {
            let mut g = self.0.lock().unwrap();
            g.frames += 1;
            g.samples += frame.samples.len() as u64;
            g.formats.push(frame.format);
            Ok(())
        }
        fn finish(&mut self) -> Result<(), DomainError> {
            self.0.lock().unwrap().finished = true;
            Ok(())
        }
    }

    #[tokio::test]
    async fn drives_fake_capture_until_stream_ends() {
        let capture = Arc::new(FakeCapture {
            format: AudioFormat::WHISPER,
            chunk: vec![0.0; 1_600], // 100 ms at 16 kHz mono
            frames: 5,
            switch: None,
        });
        let sink = CountingSink::default();
        let report = RecordToSink::new(capture.clone(), sink.clone())
            .execute(CaptureSpec::default_microphone(), Duration::from_secs(10))
            .await
            .unwrap();

        assert_eq!(report.frames, 5);
        assert_eq!(report.samples, 5 * 1_600);
        let g = sink.0.lock().unwrap();
        assert_eq!(g.frames, 5);
        assert_eq!(g.samples, 5 * 1_600);
        assert!(g.finished, "sink.finish was not called");
    }

    #[tokio::test]
    async fn deadline_terminates_long_capture() {
        let capture = Arc::new(FakeCapture {
            format: AudioFormat::WHISPER,
            chunk: vec![0.0; 16],
            frames: usize::MAX,
            switch: None,
        });
        let sink = CountingSink::default();
        let report = RecordToSink::new(capture, sink.clone())
            .execute(CaptureSpec::default_microphone(), Duration::from_millis(50))
            .await
            .unwrap();
        assert!(report.elapsed >= Duration::from_millis(50));
        assert!(sink.0.lock().unwrap().finished);
    }

    /// A device that comes back in another format keeps the recording
    /// going: every frame reaches the sink with its own format, and the
    /// sink (a WAV file converts) decides how to store it.
    #[tokio::test]
    async fn format_change_mid_recording_forwards_frames_in_their_format() {
        let later = AudioFormat {
            sample_rate_hz: 48_000,
            channels: 2,
        };
        let capture = Arc::new(FakeCapture {
            format: AudioFormat::WHISPER,
            chunk: vec![0.0; 1_600],
            frames: 5,
            switch: Some((2, later)),
        });
        let sink = CountingSink::default();
        let report = RecordToSink::new(capture, sink.clone())
            .execute(CaptureSpec::default_microphone(), Duration::from_secs(10))
            .await
            .unwrap();

        assert_eq!(report.frames, 5);
        assert_eq!(report.format, AudioFormat::WHISPER, "format at start");
        let g = sink.0.lock().unwrap();
        assert_eq!(g.frames, 5);
        assert_eq!(
            g.formats,
            vec![
                AudioFormat::WHISPER,
                AudioFormat::WHISPER,
                later,
                later,
                later
            ]
        );
        assert!(g.finished, "sink.finish was not called");
    }

    /// Rejects every frame and fails to finish, recording that it tried.
    #[derive(Default, Clone)]
    struct BrokenSink(Arc<Mutex<bool>>);
    impl FrameSink for BrokenSink {
        fn accept(&mut self, _frame: &AudioFrame) -> Result<(), DomainError> {
            Err(DomainError::Invariant("disk full".into()))
        }
        fn finish(&mut self) -> Result<(), DomainError> {
            *self.0.lock().unwrap() = true;
            Err(DomainError::Invariant("header not written".into()))
        }
    }

    /// The caller gets the rejection, not the finish failure, and the
    /// sink is still asked to finish what it wrote.
    #[tokio::test]
    async fn rejected_frame_reports_the_rejection_after_finishing_the_sink() {
        let capture = Arc::new(FakeCapture {
            format: AudioFormat::WHISPER,
            chunk: vec![0.0; 160],
            frames: 3,
            switch: None,
        });
        let sink = BrokenSink::default();
        let err = RecordToSink::new(capture, sink.clone())
            .execute(CaptureSpec::default_microphone(), Duration::from_secs(10))
            .await
            .unwrap_err();
        assert!(
            matches!(&err, DomainError::Invariant(m) if m == "disk full"),
            "{err:?}"
        );
        assert!(*sink.0.lock().unwrap(), "sink.finish was not called");
    }
}
