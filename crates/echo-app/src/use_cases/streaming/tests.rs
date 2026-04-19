//! Pipeline tests with synthetic capture + transcriber.
//!
//! Goals:
//!
//! 1. Started → N × Chunk → Stopped happy path.
//! 2. The silence gate emits Skipped instead of calling the ASR.
//! 3. Sub-chunk audio at EOF still gets flushed as a final chunk.
//! 4. `stop()` cleanly shuts the pipeline down.

use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc, Mutex,
};
use std::time::Duration;

use async_trait::async_trait;
use pretty_assertions::assert_eq;

use echo_domain::{
    AudioCapture, AudioFormat, AudioFrame, AudioSource, AudioStream, CaptureSpec, DeviceInfo,
    DomainError, Resampler, Sample, Segment, SegmentId, StreamingOptions, TranscribeOptions,
    Transcriber, Transcript, TranscriptEvent,
};

use super::{StreamingHandle, StreamingPipeline};

/// Yields N pre-baked frames then EOF.
struct FakeStream {
    format: AudioFormat,
    frames: std::vec::IntoIter<Vec<f32>>,
}

#[async_trait]
impl AudioStream for FakeStream {
    fn format(&self) -> AudioFormat {
        self.format
    }
    async fn next_frame(&mut self) -> Option<AudioFrame> {
        // Yield to the runtime so select! / stop signals interleave.
        tokio::task::yield_now().await;
        let samples = self.frames.next()?;
        Some(AudioFrame {
            samples,
            format: self.format,
            captured_at_ns: 0,
        })
    }
    async fn stop(&mut self) -> Result<(), DomainError> {
        self.frames.by_ref().for_each(drop);
        Ok(())
    }
}

struct FakeCapture {
    format: AudioFormat,
    frames: Mutex<Option<Vec<Vec<f32>>>>,
}

impl FakeCapture {
    fn new(format: AudioFormat, frames: Vec<Vec<f32>>) -> Arc<Self> {
        Arc::new(Self {
            format,
            frames: Mutex::new(Some(frames)),
        })
    }
}

#[async_trait]
impl AudioCapture for FakeCapture {
    async fn list_devices(&self, _source: AudioSource) -> Result<Vec<DeviceInfo>, DomainError> {
        Ok(vec![])
    }
    async fn start(&self, _spec: CaptureSpec) -> Result<Box<dyn AudioStream>, DomainError> {
        let frames = self
            .frames
            .lock()
            .unwrap()
            .take()
            .unwrap_or_default()
            .into_iter();
        Ok(Box::new(FakeStream {
            format: self.format,
            frames,
        }))
    }
}

/// Identity resampler: trusts the caller is already feeding 16 kHz mono.
/// Sufficient for unit tests; production uses RubatoResamplerAdapter.
struct PassThroughResampler;
impl Resampler for PassThroughResampler {
    fn to_whisper(
        &self,
        samples: &[Sample],
        _input: AudioFormat,
    ) -> Result<Vec<Sample>, DomainError> {
        Ok(samples.to_vec())
    }
}

/// Counts ASR calls and returns one fake segment per call.
struct FakeTranscriber {
    calls: AtomicU32,
}

impl FakeTranscriber {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            calls: AtomicU32::new(0),
        })
    }
    fn call_count(&self) -> u32 {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Transcriber for FakeTranscriber {
    async fn transcribe(
        &self,
        samples: &[Sample],
        _options: &TranscribeOptions,
    ) -> Result<Transcript, DomainError> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        let duration_ms = ((samples.len() as u64 * 1_000) / 16_000) as u32;
        Ok(Transcript {
            segments: vec![Segment {
                id: SegmentId::new(),
                start_ms: 0,
                end_ms: duration_ms,
                text: format!("chunk-{n}"),
                speaker_id: None,
                confidence: None,
            }],
            language: Some("en".to_string()),
            duration_ms,
        })
    }
}

fn sine(samples: usize, amplitude: f32) -> Vec<f32> {
    (0..samples)
        .map(|i| {
            let t = i as f32 / 16_000.0;
            amplitude * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
        })
        .collect()
}

async fn drain_until_stopped(handle: &mut StreamingHandle) -> Vec<TranscriptEvent> {
    let mut events = Vec::new();
    while let Some(evt) = handle.next_event().await {
        let stop = matches!(
            evt,
            TranscriptEvent::Stopped { .. } | TranscriptEvent::Failed { .. }
        );
        events.push(evt);
        if stop {
            break;
        }
    }
    events
}

#[tokio::test]
async fn pipeline_emits_started_chunks_and_stopped() {
    let format = AudioFormat::WHISPER;
    // 3 chunks of 1 s each, all loud (above silence threshold).
    let frames = vec![sine(16_000, 0.5), sine(16_000, 0.5), sine(16_000, 0.5)];
    let capture = FakeCapture::new(format, frames);
    let resampler = Arc::new(PassThroughResampler) as Arc<dyn Resampler>;
    let transcriber = FakeTranscriber::new();

    let pipeline = StreamingPipeline::new(capture, resampler, transcriber.clone());
    let mut handle = pipeline
        .start(StreamingOptions {
            chunk_ms: 1_000,
            silence_rms_threshold: 0.005,
            language: None,
        })
        .await
        .unwrap();

    let events = drain_until_stopped(&mut handle).await;

    // 1 Started + 3 Chunk + 1 Stopped
    assert_eq!(events.len(), 5, "events: {events:#?}");
    assert!(matches!(events[0], TranscriptEvent::Started { .. }));
    let chunk_count = events
        .iter()
        .filter(|e| matches!(e, TranscriptEvent::Chunk { .. }))
        .count();
    assert_eq!(chunk_count, 3);
    assert!(matches!(
        events.last().unwrap(),
        TranscriptEvent::Stopped {
            total_segments: 3,
            total_audio_ms: 3_000,
            ..
        }
    ));
    assert_eq!(transcriber.call_count(), 3);
}

#[tokio::test]
async fn silence_gate_emits_skipped_and_does_not_call_asr() {
    let format = AudioFormat::WHISPER;
    // Pattern: voiced, silence, voiced — VAD should drop the middle chunk.
    let frames = vec![sine(16_000, 0.5), vec![0.0; 16_000], sine(16_000, 0.5)];
    let capture = FakeCapture::new(format, frames);
    let resampler = Arc::new(PassThroughResampler) as Arc<dyn Resampler>;
    let transcriber = FakeTranscriber::new();

    let pipeline = StreamingPipeline::new(capture, resampler, transcriber.clone());
    let mut handle = pipeline
        .start(StreamingOptions {
            chunk_ms: 1_000,
            silence_rms_threshold: 0.01,
            language: None,
        })
        .await
        .unwrap();

    let events = drain_until_stopped(&mut handle).await;

    let chunks = events
        .iter()
        .filter(|e| matches!(e, TranscriptEvent::Chunk { .. }))
        .count();
    let skipped = events
        .iter()
        .filter(|e| matches!(e, TranscriptEvent::Skipped { .. }))
        .count();
    assert_eq!(chunks, 2, "events: {events:#?}");
    assert_eq!(skipped, 1, "events: {events:#?}");
    assert_eq!(transcriber.call_count(), 2);
}

#[tokio::test]
async fn final_partial_chunk_is_flushed_on_eof() {
    let format = AudioFormat::WHISPER;
    // 1.5 s of audio with a 1-second chunk → 1 full chunk + 1 final 500 ms flush.
    let frames = vec![sine(16_000, 0.5), sine(8_000, 0.5)];
    let capture = FakeCapture::new(format, frames);
    let resampler = Arc::new(PassThroughResampler) as Arc<dyn Resampler>;
    let transcriber = FakeTranscriber::new();

    let pipeline = StreamingPipeline::new(capture, resampler, transcriber.clone());
    let mut handle = pipeline
        .start(StreamingOptions {
            chunk_ms: 1_000,
            silence_rms_threshold: 0.005,
            language: None,
        })
        .await
        .unwrap();

    let events = drain_until_stopped(&mut handle).await;
    let chunks: Vec<&TranscriptEvent> = events
        .iter()
        .filter(|e| matches!(e, TranscriptEvent::Chunk { .. }))
        .collect();
    assert_eq!(chunks.len(), 2, "events: {events:#?}");
    assert_eq!(transcriber.call_count(), 2);
    if let TranscriptEvent::Stopped { total_audio_ms, .. } = events.last().unwrap() {
        assert_eq!(*total_audio_ms, 1_500);
    } else {
        panic!("expected Stopped, got {events:#?}");
    }
}

#[tokio::test]
async fn stop_drains_cleanly_even_with_pending_frames() {
    let format = AudioFormat::WHISPER;
    // 100 frames of 100 ms each = 10 s of audio. We stop after a few.
    let frames = (0..100).map(|_| sine(1_600, 0.5)).collect();
    let capture = FakeCapture::new(format, frames);
    let resampler = Arc::new(PassThroughResampler) as Arc<dyn Resampler>;
    let transcriber = FakeTranscriber::new();

    let pipeline = StreamingPipeline::new(capture, resampler, transcriber.clone());
    let mut handle = pipeline
        .start(StreamingOptions {
            chunk_ms: 1_000,
            silence_rms_threshold: 0.005,
            language: None,
        })
        .await
        .unwrap();

    // Wait for at least the Started event to arrive.
    let started = handle.next_event().await.unwrap();
    assert!(matches!(started, TranscriptEvent::Started { .. }));

    tokio::time::sleep(Duration::from_millis(50)).await;
    handle.stop().await.unwrap();

    // Drain remaining events; Stopped must show up.
    let mut saw_stopped = false;
    while let Some(evt) = handle.next_event().await {
        if matches!(evt, TranscriptEvent::Stopped { .. }) {
            saw_stopped = true;
        }
    }
    assert!(saw_stopped, "pipeline did not emit Stopped after stop()");
}
