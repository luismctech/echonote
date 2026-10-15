//! Pipeline tests with synthetic capture + transcriber.
//!
//! Goals:
//!
//! 1. Started → N × Chunk → Stopped happy path.
//! 2. The silence gate emits Skipped instead of calling the ASR.
//! 3. Sub-chunk audio at EOF still gets flushed as a final chunk.
//! 4. `stop()` cleanly shuts the pipeline down.

use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc, Mutex,
};
use std::time::Duration;

use async_trait::async_trait;
use pretty_assertions::assert_eq;

use echo_domain::ports::vad::VadFrame;
use echo_domain::{
    AudioCapture, AudioFormat, AudioFrame, AudioSource, AudioStream, CaptureSpec, DeviceInfo,
    Diarizer, DomainError, Resampler, Sample, Segment, SegmentId, Speaker, SpeakerId,
    StreamingOptions, TranscribeOptions, Transcriber, Transcript, TranscriptEvent, Vad, VoiceState,
};

use super::{referenced_speakers, StreamingHandle, StreamingPipeline};
use echo_domain::SpeakerRef;

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

/// In-process fake [`Diarizer`] for wiring tests.
///
/// Maps each chunk to a "speaker bucket" derived from the sign of its
/// first non-zero sample. Same sign ⇒ same `SpeakerId`, opposite sign
/// ⇒ a new speaker (with the next slot). Empty chunks return `None`,
/// matching the contract of a real embedder that refuses to embed
/// silence.
///
/// This bypasses ERes2Net and `echo-diarize` so the application
/// layer's tests do not pull a model crate into their dev-deps.
struct FakeDiarizer {
    /// `bucket key (-1 / 0 / 1)` → assigned speaker (id + slot).
    buckets: HashMap<i8, Speaker>,
    /// Total ids issued so far. Drives slot assignment.
    next_slot: u32,
}

impl FakeDiarizer {
    fn new() -> Self {
        Self {
            buckets: HashMap::new(),
            next_slot: 0,
        }
    }

    fn bucket(samples: &[Sample]) -> Option<i8> {
        samples
            .iter()
            .find(|s| s.abs() > f32::EPSILON)
            .map(|s| if *s >= 0.0 { 1_i8 } else { -1_i8 })
    }
}

#[async_trait]
impl Diarizer for FakeDiarizer {
    fn sample_rate_hz(&self) -> u32 {
        16_000
    }

    async fn assign(&mut self, samples: &[Sample]) -> Result<Option<SpeakerId>, DomainError> {
        let Some(key) = Self::bucket(samples) else {
            return Ok(None);
        };
        let speaker = self.buckets.entry(key).or_insert_with(|| {
            let s = Speaker::anonymous(self.next_slot);
            self.next_slot = self.next_slot.saturating_add(1);
            s
        });
        Ok(Some(speaker.id))
    }

    fn speakers(&self) -> Vec<Speaker> {
        let mut out: Vec<Speaker> = self.buckets.values().cloned().collect();
        out.sort_by_key(|s| s.slot);
        out
    }

    fn rename(&mut self, id: SpeakerId, label: &str) -> Result<bool, DomainError> {
        for s in self.buckets.values_mut() {
            if s.id == id {
                let trimmed = label.trim();
                s.label = if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                };
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn reset(&mut self) {
        self.buckets.clear();
        self.next_slot = 0;
    }
}

/// Every event up to `Stopped`/`Failed`, except the `Progress`
/// telemetry (covered by its own tests), so lifecycle assertions stay
/// readable.
async fn drain_until_stopped(handle: &mut StreamingHandle) -> Vec<TranscriptEvent> {
    drain_with_progress(handle)
        .await
        .into_iter()
        .filter(|e| !matches!(e, TranscriptEvent::Progress { .. }))
        .collect()
}

/// Every event up to `Stopped`/`Failed`, `Progress` included.
async fn drain_with_progress(handle: &mut StreamingHandle) -> Vec<TranscriptEvent> {
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
            ..StreamingOptions::default()
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
            ..StreamingOptions::default()
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
            ..StreamingOptions::default()
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
async fn diarizer_attaches_stable_speaker_ids_across_chunks() {
    let format = AudioFormat::WHISPER;
    // Three 1-second chunks. Speakers alternate by sign of the
    // sine wave: +amplitude → speaker A, -amplitude → speaker B.
    // Order: A, B, A. The fake diarizer must assign the same id to
    // chunks 0 and 2, and a fresh id to chunk 1.
    let frames = vec![sine(16_000, 0.5), sine(16_000, -0.5), sine(16_000, 0.5)];
    let capture = FakeCapture::new(format, frames);
    let resampler = Arc::new(PassThroughResampler) as Arc<dyn Resampler>;
    let transcriber = FakeTranscriber::new();

    let pipeline = StreamingPipeline::new(capture, resampler, transcriber.clone())
        .with_diarizer(Box::new(FakeDiarizer::new()));
    let mut handle = pipeline
        .start(StreamingOptions {
            chunk_ms: 1_000,
            silence_rms_threshold: 0.005,
            language: None,
            ..StreamingOptions::default()
        })
        .await
        .unwrap();

    let events = drain_until_stopped(&mut handle).await;

    let chunks: Vec<&TranscriptEvent> = events
        .iter()
        .filter(|e| matches!(e, TranscriptEvent::Chunk { .. }))
        .collect();
    assert_eq!(chunks.len(), 3, "events: {events:#?}");

    let speakers: Vec<(Option<SpeakerId>, Option<u32>)> = chunks
        .iter()
        .map(|e| match e {
            TranscriptEvent::Chunk {
                speaker_id,
                speaker_slot,
                ..
            } => (*speaker_id, *speaker_slot),
            _ => unreachable!(),
        })
        .collect();

    let (id0, slot0) = speakers[0];
    let (id1, slot1) = speakers[1];
    let (id2, slot2) = speakers[2];

    assert!(id0.is_some(), "chunk 0 must carry a speaker id");
    assert!(id1.is_some(), "chunk 1 must carry a speaker id");
    assert!(id2.is_some(), "chunk 2 must carry a speaker id");
    assert_eq!(id0, id2, "same bucket must collapse to the same id");
    assert_ne!(id0, id1, "different buckets must spawn different ids");
    assert_eq!(slot0, Some(0));
    assert_eq!(slot1, Some(1));
    assert_eq!(slot2, Some(0));

    // Segments inside each chunk must inherit the speaker id so the
    // storage layer can persist (segment, speaker) pairs without
    // re-running the diarizer.
    for (chunk, (id, _)) in chunks.iter().zip(speakers.iter()) {
        if let TranscriptEvent::Chunk { segments, .. } = chunk {
            for seg in segments {
                assert_eq!(seg.speaker_id, *id, "segment speaker mismatch");
            }
        }
    }
}

#[tokio::test]
async fn pipeline_without_diarizer_emits_chunks_without_speaker_fields() {
    // Regression: existing wiring (no diarizer) must keep emitting
    // Chunk events with `speaker_id: None`.
    let format = AudioFormat::WHISPER;
    let frames = vec![sine(16_000, 0.5)];
    let capture = FakeCapture::new(format, frames);
    let resampler = Arc::new(PassThroughResampler) as Arc<dyn Resampler>;
    let transcriber = FakeTranscriber::new();

    let pipeline = StreamingPipeline::new(capture, resampler, transcriber.clone());
    let mut handle = pipeline
        .start(StreamingOptions {
            chunk_ms: 1_000,
            silence_rms_threshold: 0.005,
            language: None,
            ..StreamingOptions::default()
        })
        .await
        .unwrap();

    let events = drain_until_stopped(&mut handle).await;
    let chunk = events
        .iter()
        .find(|e| matches!(e, TranscriptEvent::Chunk { .. }))
        .expect("expected one Chunk");
    if let TranscriptEvent::Chunk {
        speaker_id,
        speaker_slot,
        segments,
        ..
    } = chunk
    {
        assert!(speaker_id.is_none());
        assert!(speaker_slot.is_none());
        assert!(segments.iter().all(|s| s.speaker_id.is_none()));
    }
}

#[tokio::test]
async fn diarizer_with_wrong_sample_rate_fails_to_start() {
    struct WrongRateDiarizer;
    #[async_trait]
    impl Diarizer for WrongRateDiarizer {
        fn sample_rate_hz(&self) -> u32 {
            8_000
        }
        async fn assign(&mut self, _: &[Sample]) -> Result<Option<SpeakerId>, DomainError> {
            Ok(None)
        }
        fn speakers(&self) -> Vec<Speaker> {
            vec![]
        }
        fn rename(&mut self, _: SpeakerId, _: &str) -> Result<bool, DomainError> {
            Ok(false)
        }
        fn reset(&mut self) {}
    }

    let format = AudioFormat::WHISPER;
    let capture = FakeCapture::new(format, vec![sine(16_000, 0.5)]);
    let resampler = Arc::new(PassThroughResampler) as Arc<dyn Resampler>;
    let transcriber = FakeTranscriber::new();
    let pipeline = StreamingPipeline::new(capture, resampler, transcriber)
        .with_diarizer(Box::new(WrongRateDiarizer));

    let result = pipeline.start(StreamingOptions::default()).await;
    assert!(
        result.is_err(),
        "pipeline should refuse to start with a misconfigured diarizer"
    );
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
            ..StreamingOptions::default()
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

// ---------------------------------------------------------------------------
// VAD gate tests
// ---------------------------------------------------------------------------

/// Configurable VAD that returns a fixed `VoiceState` and counts calls.
/// Lets tests pin the gate decision without needing the real Silero
/// model on disk.
#[derive(Debug)]
struct MockVad {
    fixed_state: VoiceState,
    pushes: AtomicU32,
    resets: AtomicU32,
}

impl MockVad {
    fn new(state: VoiceState) -> Self {
        Self {
            fixed_state: state,
            pushes: AtomicU32::new(0),
            resets: AtomicU32::new(0),
        }
    }
    fn push_count(&self) -> u32 {
        self.pushes.load(Ordering::SeqCst)
    }
    fn reset_count(&self) -> u32 {
        self.resets.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Vad for MockVad {
    fn sample_rate_hz(&self) -> u32 {
        16_000
    }
    async fn push(&mut self, _samples: &[Sample]) -> Result<VoiceState, DomainError> {
        self.pushes.fetch_add(1, Ordering::SeqCst);
        Ok(self.fixed_state)
    }
    fn reset(&mut self) {
        self.resets.fetch_add(1, Ordering::SeqCst);
    }
}

#[tokio::test]
async fn vad_silence_skips_chunks_even_when_rms_is_loud() {
    // Three loud chunks (RMS ≈ 0.35) that the energy gate at 0.005
    // would happily forward to Whisper. The neural VAD is configured
    // to call them all `Silence`, so none should reach the ASR.
    let format = AudioFormat::WHISPER;
    let frames = vec![sine(16_000, 0.5), sine(16_000, 0.5), sine(16_000, 0.5)];
    let capture = FakeCapture::new(format, frames);
    let resampler = Arc::new(PassThroughResampler) as Arc<dyn Resampler>;
    let transcriber = FakeTranscriber::new();
    let vad = Box::new(MockVad::new(VoiceState::Silence));

    let pipeline = StreamingPipeline::new(capture, resampler, transcriber.clone()).with_vad(vad);

    let mut handle = pipeline
        .start(StreamingOptions {
            chunk_ms: 1_000,
            silence_rms_threshold: 0.005,
            language: None,
            // Per-chunk VAD gate under test, not speech-aligned cuts.
            speech_aligned_chunks: false,
            ..StreamingOptions::default()
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
    assert_eq!(
        chunks, 0,
        "VAD-Silence chunks must not reach the transcriber: {events:#?}"
    );
    assert_eq!(skipped, 3, "expected one Skipped per chunk: {events:#?}");
    assert_eq!(transcriber.call_count(), 0, "ASR must not be invoked");
}

#[tokio::test]
async fn vad_voiced_forwards_chunks_to_transcriber() {
    let format = AudioFormat::WHISPER;
    let frames = vec![sine(16_000, 0.5), sine(16_000, 0.5)];
    let capture = FakeCapture::new(format, frames);
    let resampler = Arc::new(PassThroughResampler) as Arc<dyn Resampler>;
    let transcriber = FakeTranscriber::new();
    let vad = Box::new(MockVad::new(VoiceState::Voiced));

    let pipeline = StreamingPipeline::new(capture, resampler, transcriber.clone()).with_vad(vad);

    let mut handle = pipeline
        .start(StreamingOptions {
            chunk_ms: 1_000,
            silence_rms_threshold: 0.005,
            language: None,
            // Per-chunk VAD gate under test, not speech-aligned cuts.
            speech_aligned_chunks: false,
            ..StreamingOptions::default()
        })
        .await
        .unwrap();

    let events = drain_until_stopped(&mut handle).await;
    let chunks = events
        .iter()
        .filter(|e| matches!(e, TranscriptEvent::Chunk { .. }))
        .count();
    assert_eq!(chunks, 2, "events: {events:#?}");
    assert_eq!(transcriber.call_count(), 2);
}

#[tokio::test]
async fn vad_bypasses_rms_gate_for_silent_chunks() {
    // Pure-zero chunks — the RMS gate at 0.005 would normally skip
    // them. With a VAD configured the pipeline must instead push
    // every chunk through the VAD (so its temporal model sees the
    // silence too), and the VAD's verdict drives the skip decision.
    // Setting the mock to `Voiced` lets us prove the chunks made it
    // past the (would-be) RMS gate all the way to the transcriber.
    let format = AudioFormat::WHISPER;
    let frames = vec![vec![0.0_f32; 16_000], vec![0.0_f32; 16_000]];
    let capture = FakeCapture::new(format, frames);
    let resampler = Arc::new(PassThroughResampler) as Arc<dyn Resampler>;
    let transcriber = FakeTranscriber::new();
    let vad = Box::new(MockVad::new(VoiceState::Voiced));

    let pipeline = StreamingPipeline::new(capture, resampler, transcriber.clone()).with_vad(vad);
    let mut handle = pipeline
        .start(StreamingOptions {
            chunk_ms: 1_000,
            silence_rms_threshold: 0.005,
            language: None,
            // Per-chunk VAD gate under test, not speech-aligned cuts.
            speech_aligned_chunks: false,
            ..StreamingOptions::default()
        })
        .await
        .unwrap();

    let events = drain_until_stopped(&mut handle).await;
    let chunks = events
        .iter()
        .filter(|e| matches!(e, TranscriptEvent::Chunk { .. }))
        .count();
    assert_eq!(
        chunks, 2,
        "RMS gate must be bypassed when VAD is configured: {events:#?}"
    );
    assert_eq!(transcriber.call_count(), 2);
}

#[tokio::test]
async fn vad_is_reset_on_session_start() {
    // Build a MockVad outside the pipeline so we can check reset()
    // was invoked on session start. We pass it via a small wrapper
    // that holds an Arc and delegates Vad calls through it.
    use std::sync::Arc as StdArc;

    struct ArcVad(StdArc<MockVad>);
    #[async_trait]
    impl Vad for ArcVad {
        fn sample_rate_hz(&self) -> u32 {
            16_000
        }
        async fn push(&mut self, _samples: &[Sample]) -> Result<VoiceState, DomainError> {
            self.0.pushes.fetch_add(1, Ordering::SeqCst);
            Ok(self.0.fixed_state)
        }
        fn reset(&mut self) {
            self.0.resets.fetch_add(1, Ordering::SeqCst);
        }
    }

    let inner = StdArc::new(MockVad::new(VoiceState::Voiced));
    let format = AudioFormat::WHISPER;
    let frames = vec![sine(16_000, 0.5)];
    let capture = FakeCapture::new(format, frames);
    let resampler = Arc::new(PassThroughResampler) as Arc<dyn Resampler>;
    let transcriber = FakeTranscriber::new();

    let pipeline = StreamingPipeline::new(capture, resampler, transcriber)
        .with_vad(Box::new(ArcVad(inner.clone())));
    let mut handle = pipeline
        .start(StreamingOptions {
            chunk_ms: 1_000,
            silence_rms_threshold: 0.005,
            language: None,
            // Per-chunk VAD gate under test, not speech-aligned cuts.
            speech_aligned_chunks: false,
            ..StreamingOptions::default()
        })
        .await
        .unwrap();

    let _ = drain_until_stopped(&mut handle).await;
    assert_eq!(inner.reset_count(), 1, "VAD must be reset on session start");
    assert_eq!(inner.push_count(), 1, "VAD must see the loud chunk");
}

// ---------------------------------------------------------------------------
// Speech-aligned chunking
// ---------------------------------------------------------------------------

/// One second at the test format (16 kHz mono).
const SEC: usize = 16_000;

/// VAD whose k-th push reports a single frame spanning the push with
/// the k-th scripted `(state, probability)`. The fake capture delivers
/// one-second frames, each analysed in one push, so the script reads
/// as one verdict per second.
struct ScriptedVad {
    script: std::vec::IntoIter<(VoiceState, f32)>,
    resets: Arc<AtomicU32>,
}

impl ScriptedVad {
    fn new(script: Vec<(VoiceState, f32)>) -> (Self, Arc<AtomicU32>) {
        let resets = Arc::new(AtomicU32::new(0));
        let vad = Self {
            script: script.into_iter(),
            resets: Arc::clone(&resets),
        };
        (vad, resets)
    }
}

#[async_trait]
impl Vad for ScriptedVad {
    fn sample_rate_hz(&self) -> u32 {
        16_000
    }
    async fn push(&mut self, samples: &[Sample]) -> Result<VoiceState, DomainError> {
        let frames = self.push_frames(samples).await?;
        Ok(frames.last().map_or(VoiceState::Silence, |f| f.state))
    }
    async fn push_frames(&mut self, samples: &[Sample]) -> Result<Vec<VadFrame>, DomainError> {
        let (state, probability) = self.script.next().expect("VAD script exhausted");
        Ok(vec![VadFrame {
            end_sample: samples.len(),
            probability,
            state,
        }])
    }
    fn reset(&mut self) {
        self.resets.fetch_add(1, Ordering::SeqCst);
    }
}

const V: (VoiceState, f32) = (VoiceState::Voiced, 0.9);
const S: (VoiceState, f32) = (VoiceState::Silence, 0.1);

/// Speech-aligned options with drafts disabled, so the chunking tests
/// pin final chunks and exact transcriber call counts; the draft tests
/// opt in through [`draft_options`].
fn aligned_options(min_chunk_ms: u32, max_chunk_ms: u32) -> StreamingOptions {
    StreamingOptions {
        chunk_ms: 1_000,
        silence_rms_threshold: 0.005,
        min_chunk_ms,
        max_chunk_ms,
        draft_interval_ms: 0,
        ..StreamingOptions::default()
    }
}

/// Chunk / skip timeline reduced to `(offset_ms, duration_ms)`.
#[derive(Debug, PartialEq)]
enum Span {
    Chunk(u32, u32),
    Skip(u32, u32),
}

/// Reduce events to spans. For chunks the duration comes from the
/// fake transcriber's single segment, which covers the whole chunk,
/// and its start must equal the chunk offset (session time).
fn spans(events: &[TranscriptEvent]) -> Vec<Span> {
    events
        .iter()
        .filter_map(|e| match e {
            TranscriptEvent::Chunk {
                offset_ms,
                segments,
                ..
            } => {
                assert_eq!(segments.len(), 1);
                assert_eq!(segments[0].start_ms, *offset_ms, "segment in session time");
                Some(Span::Chunk(
                    *offset_ms,
                    segments[0].end_ms - segments[0].start_ms,
                ))
            }
            TranscriptEvent::Skipped {
                offset_ms,
                duration_ms,
                ..
            } => Some(Span::Skip(*offset_ms, *duration_ms)),
            _ => None,
        })
        .collect()
}

/// Stream one second of loud audio per script entry through a
/// speech-aligned session and return every event.
async fn run_aligned(
    script: Vec<(VoiceState, f32)>,
    options: StreamingOptions,
) -> (Vec<TranscriptEvent>, Arc<FakeTranscriber>) {
    let frames = script.iter().map(|_| sine(SEC, 0.5)).collect();
    let capture = FakeCapture::new(AudioFormat::WHISPER, frames);
    let resampler = Arc::new(PassThroughResampler) as Arc<dyn Resampler>;
    let transcriber = FakeTranscriber::new();
    let (vad, _) = ScriptedVad::new(script);
    let pipeline =
        StreamingPipeline::new(capture, resampler, transcriber.clone()).with_vad(Box::new(vad));
    let mut handle = pipeline.start(options).await.unwrap();
    (drain_until_stopped(&mut handle).await, transcriber)
}

fn stopped_totals(events: &[TranscriptEvent]) -> (u32, u32) {
    match events.last() {
        Some(TranscriptEvent::Stopped {
            total_segments,
            total_audio_ms,
            ..
        }) => (*total_segments, *total_audio_ms),
        other => panic!("expected Stopped, got {other:?}"),
    }
}

#[tokio::test]
async fn speech_aligned_chunks_end_at_scripted_pauses() {
    let script = vec![V, V, V, V, S, S, V, V, S, S];
    let (events, transcriber) = run_aligned(script, aligned_options(3_000, 15_000)).await;

    assert_eq!(
        spans(&events),
        vec![
            // First pause at 4 s, past the 3 s minimum.
            Span::Chunk(0, 4_000),
            // Leading silence of the next utterance, minus the pre-roll.
            Span::Skip(4_000, 1_000),
            Span::Chunk(5_000, 3_000),
            // Trailing silence flushed at EOF, never transcribed.
            Span::Skip(8_000, 2_000),
        ]
    );
    assert_eq!(transcriber.call_count(), 2);
    assert_eq!(stopped_totals(&events), (2, 10_000));
}

#[tokio::test]
async fn speech_without_pause_is_force_cut_at_the_quietest_point_before_max() {
    let mut script = vec![V, V, V, V, (VoiceState::Voiced, 0.4), V];
    script.extend(std::iter::repeat_n(V, 7));
    let (events, _) = run_aligned(script, aligned_options(3_000, 6_000)).await;

    assert_eq!(
        spans(&events),
        vec![
            // The 0.4 dip ending at 5 s lies in the lookback before 6 s.
            Span::Chunk(0, 5_000),
            // No dip: the cut lands right at the 6 s maximum.
            Span::Chunk(5_000, 6_000),
            Span::Chunk(11_000, 2_000),
        ]
    );
    assert_eq!(stopped_totals(&events), (3, 13_000));
}

#[tokio::test]
async fn speech_aligned_all_silence_is_skipped() {
    let (events, transcriber) = run_aligned(vec![S, S, S, S], aligned_options(3_000, 15_000)).await;

    assert_eq!(
        spans(&events),
        vec![Span::Skip(0, 2_000), Span::Skip(2_000, 2_000)]
    );
    assert_eq!(transcriber.call_count(), 0);
    assert_eq!(stopped_totals(&events), (0, 4_000));
}

#[tokio::test]
async fn speech_shorter_than_min_is_flushed_when_the_stream_ends() {
    let (events, transcriber) = run_aligned(vec![V, V], aligned_options(3_000, 15_000)).await;

    assert_eq!(spans(&events), vec![Span::Chunk(0, 2_000)]);
    assert_eq!(transcriber.call_count(), 1);
}

#[tokio::test]
async fn speech_aligned_bounds_are_validated_at_start() {
    for (min, max) in [(0, 15_000), (6_000, 3_000)] {
        let (vad, _) = ScriptedVad::new(vec![]);
        let capture = FakeCapture::new(AudioFormat::WHISPER, vec![]);
        let resampler = Arc::new(PassThroughResampler) as Arc<dyn Resampler>;
        let pipeline = StreamingPipeline::new(capture, resampler, FakeTranscriber::new())
            .with_vad(Box::new(vad));
        let result = pipeline.start(aligned_options(min, max)).await;
        assert!(result.is_err(), "min={min} max={max} must be rejected");
    }
}

// ---------------------------------------------------------------------------
// Provisional (draft) text for the open speech chunk
// ---------------------------------------------------------------------------

/// Speech-aligned 4 s / 20 s bounds with a draft every 4 s.
fn draft_options() -> StreamingOptions {
    StreamingOptions {
        draft_interval_ms: 4_000,
        ..aligned_options(4_000, 20_000)
    }
}

/// Two seconds of silence, 12 s of speech, two seconds of silence.
fn draft_script() -> Vec<(VoiceState, f32)> {
    let mut script = vec![S, S];
    script.extend(std::iter::repeat_n(V, 12));
    script.extend([S, S]);
    script
}

/// `(chunk_index, offset_ms, text)` of every `Draft` event.
fn drafts(events: &[TranscriptEvent]) -> Vec<(u32, u32, String)> {
    events
        .iter()
        .filter_map(|e| match e {
            TranscriptEvent::Draft {
                chunk_index,
                offset_ms,
                text,
                ..
            } => Some((*chunk_index, *offset_ms, text.clone())),
            _ => None,
        })
        .collect()
}

/// Events with the drafts removed.
fn without_drafts(events: &[TranscriptEvent]) -> Vec<&TranscriptEvent> {
    events
        .iter()
        .filter(|e| !matches!(e, TranscriptEvent::Draft { .. }))
        .collect()
}

/// Like [`run_aligned`], also saving the session audio; returns the
/// number of samples the sink received.
async fn run_aligned_saving(
    script: Vec<(VoiceState, f32)>,
    options: StreamingOptions,
) -> (Vec<TranscriptEvent>, Arc<FakeTranscriber>, usize) {
    let frames = script.iter().map(|_| sine(SEC, 0.5)).collect();
    let capture = FakeCapture::new(AudioFormat::WHISPER, frames);
    let transcriber = FakeTranscriber::new();
    let (vad, _) = ScriptedVad::new(script);
    let (sink, log) = RecordingSink::new(false);
    let pipeline =
        StreamingPipeline::new(capture, Arc::new(PassThroughResampler), transcriber.clone())
            .with_vad(Box::new(vad))
            .with_audio_sink(Box::new(sink));
    let mut handle = pipeline.start(options).await.unwrap();
    let events = drain_until_stopped(&mut handle).await;
    let saved = log.lock().unwrap().samples.len();
    (events, transcriber, saved)
}

#[tokio::test]
async fn open_speech_chunk_emits_drafts_before_its_final_chunk() {
    let (events, _) = run_aligned(draft_script(), draft_options()).await;

    // The leading second of silence is cut first (one second of
    // pre-roll kept), so the speech chunk is chunk 1 at 1 s; it is
    // drafted after 4, 8 and 12 s of analysed audio.
    assert_eq!(
        spans(&events),
        vec![
            Span::Skip(0, 1_000),
            Span::Chunk(1_000, 13_000),
            Span::Skip(14_000, 2_000),
        ]
    );
    let drafts = drafts(&events);
    assert_eq!(
        drafts.iter().map(|(i, o, _)| (*i, *o)).collect::<Vec<_>>(),
        vec![(1, 1_000), (1, 1_000), (1, 1_000)],
        "events: {events:#?}"
    );
    assert!(drafts.iter().all(|(_, _, text)| !text.is_empty()));
    let last_draft = events
        .iter()
        .rposition(|e| matches!(e, TranscriptEvent::Draft { .. }))
        .unwrap();
    let chunk = events
        .iter()
        .position(|e| matches!(e, TranscriptEvent::Chunk { .. }))
        .unwrap();
    assert!(last_draft < chunk, "drafts precede their final chunk");
}

#[tokio::test]
async fn drafts_never_change_final_chunks_audio_or_totals() {
    let disabled = StreamingOptions {
        draft_interval_ms: 0,
        ..draft_options()
    };
    let (with, with_asr, with_saved) = run_aligned_saving(draft_script(), draft_options()).await;
    let (without, without_asr, without_saved) = run_aligned_saving(draft_script(), disabled).await;

    assert_eq!(drafts(&without), vec![], "interval 0 disables drafts");
    let draft_count = drafts(&with).len() as u32;
    assert_eq!(draft_count, 3);
    assert_eq!(spans(&with), spans(&without));
    assert_eq!(stopped_totals(&with), stopped_totals(&without));
    assert_eq!(
        without_drafts(&with).len(),
        without.len(),
        "drafts add no other event"
    );
    // Drafts are transcribed but never saved as session audio.
    assert_eq!(
        with_asr.call_count(),
        without_asr.call_count() + draft_count
    );
    assert_eq!(with_saved, without_saved);
    assert_eq!(with_saved, draft_script().len() * SEC);
}

#[tokio::test]
async fn fixed_chunks_never_draft() {
    let options = StreamingOptions {
        speech_aligned_chunks: false,
        ..draft_options()
    };
    let (events, _) = run_aligned(vec![V; 10], options).await;
    assert_eq!(drafts(&events), vec![]);
}

/// Item of a [`SteppedStream`]: a frame, or a pause-flag flip applied
/// before the next frame is delivered.
enum Step {
    Frame(Vec<f32>),
    Pause,
    Resume,
}

/// Stream that drives the pipeline's pause flag between frames, so
/// pause/resume lands at deterministic points of the audio.
struct SteppedStream {
    steps: std::vec::IntoIter<Step>,
    paused: Arc<std::sync::OnceLock<Arc<std::sync::atomic::AtomicBool>>>,
}

#[async_trait]
impl AudioStream for SteppedStream {
    fn format(&self) -> AudioFormat {
        AudioFormat::WHISPER
    }
    async fn next_frame(&mut self) -> Option<AudioFrame> {
        // The flag only exists once the pipeline has started.
        while self.paused.get().is_none() {
            tokio::task::yield_now().await;
        }
        let flag = self.paused.get().unwrap();
        loop {
            tokio::task::yield_now().await;
            match self.steps.next()? {
                Step::Frame(samples) => {
                    return Some(AudioFrame {
                        samples,
                        format: AudioFormat::WHISPER,
                        captured_at_ns: 0,
                    })
                }
                Step::Pause => flag.store(true, Ordering::Release),
                Step::Resume => flag.store(false, Ordering::Release),
            }
        }
    }
    async fn stop(&mut self) -> Result<(), DomainError> {
        Ok(())
    }
}

#[tokio::test]
async fn pause_flushes_buffered_speech_and_resume_resets_the_vad() {
    let loud = || Step::Frame(sine(SEC, 0.5));
    let steps = vec![
        loud(),
        loud(),
        Step::Pause,
        loud(), // discarded while paused
        loud(), // discarded while paused
        Step::Resume,
        loud(),
        loud(),
    ];
    let paused_slot = Arc::new(std::sync::OnceLock::new());
    let stream = SteppedStream {
        steps: steps.into_iter(),
        paused: Arc::clone(&paused_slot),
    };
    let (vad, resets) = ScriptedVad::new(vec![V, V, V, V]);
    let capture = FakeCapture::new(AudioFormat::WHISPER, vec![]);
    let resampler = Arc::new(PassThroughResampler) as Arc<dyn Resampler>;
    let transcriber = FakeTranscriber::new();
    let (diarizer, contexts) = RefiningDiarizer::new(Refine::Identity);
    let pipeline = StreamingPipeline::new(capture, resampler, transcriber.clone())
        .with_vad(Box::new(vad))
        .with_diarizer(Box::new(diarizer));
    let mut handle = pipeline
        .start_with_stream(Box::new(stream), aligned_options(3_000, 15_000))
        .await
        .unwrap();
    paused_slot.set(handle.paused_flag()).unwrap();

    let events = drain_until_stopped(&mut handle).await;
    let lifecycle: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            TranscriptEvent::Chunk { .. } => Some("chunk"),
            TranscriptEvent::Paused { .. } => Some("paused"),
            TranscriptEvent::Resumed { .. } => Some("resumed"),
            _ => None,
        })
        .collect();
    assert_eq!(lifecycle, vec!["chunk", "paused", "resumed", "chunk"]);
    // Paused audio is not part of session time.
    assert_eq!(
        spans(&events),
        vec![Span::Chunk(0, 2_000), Span::Chunk(2_000, 2_000)]
    );
    assert_eq!(resets.load(Ordering::SeqCst), 2, "session start + resume");
    assert_eq!(transcriber.call_count(), 2);
    let epochs: Vec<u16> = contexts.lock().unwrap().iter().map(|c| c.epoch).collect();
    assert_eq!(
        epochs,
        vec![0, 1],
        "the diarizer sees the resume as a new run"
    );
}

#[test]
fn referenced_speakers_lists_every_span_speaker_once_with_dominant_included() {
    use echo_domain::{SpeakerRef, SpeakerSpan};

    let a = SpeakerId::new();
    let b = SpeakerId::new();
    let span = |id, slot, start, end| SpeakerSpan {
        speaker_id: id,
        slot,
        start_sample: start,
        end_sample: end,
    };
    let spans = [
        span(a, 0, 0, 800),
        span(b, 1, 800, 900),
        span(a, 0, 900, 1_000),
    ];
    let out = referenced_speakers(&spans, Some((a, 0)));
    assert_eq!(
        out,
        vec![SpeakerRef { id: a, slot: 0 }, SpeakerRef { id: b, slot: 1 }]
    );
    assert!(referenced_speakers(&[], None).is_empty());
}

// ---------- End-of-session re-segmentation (ADR 0001, D3) ----------------

/// What [`RefiningDiarizer::refine_plan`] does at stop.
#[derive(Clone)]
enum Refine {
    /// Relabel every logged span to one fresh speaker (slot 9).
    AllTo(Speaker),
    /// Relabel every span to the speaker it already has.
    Identity,
    Fail,
    Panic,
    /// Sleep past the deadline, ignoring it.
    Hang(Duration),
}

/// [`FakeDiarizer`] that records the contexts it was called with and
/// returns a scripted refine plan.
struct RefiningDiarizer {
    inner: FakeDiarizer,
    refine: Refine,
    contexts: Arc<Mutex<Vec<echo_domain::ChunkContext>>>,
    spans: Vec<(echo_domain::SpanRef, SpeakerId)>,
}

impl RefiningDiarizer {
    fn new(refine: Refine) -> (Self, Arc<Mutex<Vec<echo_domain::ChunkContext>>>) {
        let contexts = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                inner: FakeDiarizer::new(),
                refine,
                contexts: Arc::clone(&contexts),
                spans: Vec::new(),
            },
            contexts,
        )
    }
}

#[async_trait]
impl Diarizer for RefiningDiarizer {
    fn sample_rate_hz(&self) -> u32 {
        16_000
    }
    async fn assign(&mut self, samples: &[Sample]) -> Result<Option<SpeakerId>, DomainError> {
        self.inner.assign(samples).await
    }
    async fn assign_spans_at(
        &mut self,
        samples: &[Sample],
        ctx: echo_domain::ChunkContext,
    ) -> Result<Vec<echo_domain::SpeakerSpan>, DomainError> {
        self.contexts.lock().unwrap().push(ctx);
        let spans = self.assign_spans(samples).await?;
        for (i, s) in spans.iter().enumerate() {
            let at = echo_domain::SpanRef {
                chunk_index: ctx.chunk_index,
                span_index: i as u16,
            };
            self.spans.push((at, s.speaker_id));
        }
        Ok(spans)
    }
    fn speakers(&self) -> Vec<Speaker> {
        self.inner.speakers()
    }
    fn rename(&mut self, id: SpeakerId, label: &str) -> Result<bool, DomainError> {
        self.inner.rename(id, label)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn refine_plan(
        &self,
        budget: echo_domain::RefineBudget,
    ) -> Result<echo_domain::RefineOutcome, DomainError> {
        let plan = |spans: Vec<(echo_domain::SpanRef, SpeakerId)>, created| {
            echo_domain::RefineOutcome::Relabel(echo_domain::SpanRelabelPlan {
                spans,
                created,
                retired: vec![],
            })
        };
        match &self.refine {
            Refine::AllTo(s) => Ok(plan(
                self.spans.iter().map(|(at, _)| (*at, s.id)).collect(),
                vec![s.clone()],
            )),
            Refine::Identity => Ok(plan(self.spans.clone(), vec![])),
            Refine::Fail => Err(DomainError::DiarizationFailed("scripted".into())),
            Refine::Panic => panic!("scripted refine panic"),
            Refine::Hang(d) => {
                std::thread::sleep(*d);
                let _ = budget;
                Ok(plan(vec![], vec![]))
            }
        }
    }
}

async fn run_refining(
    refine: Refine,
    timeout: Duration,
) -> (Vec<TranscriptEvent>, Vec<echo_domain::ChunkContext>) {
    let frames = vec![sine(16_000, 0.5), sine(16_000, -0.5), sine(16_000, 0.5)];
    let capture = FakeCapture::new(AudioFormat::WHISPER, frames);
    let (diarizer, contexts) = RefiningDiarizer::new(refine);
    let pipeline = StreamingPipeline::new(
        capture,
        Arc::new(PassThroughResampler),
        FakeTranscriber::new(),
    )
    .with_diarizer(Box::new(diarizer))
    .with_refine_timeout(timeout);
    let mut handle = pipeline
        .start(StreamingOptions {
            chunk_ms: 1_000,
            silence_rms_threshold: 0.005,
            ..StreamingOptions::default()
        })
        .await
        .unwrap();
    let events = drain_until_stopped(&mut handle).await;
    let contexts = contexts.lock().unwrap().clone();
    (events, contexts)
}

fn reassigned(events: &[TranscriptEvent]) -> Vec<&TranscriptEvent> {
    events
        .iter()
        .filter(|e| matches!(e, TranscriptEvent::SpeakersReassigned { .. }))
        .collect()
}

#[tokio::test]
async fn diarizer_receives_the_session_context_of_each_chunk() {
    let (events, contexts) = run_refining(Refine::Identity, Duration::from_secs(5)).await;
    let chunks: Vec<(u32, u32)> = events
        .iter()
        .filter_map(|e| match e {
            TranscriptEvent::Chunk {
                chunk_index,
                offset_ms,
                ..
            } => Some((*chunk_index, *offset_ms)),
            _ => None,
        })
        .collect();
    let seen: Vec<(u32, u32, u16)> = contexts
        .iter()
        .map(|c| (c.chunk_index, c.offset_ms, c.epoch))
        .collect();
    assert_eq!(chunks, vec![(0, 0), (1, 1_000), (2, 2_000)]);
    assert_eq!(seen, vec![(0, 0, 0), (1, 1_000, 0), (2, 2_000, 0)]);
}

#[tokio::test]
async fn reassignment_covers_every_emitted_segment_between_last_chunk_and_stopped() {
    let fresh = Speaker::anonymous(9);
    let (events, _) = run_refining(Refine::AllTo(fresh.clone()), Duration::from_secs(5)).await;

    let emitted: Vec<SegmentId> = events
        .iter()
        .filter_map(|e| match e {
            TranscriptEvent::Chunk { segments, .. } => Some(segments.iter().map(|s| s.id)),
            _ => None,
        })
        .flatten()
        .collect();
    let at = |pred: fn(&TranscriptEvent) -> bool| events.iter().rposition(pred).unwrap();
    let last_chunk = at(|e| matches!(e, TranscriptEvent::Chunk { .. }));
    let reassign_at = at(|e| matches!(e, TranscriptEvent::SpeakersReassigned { .. }));
    let stopped_at = at(|e| matches!(e, TranscriptEvent::Stopped { .. }));
    assert!(
        last_chunk < reassign_at && reassign_at < stopped_at,
        "{events:#?}"
    );

    let TranscriptEvent::SpeakersReassigned {
        created,
        retired,
        segments,
        chunks,
        ..
    } = &events[reassign_at]
    else {
        unreachable!()
    };
    let moved: Vec<SegmentId> = segments.iter().map(|s| s.segment_id).collect();
    assert_eq!(moved, emitted, "the plan covers every emitted segment");
    assert!(segments.iter().all(|s| s.speaker_id == Some(fresh.id)));
    assert_eq!(
        created,
        &vec![SpeakerRef {
            id: fresh.id,
            slot: 9
        }]
    );
    assert_eq!(retired.len(), 2, "both online speakers lost every segment");
    assert!(retired
        .iter()
        .all(|r| r.majority_recipient == Some(fresh.id)));
    assert_eq!(chunks.len(), 3);
    assert!(chunks.iter().all(|c| c.speaker_slot == Some(9)));
}

#[tokio::test]
async fn plan_that_moves_nothing_emits_no_reassignment() {
    let (events, _) = run_refining(Refine::Identity, Duration::from_secs(5)).await;
    assert!(reassigned(&events).is_empty());
    assert!(matches!(
        events.last(),
        Some(TranscriptEvent::Stopped { .. })
    ));
}

#[tokio::test]
async fn failed_or_panicking_refine_keeps_the_online_result() {
    for refine in [Refine::Fail, Refine::Panic] {
        let (events, _) = run_refining(refine, Duration::from_secs(5)).await;
        assert!(reassigned(&events).is_empty());
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, TranscriptEvent::Chunk { .. }))
                .count(),
            3
        );
        assert!(matches!(
            events.last(),
            Some(TranscriptEvent::Stopped { .. })
        ));
    }
}

#[tokio::test]
async fn refine_ignoring_its_deadline_still_lets_the_session_stop() {
    let started = std::time::Instant::now();
    let (events, _) = run_refining(
        Refine::Hang(Duration::from_secs(3)),
        Duration::from_millis(100),
    )
    .await;
    assert!(reassigned(&events).is_empty());
    assert!(matches!(
        events.last(),
        Some(TranscriptEvent::Stopped { .. })
    ));
    assert!(
        started.elapsed() < Duration::from_millis(2_500),
        "Stopped waited for the hung refine: {:?}",
        started.elapsed()
    );
}

#[tokio::test]
async fn an_unbounded_refine_timeout_does_not_overflow_and_still_stops() {
    let fresh = Speaker::anonymous(9);
    let (events, _) = run_refining(Refine::AllTo(fresh), Duration::MAX).await;
    assert_eq!(reassigned(&events).len(), 1);
    assert!(matches!(
        events.last(),
        Some(TranscriptEvent::Stopped { .. })
    ));
}

/// Records what the pipeline writes to its session audio sink.
#[derive(Default)]
struct SinkLog {
    samples: Vec<f32>,
    formats: Vec<AudioFormat>,
    accepts: u32,
    finishes: u32,
}

/// [`crate::FrameSink`] that appends to a shared [`SinkLog`]; with
/// `fail` set, every write errors.
struct RecordingSink {
    log: Arc<Mutex<SinkLog>>,
    fail: bool,
}

impl RecordingSink {
    fn new(fail: bool) -> (Self, Arc<Mutex<SinkLog>>) {
        let log = Arc::new(Mutex::new(SinkLog::default()));
        (
            Self {
                log: Arc::clone(&log),
                fail,
            },
            log,
        )
    }
}

impl crate::FrameSink for RecordingSink {
    fn accept(&mut self, frame: &AudioFrame) -> Result<(), DomainError> {
        let mut log = self.log.lock().unwrap();
        log.accepts += 1;
        if self.fail {
            return Err(DomainError::AudioDeviceUnavailable("disk full".into()));
        }
        log.samples.extend_from_slice(&frame.samples);
        log.formats.push(frame.format);
        Ok(())
    }
    fn finish(&mut self) -> Result<(), DomainError> {
        self.log.lock().unwrap().finishes += 1;
        Ok(())
    }
}

/// Stereo → mono by keeping the left channel; enough to tell resampled
/// audio from device audio.
struct LeftChannelResampler;
impl Resampler for LeftChannelResampler {
    fn to_whisper(
        &self,
        samples: &[Sample],
        input: AudioFormat,
    ) -> Result<Vec<Sample>, DomainError> {
        Ok(samples
            .iter()
            .step_by(usize::from(input.channels))
            .copied()
            .collect())
    }
}

#[tokio::test]
async fn audio_sink_receives_every_emitted_chunk_as_16k_mono() {
    let stereo = AudioFormat {
        sample_rate_hz: 16_000,
        channels: 2,
    };
    // Loud, silent (skipped by the RMS gate), loud: 1 s each, stereo.
    let frames = vec![sine(32_000, 0.5), vec![0.0; 32_000], sine(32_000, 0.5)];
    let capture = FakeCapture::new(stereo, frames);
    let transcriber = FakeTranscriber::new();
    let (sink, log) = RecordingSink::new(false);
    let pipeline =
        StreamingPipeline::new(capture, Arc::new(LeftChannelResampler), transcriber.clone())
            .with_audio_sink(Box::new(sink));
    let mut handle = pipeline
        .start(StreamingOptions {
            chunk_ms: 1_000,
            silence_rms_threshold: 0.01,
            ..StreamingOptions::default()
        })
        .await
        .unwrap();

    let events = drain_until_stopped(&mut handle).await;
    assert!(matches!(
        events.last(),
        Some(TranscriptEvent::Stopped { .. })
    ));

    let log = log.lock().unwrap();
    // Skipped silence is saved too, so file time equals session time.
    assert_eq!(log.samples.len(), 3 * 16_000);
    assert!(log.samples[16_000..32_000].iter().all(|&s| s == 0.0));
    assert!(log.formats.iter().all(|&f| f == AudioFormat::WHISPER));
    assert_eq!(log.finishes, 1);
    assert_eq!(transcriber.call_count(), 2);
}

#[tokio::test]
async fn failing_audio_sink_never_stops_transcription() {
    let frames = vec![sine(16_000, 0.5), sine(16_000, 0.5), sine(16_000, 0.5)];
    let capture = FakeCapture::new(AudioFormat::WHISPER, frames);
    let transcriber = FakeTranscriber::new();
    let (sink, log) = RecordingSink::new(true);
    let pipeline =
        StreamingPipeline::new(capture, Arc::new(PassThroughResampler), transcriber.clone())
            .with_audio_sink(Box::new(sink));
    let mut handle = pipeline
        .start(StreamingOptions {
            chunk_ms: 1_000,
            silence_rms_threshold: 0.005,
            ..StreamingOptions::default()
        })
        .await
        .unwrap();

    let events = drain_until_stopped(&mut handle).await;
    assert!(matches!(
        events.last(),
        Some(TranscriptEvent::Stopped {
            total_segments: 3,
            ..
        })
    ));
    assert_eq!(transcriber.call_count(), 3);
    let log = log.lock().unwrap();
    assert_eq!(
        log.accepts, 1,
        "the sink is dropped after its first failure"
    );
}

#[tokio::test]
async fn audio_sink_skips_paused_audio() {
    let loud = || Step::Frame(sine(SEC, 0.5));
    let quiet = || Step::Frame(vec![0.25; SEC]);
    let steps = vec![
        loud(),
        Step::Pause,
        quiet(), // discarded while paused
        Step::Resume,
        loud(),
    ];
    let paused_slot = Arc::new(std::sync::OnceLock::new());
    let stream = SteppedStream {
        steps: steps.into_iter(),
        paused: Arc::clone(&paused_slot),
    };
    let capture = FakeCapture::new(AudioFormat::WHISPER, vec![]);
    let (sink, log) = RecordingSink::new(false);
    let pipeline = StreamingPipeline::new(
        capture,
        Arc::new(PassThroughResampler),
        FakeTranscriber::new(),
    )
    .with_audio_sink(Box::new(sink));
    let mut handle = pipeline
        .start_with_stream(
            Box::new(stream),
            StreamingOptions {
                chunk_ms: 1_000,
                silence_rms_threshold: 0.005,
                ..StreamingOptions::default()
            },
        )
        .await
        .unwrap();
    paused_slot.set(handle.paused_flag()).unwrap();

    drain_until_stopped(&mut handle).await;
    let log = log.lock().unwrap();
    assert_eq!(log.samples.len(), 2 * SEC);
    assert!(log.samples.iter().all(|&s| s != 0.25));
}

/// 440 Hz bursts at `voiced_rms` (200 ms on, 100 ms off, silent gaps),
/// cut into 100 ms capture frames: speech-like dynamics at a level a
/// quiet microphone delivers (-50 dBFS).
fn quiet_speech_frames(seconds: usize, voiced_rms: f32) -> Vec<Vec<f32>> {
    let amplitude = voiced_rms * std::f32::consts::SQRT_2;
    let samples: Vec<f32> = (0..seconds * 16_000)
        .map(|i| {
            if i % 4_800 < 3_200 {
                amplitude * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin()
            } else {
                0.0
            }
        })
        .collect();
    samples.chunks(1_600).map(<[f32]>::to_vec).collect()
}

async fn run_quiet_speech(auto_gain: bool) -> (usize, usize) {
    let capture = FakeCapture::new(AudioFormat::WHISPER, quiet_speech_frames(6, 0.003));
    let resampler = Arc::new(PassThroughResampler) as Arc<dyn Resampler>;
    let pipeline = StreamingPipeline::new(capture, resampler, FakeTranscriber::new());
    let mut handle = pipeline
        .start(StreamingOptions {
            chunk_ms: 1_000,
            auto_gain,
            ..StreamingOptions::default()
        })
        .await
        .unwrap();
    let events = drain_until_stopped(&mut handle).await;
    let count = |f: fn(&TranscriptEvent) -> bool| events.iter().filter(|e| f(e)).count();
    (
        count(|e| matches!(e, TranscriptEvent::Chunk { .. })),
        count(|e| matches!(e, TranscriptEvent::Skipped { .. })),
    )
}

#[tokio::test]
async fn auto_gain_lets_quiet_speech_past_the_rms_gate() {
    // Default RMS gate (0.02) against 0.003 RMS speech: only the first
    // second, while the gain is still rising, may be skipped.
    let (chunks, skipped) = run_quiet_speech(true).await;
    assert!(chunks >= 5, "chunks {chunks}, skipped {skipped}");
    assert!(skipped <= 1, "chunks {chunks}, skipped {skipped}");
}

#[tokio::test]
async fn without_auto_gain_quiet_speech_is_skipped_as_before() {
    let (chunks, skipped) = run_quiet_speech(false).await;
    assert_eq!((chunks, skipped), (0, 6));
}

// ---------------------------------------------------------------------------
// Capture decoupled from transcription, processing lag
// ---------------------------------------------------------------------------

/// Models a real capture device: a producer pushes one frame per
/// `cadence` into a small bounded queue with `try_send` and drops the
/// frame when the queue is full, like the cpal adapters do.
struct HotStream {
    rx: tokio::sync::mpsc::Receiver<Vec<f32>>,
    producer: tokio::task::JoinHandle<()>,
}

/// Device-side queue capacity of a [`HotStream`]: small, so a consumer
/// that stops reading for a few frames loses audio.
const HOT_STREAM_CAPACITY: usize = 2;

impl HotStream {
    /// Produce `frames` frames of `frame_samples` loud samples, one per
    /// `cadence`. With `end_when_done` the stream then ends; otherwise
    /// it stays open (silent) until stopped. Returns the stream and
    /// the number of frames produced so far.
    fn new(
        frames: usize,
        frame_samples: usize,
        cadence: Duration,
        end_when_done: bool,
    ) -> (Self, Arc<AtomicU32>) {
        let (tx, rx) = tokio::sync::mpsc::channel(HOT_STREAM_CAPACITY);
        let produced = Arc::new(AtomicU32::new(0));
        let counter = Arc::clone(&produced);
        let producer = tokio::spawn(async move {
            for _ in 0..frames {
                tokio::time::sleep(cadence).await;
                // A full queue drops the frame, exactly like the device.
                let _ = tx.try_send(sine(frame_samples, 0.5));
                counter.fetch_add(1, Ordering::SeqCst);
            }
            if !end_when_done {
                std::future::pending::<()>().await;
            }
        });
        (Self { rx, producer }, produced)
    }
}

#[async_trait]
impl AudioStream for HotStream {
    fn format(&self) -> AudioFormat {
        AudioFormat::WHISPER
    }
    async fn next_frame(&mut self) -> Option<AudioFrame> {
        let samples = self.rx.recv().await?;
        Some(AudioFrame {
            samples,
            format: AudioFormat::WHISPER,
            captured_at_ns: 0,
        })
    }
    async fn stop(&mut self) -> Result<(), DomainError> {
        self.producer.abort();
        self.rx.close();
        Ok(())
    }
}

/// Transcriber that takes `delay` (real time) per call.
struct SlowTranscriber {
    delay: Duration,
    inner: Arc<FakeTranscriber>,
}

#[async_trait]
impl Transcriber for SlowTranscriber {
    async fn transcribe(
        &self,
        samples: &[Sample],
        options: &TranscribeOptions,
    ) -> Result<Transcript, DomainError> {
        tokio::time::sleep(self.delay).await;
        self.inner.transcribe(samples, options).await
    }
}

/// 50 ms frames, one every 50 ms (real time).
const HOT_FRAME_MS: u64 = 50;
const HOT_FRAME_SAMPLES: usize = 800;
const HOT_FRAMES: usize = 16;

fn hot_options() -> StreamingOptions {
    StreamingOptions {
        chunk_ms: 100,
        silence_rms_threshold: 0.005,
        ..StreamingOptions::default()
    }
}

/// Start a pipeline over a fresh [`HotStream`] with a transcriber that
/// takes `asr_delay` per 100 ms chunk.
async fn start_hot(
    asr_delay: Duration,
    end_when_done: bool,
) -> (StreamingHandle, Arc<AtomicU32>, Arc<FakeTranscriber>) {
    let (stream, produced) = HotStream::new(
        HOT_FRAMES,
        HOT_FRAME_SAMPLES,
        Duration::from_millis(HOT_FRAME_MS),
        end_when_done,
    );
    let inner = FakeTranscriber::new();
    let transcriber = Arc::new(SlowTranscriber {
        delay: asr_delay,
        inner: Arc::clone(&inner),
    });
    let capture = FakeCapture::new(AudioFormat::WHISPER, vec![]);
    let pipeline = StreamingPipeline::new(capture, Arc::new(PassThroughResampler), transcriber);
    let handle = pipeline
        .start_with_stream(Box::new(stream), hot_options())
        .await
        .unwrap();
    (handle, produced, inner)
}

const FED_MS: u32 = (HOT_FRAMES as u64 * HOT_FRAME_MS) as u32;

#[tokio::test]
async fn transcriber_slower_than_real_time_loses_no_captured_audio() {
    // 200 ms of ASR per 100 ms chunk: twice slower than real time.
    let (mut handle, produced, inner) = start_hot(Duration::from_millis(200), true).await;
    let events = drain_until_stopped(&mut handle).await;
    assert_eq!(produced.load(Ordering::SeqCst), HOT_FRAMES as u32);
    assert_eq!(stopped_totals(&events).1, FED_MS, "events: {events:#?}");
    assert_eq!(inner.call_count(), FED_MS / 100);
}

#[tokio::test]
async fn stop_processes_the_audio_still_queued() {
    let (mut handle, produced, inner) = start_hot(Duration::from_millis(200), false).await;
    while produced.load(Ordering::SeqCst) < HOT_FRAMES as u32 {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    // Let the capture reader pick up the last frame, then stop while
    // the slow transcriber still has most of the session queued.
    tokio::time::sleep(Duration::from_millis(30)).await;
    assert!(inner.call_count() < FED_MS / 100, "nothing was queued");
    handle.stop().await.unwrap();
    let events = drain_until_stopped(&mut handle).await;
    assert_eq!(stopped_totals(&events).1, FED_MS, "events: {events:#?}");
    assert_eq!(inner.call_count(), FED_MS / 100);
}

/// `(lag_ms, backlog_ms)` of every `Progress` event.
fn progress(events: &[TranscriptEvent]) -> Vec<(u32, u32)> {
    events
        .iter()
        .filter_map(|e| match e {
            TranscriptEvent::Progress {
                lag_ms, backlog_ms, ..
            } => Some((*lag_ms, *backlog_ms)),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn progress_reports_growing_lag_when_the_transcriber_is_slow() {
    let (mut handle, _, _) = start_hot(Duration::from_millis(200), true).await;
    let events = drain_with_progress(&mut handle).await;
    let progress = progress(&events);
    // One report per emitted chunk.
    assert_eq!(progress.len(), (FED_MS / 100) as usize, "{progress:?}");
    let max_lag = progress.iter().map(|p| p.0).max().unwrap();
    // Chunk k lands at ~100 + 200·k ms holding 100·k ms of audio, so
    // the lag peaks near 500 ms when capture ends.
    assert!(max_lag >= 300, "lag never grew: {progress:?}");
    assert!(
        progress.iter().any(|p| p.1 > 0),
        "audio queued behind the transcriber must show as backlog: {progress:?}"
    );
}

#[tokio::test]
async fn progress_reports_no_lag_when_the_transcriber_keeps_up() {
    let (mut handle, _, _) = start_hot(Duration::ZERO, true).await;
    let events = drain_with_progress(&mut handle).await;
    let progress = progress(&events);
    assert_eq!(progress.len(), (FED_MS / 100) as usize, "{progress:?}");
    // Each chunk is emitted as soon as its second frame arrives; only
    // timer jitter adds lag.
    assert!(
        progress.iter().all(|p| p.0 <= 250),
        "lag without a slow transcriber: {progress:?}"
    );
}

#[tokio::test]
async fn pause_applies_to_frames_by_capture_time_not_dequeue_time() {
    // The stream races ahead of a slow transcriber, so the pipeline
    // dequeues the paused frames after the flag is already cleared.
    let loud = || Step::Frame(sine(SEC, 0.5));
    let steps = vec![
        loud(),
        loud(),
        Step::Pause,
        Step::Frame(vec![0.25; SEC]), // captured while paused
        Step::Frame(vec![0.25; SEC]), // captured while paused
        Step::Resume,
        loud(),
        loud(),
    ];
    let paused_slot = Arc::new(std::sync::OnceLock::new());
    let stream = SteppedStream {
        steps: steps.into_iter(),
        paused: Arc::clone(&paused_slot),
    };
    let inner = FakeTranscriber::new();
    let transcriber = Arc::new(SlowTranscriber {
        delay: Duration::from_millis(100),
        inner: Arc::clone(&inner),
    });
    let capture = FakeCapture::new(AudioFormat::WHISPER, vec![]);
    let (sink, log) = RecordingSink::new(false);
    let pipeline = StreamingPipeline::new(capture, Arc::new(PassThroughResampler), transcriber)
        .with_audio_sink(Box::new(sink));
    let mut handle = pipeline
        .start_with_stream(
            Box::new(stream),
            StreamingOptions {
                chunk_ms: 1_000,
                silence_rms_threshold: 0.005,
                auto_gain: false,
                ..StreamingOptions::default()
            },
        )
        .await
        .unwrap();
    paused_slot.set(handle.paused_flag()).unwrap();

    let events = drain_until_stopped(&mut handle).await;
    let lifecycle: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            TranscriptEvent::Chunk { .. } => Some("chunk"),
            TranscriptEvent::Paused { .. } => Some("paused"),
            TranscriptEvent::Resumed { .. } => Some("resumed"),
            _ => None,
        })
        .collect();
    assert_eq!(
        lifecycle,
        vec!["chunk", "chunk", "paused", "resumed", "chunk", "chunk"]
    );
    assert_eq!(stopped_totals(&events).1, 4_000);
    assert_eq!(inner.call_count(), 4);
    let log = log.lock().unwrap();
    assert!(
        log.samples.iter().all(|&s| s != 0.25),
        "paused audio was saved"
    );
}

#[tokio::test]
async fn stop_without_reading_events_does_not_deadlock_on_a_large_backlog() {
    // 200 chunks queued: ~400 events (chunk + progress) emitted while
    // the caller awaits `stop()` and reads nothing, as the shell does.
    let frames = (0..200).map(|_| sine(1_600, 0.5)).collect();
    let capture = FakeCapture::new(AudioFormat::WHISPER, frames);
    let pipeline = StreamingPipeline::new(
        capture,
        Arc::new(PassThroughResampler),
        FakeTranscriber::new(),
    );
    let mut handle = pipeline.start(hot_options()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;
    tokio::time::timeout(Duration::from_secs(10), handle.stop())
        .await
        .expect("stop() deadlocked on a full event channel")
        .unwrap();
    let events = drain_until_stopped(&mut handle).await;
    assert_eq!(stopped_totals(&events).1, 20_000);
}
