//! Integration tests for the streaming pipeline.
//!
//! Unlike the unit tests in `streaming/tests.rs` (which use
//! `PassThroughResampler` and no persistence), these wire **real**
//! adapters from sibling crates:
//!
//! - [`RubatoResamplerAdapter`] for sample-rate conversion
//! - [`SqliteMeetingStore`] for in-memory persistence
//!
//! The transcriber is still a fake (loading a Whisper model in CI is
//! not practical), but crossing the crate boundary with the production
//! resampler and store exercises the actual data flow.

use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc, Mutex,
};

use async_trait::async_trait;
use pretty_assertions::assert_eq;

use echo_audio::RubatoResamplerAdapter;
use echo_domain::ports::vad::VadFrame;
use echo_domain::{
    AudioCapture, AudioFormat, AudioFrame, AudioSource, AudioStream, CaptureSpec, CreateMeeting,
    DeviceInfo, DomainError, MeetingId, MeetingStore, Resampler, Sample, Segment, SegmentId,
    StreamingOptions, TranscribeOptions, Transcriber, Transcript, TranscriptEvent, Vad, VoiceState,
};
use echo_storage::SqliteMeetingStore;

use echo_app::StreamingPipeline;

// ---------------------------------------------------------------------------
// Test doubles
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn sine_at(sample_rate: u32, samples: usize, amplitude: f32) -> Vec<f32> {
    (0..samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            amplitude * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
        })
        .collect()
}

async fn drain_until_stopped(handle: &mut echo_app::StreamingHandle) -> Vec<TranscriptEvent> {
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// End-to-end: 48 kHz capture → RubatoResampler → FakeTranscriber.
///
/// Verifies the production resampler correctly converts non-16 kHz
/// audio before it reaches the transcriber, and the pipeline emits the
/// expected event sequence.
#[tokio::test]
async fn pipeline_with_real_resampler_48khz() {
    let input_rate = 48_000_u32;
    let format = AudioFormat {
        sample_rate_hz: input_rate,
        channels: 1,
    };
    // 1 second of 48 kHz sine wave
    let frames = vec![sine_at(input_rate, input_rate as usize, 0.5)];
    let capture = FakeCapture::new(format, frames);
    let resampler = Arc::new(RubatoResamplerAdapter::new()) as Arc<dyn Resampler>;
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

    assert!(matches!(events[0], TranscriptEvent::Started { .. }));
    let chunks: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, TranscriptEvent::Chunk { .. }))
        .collect();
    assert_eq!(chunks.len(), 1);
    assert_eq!(transcriber.call_count(), 1);
    assert!(matches!(
        events.last().unwrap(),
        TranscriptEvent::Stopped { .. }
    ));
}

/// VAD that reports one frame per push from a script of states.
struct ScriptedVad(std::vec::IntoIter<VoiceState>);

#[async_trait]
impl Vad for ScriptedVad {
    fn sample_rate_hz(&self) -> u32 {
        16_000
    }
    async fn push(&mut self, _samples: &[Sample]) -> Result<VoiceState, DomainError> {
        unreachable!("speech-aligned chunking uses push_frames")
    }
    async fn push_frames(&mut self, samples: &[Sample]) -> Result<Vec<VadFrame>, DomainError> {
        let state = self.0.next().expect("VAD script exhausted");
        let probability = if state == VoiceState::Voiced {
            0.9
        } else {
            0.1
        };
        Ok(vec![VadFrame {
            end_sample: samples.len(),
            probability,
            state,
        }])
    }
    fn reset(&mut self) {}
}

/// Speech-aligned cuts found on the 16 kHz VAD stream must land on the
/// matching 48 kHz stereo device frames: chunk offsets and durations
/// follow real sample counts through the production resampler.
#[tokio::test]
async fn speech_aligned_cuts_map_back_to_48khz_stereo_frames() {
    let format = AudioFormat {
        sample_rate_hz: 48_000,
        channels: 2,
    };
    // 6 s in 100 ms stereo frames; the pipeline analyses 200 ms blocks,
    // so the script has one verdict per 200 ms: 4 s speech, 2 s silence.
    let frames = (0..60)
        .map(|_| {
            sine_at(48_000, 4_800, 0.5)
                .into_iter()
                .flat_map(|s| [s, s])
                .collect()
        })
        .collect();
    let mut script = vec![VoiceState::Voiced; 20];
    script.extend([VoiceState::Silence; 10]);

    let capture = FakeCapture::new(format, frames);
    let resampler = Arc::new(RubatoResamplerAdapter::new()) as Arc<dyn Resampler>;
    let transcriber = FakeTranscriber::new();
    let pipeline = StreamingPipeline::new(capture, resampler, transcriber.clone())
        .with_vad(Box::new(ScriptedVad(script.into_iter())));
    let mut handle = pipeline
        .start(StreamingOptions {
            min_chunk_ms: 3_000,
            max_chunk_ms: 15_000,
            ..StreamingOptions::default()
        })
        .await
        .unwrap();

    let events = drain_until_stopped(&mut handle).await;
    let timeline: Vec<(&str, u32, u32)> = events
        .iter()
        .filter_map(|e| match e {
            TranscriptEvent::Chunk {
                offset_ms,
                segments,
                ..
            } => Some((
                "chunk",
                *offset_ms,
                segments[0].end_ms - segments[0].start_ms,
            )),
            TranscriptEvent::Skipped {
                offset_ms,
                duration_ms,
                ..
            } => Some(("skip", *offset_ms, *duration_ms)),
            _ => None,
        })
        .collect();
    assert_eq!(timeline, vec![("chunk", 0, 4_000), ("skip", 4_000, 2_000)]);
    assert_eq!(transcriber.call_count(), 1);
    assert!(matches!(
        events.last().unwrap(),
        TranscriptEvent::Stopped {
            total_audio_ms: 6_000,
            ..
        }
    ));
}

/// Pipeline events can be persisted to an in-memory SQLite store and
/// queried back.
#[tokio::test]
async fn events_persist_to_sqlite_and_round_trip() {
    // ── Run the pipeline ─────────────────────────────────────────
    let format = AudioFormat::WHISPER;
    let frames = vec![sine_at(16_000, 16_000, 0.5), sine_at(16_000, 16_000, 0.5)];
    let capture = FakeCapture::new(format, frames);
    let resampler = Arc::new(RubatoResamplerAdapter::new()) as Arc<dyn Resampler>;
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
    let chunks: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            TranscriptEvent::Chunk { segments, .. } => Some(segments.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(chunks.len(), 2);

    // ── Persist to SQLite ────────────────────────────────────────
    let store = SqliteMeetingStore::open_in_memory().await.unwrap();
    let session_id = if let TranscriptEvent::Started { session_id, .. } = &events[0] {
        *session_id
    } else {
        panic!("first event must be Started");
    };

    let all_segments: Vec<_> = chunks.into_iter().flatten().collect();
    let meeting_id = MeetingId::new();
    let summary = store
        .create(CreateMeeting {
            id: meeting_id,
            title: format!("test-{session_id}"),
            input_format: format,
        })
        .await
        .unwrap();
    assert_eq!(summary.id, meeting_id);

    store
        .append_segments(meeting_id, &all_segments)
        .await
        .unwrap();

    // ── Query back ───────────────────────────────────────────────
    let meeting = store.get(meeting_id).await.unwrap().unwrap();
    assert_eq!(meeting.segments.len(), all_segments.len());
    assert_eq!(meeting.summary.title, format!("test-{session_id}"));
    for (stored, original) in meeting.segments.iter().zip(all_segments.iter()) {
        assert_eq!(stored.text, original.text);
    }
}

/// Two concurrent pipeline sessions produce independent event streams.
#[tokio::test]
async fn two_concurrent_sessions_do_not_interfere() {
    let format = AudioFormat::WHISPER;

    let capture_a = FakeCapture::new(format, vec![sine_at(16_000, 16_000, 0.5)]);
    let capture_b = FakeCapture::new(format, vec![sine_at(16_000, 16_000, 0.3)]);
    let resampler = Arc::new(RubatoResamplerAdapter::new()) as Arc<dyn Resampler>;
    let transcriber = FakeTranscriber::new();

    let pipeline_a = StreamingPipeline::new(capture_a, resampler.clone(), transcriber.clone());
    let pipeline_b = StreamingPipeline::new(capture_b, resampler, transcriber.clone());

    let opts = StreamingOptions {
        chunk_ms: 1_000,
        silence_rms_threshold: 0.005,
        language: None,
        ..StreamingOptions::default()
    };

    let mut handle_a = pipeline_a.start(opts.clone()).await.unwrap();
    let mut handle_b = pipeline_b.start(opts).await.unwrap();

    let (events_a, events_b) = tokio::join!(
        drain_until_stopped(&mut handle_a),
        drain_until_stopped(&mut handle_b)
    );

    // Both sessions must emit Started + Chunk + Stopped independently.
    let session_id_a = match &events_a[0] {
        TranscriptEvent::Started { session_id, .. } => *session_id,
        _ => panic!("expected Started"),
    };
    let session_id_b = match &events_b[0] {
        TranscriptEvent::Started { session_id, .. } => *session_id,
        _ => panic!("expected Started"),
    };
    assert_ne!(
        session_id_a, session_id_b,
        "sessions must have distinct ids"
    );

    let chunks_a = events_a
        .iter()
        .filter(|e| matches!(e, TranscriptEvent::Chunk { .. }))
        .count();
    let chunks_b = events_b
        .iter()
        .filter(|e| matches!(e, TranscriptEvent::Chunk { .. }))
        .count();
    assert_eq!(chunks_a, 1);
    assert_eq!(chunks_b, 1);

    // ASR was called at least twice total (one per session).
    assert!(transcriber.call_count() >= 2);
}
