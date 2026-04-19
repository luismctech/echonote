//! End-to-end WAV roundtrip without touching real hardware.
//!
//! These tests run in CI (no microphone required). A live cpal smoke
//! test is gated behind the `ECHO_E2E_AUDIO=1` env var because the CI
//! runner has no audio device.

use echo_audio::{WavSink, WriteOptions};
use echo_domain::{AudioFormat, AudioFrame};

#[test]
fn synthetic_tone_roundtrips_through_wav() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("tone.wav");
    let format = AudioFormat::WHISPER;

    // Build a 1 s, 440 Hz sine at -6 dBFS in 100 ms chunks to mimic
    // what the cpal callback would deliver.
    let chunk_ms = 100u64;
    let chunk_samples = (format.sample_rate_hz as u64 * chunk_ms / 1_000) as usize;
    let chunks = (1_000 / chunk_ms) as usize;

    let mut sink = WavSink::create(&path, format, WriteOptions::default()).unwrap();
    let mut t = 0.0f32;
    let dt = 1.0 / format.sample_rate_hz as f32;

    for _ in 0..chunks {
        let mut samples = Vec::with_capacity(chunk_samples);
        for _ in 0..chunk_samples {
            samples.push(0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin());
            t += dt;
        }
        sink.write_frame(&AudioFrame {
            samples,
            format,
            captured_at_ns: 0,
        })
        .unwrap();
    }

    let bytes_written = sink.samples_written();
    sink.finalize().unwrap();

    let reader = hound::WavReader::open(&path).unwrap();
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16_000);
    assert_eq!(spec.channels, 1);
    assert_eq!(spec.bits_per_sample, 16);
    assert_eq!(reader.duration(), 16_000); // 1 s of mono = 16000 frames
    assert_eq!(bytes_written, 16_000);
}

/// Live capture from the host microphone.
///
/// Disabled by default. Run locally with:
///
/// ```sh
/// ECHO_E2E_AUDIO=1 cargo test -p echo-audio --test wav_roundtrip -- \
///     --ignored --nocapture
/// ```
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires a real microphone; run with ECHO_E2E_AUDIO=1"]
async fn live_microphone_capture_writes_a_wav() {
    if std::env::var("ECHO_E2E_AUDIO").ok().as_deref() != Some("1") {
        eprintln!("skipping live capture (set ECHO_E2E_AUDIO=1 to run)");
        return;
    }

    use echo_audio::CpalMicrophoneCapture;
    use echo_domain::{AudioCapture, AudioSource, CaptureSpec};
    use std::sync::Arc;
    use std::time::Duration;

    let capture = Arc::new(CpalMicrophoneCapture::new());
    let spec = CaptureSpec {
        source: AudioSource::Microphone,
        device_id: None,
        preferred_format: AudioFormat::WHISPER,
    };

    let mut stream = capture
        .start(spec)
        .await
        .expect("start should succeed when a mic is present");
    let format = stream.format();

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("live.wav");
    let mut sink = WavSink::create(&path, format, WriteOptions::default()).unwrap();

    let deadline = tokio::time::Instant::now() + Duration::from_millis(500);
    let mut frames = 0u32;
    while tokio::time::Instant::now() < deadline {
        let remaining = deadline - tokio::time::Instant::now();
        match tokio::time::timeout(remaining, stream.next_frame()).await {
            Ok(Some(frame)) => {
                sink.write_frame(&frame).unwrap();
                frames += 1;
            }
            Ok(None) => break,
            Err(_) => break,
        }
    }

    stream.stop().await.unwrap();
    let written = sink.samples_written();
    sink.finalize().unwrap();

    eprintln!(
        "live capture: {frames} frames, {written} samples, {} Hz × {} ch",
        format.sample_rate_hz, format.channels
    );
    assert!(frames > 0, "no frames captured from the microphone");
    assert!(written > 0, "no samples written");
}
