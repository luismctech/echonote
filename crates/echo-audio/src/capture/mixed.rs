//! Mixed-source audio capture: mic + system audio merged into one stream.
//!
//! `MixedStream` reads concurrently from a microphone stream and a system
//! audio (loopback) stream, normalizes each source to a common target
//! format (channel downmix + sample-rate conversion), and emits a mixed
//! frame whenever both per-source buffers cover the next threshold.
//! Mixing is a per-sample average over the *normalized* samples, so
//! sources delivering different native rates or channel counts (e.g.
//! built-in mic mono 48 kHz vs ScreenCaptureKit stereo 48 kHz) line up
//! correctly in the time domain. A source that stops delivering is
//! padded with silence after a bounded skew, so one dead source never
//! stalls the mix.
//!
//! Each source can be independently muted at any time by storing `false`
//! in the corresponding `Arc<AtomicBool>` returned alongside the stream.
//! When a source is muted its samples are zeroed in the mix; the underlying
//! capture keeps running so no audio is dropped on unmute.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tracing::{debug, warn};

use echo_domain::{AudioFormat, AudioFrame, AudioStream, DomainError, Sample};

use crate::preprocess::resample::{ResampleError, StreamResampler};

/// Samples per emitted mixed frame. ~10 ms at 16 kHz mono.
/// Picked to match the cadence of upstream capture frames.
const MIX_THRESHOLD: usize = 160;

/// Maximum per-source buffer depth before old normalized samples are
/// dropped. ~340 ms at 16 kHz mono — protects against pathological clock
/// drift between mic and system streams.
const MAX_DRIFT_SAMPLES: usize = MIX_THRESHOLD * 32;

/// Lead one source may build up over the other before the lagging one
/// is treated as stalled and padded with silence: 200 ms at 16 kHz.
/// Below this the mixer waits for the lagging source so both sides of a
/// frame cover the same time span; above it a dead source would
/// otherwise stall the mix for good.
const STALL_SKEW_SAMPLES: usize = MIX_THRESHOLD * 20;

/// Channel depth in mixed frames. ~5 s of headroom at 10 ms/frame.
const CHANNEL_CAPACITY: usize = 512;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Atomic flags that independently gate each source's contribution to the mix.
///
/// Both flags start as `true` (both sources contribute). Setting a flag to
/// `false` zeros out that source's samples; setting it back to `true`
/// restores contribution immediately on the next emitted frame.
#[derive(Clone)]
pub struct MixControls {
    /// When `true` the microphone samples are included in the mix.
    pub mic_active: Arc<AtomicBool>,
    /// When `true` the system-audio samples are included in the mix.
    pub sys_active: Arc<AtomicBool>,
}

impl Default for MixControls {
    fn default() -> Self {
        Self {
            mic_active: Arc::new(AtomicBool::new(true)),
            sys_active: Arc::new(AtomicBool::new(true)),
        }
    }
}

/// An [`AudioStream`] that merges a microphone stream and a system-audio
/// stream into a single output. Construct via [`MixedStream::new`].
pub struct MixedStream {
    rx: mpsc::Receiver<AudioFrame>,
    stop_tx: Option<oneshot::Sender<()>>,
    join: Option<JoinHandle<()>>,
    format: AudioFormat,
}

impl MixedStream {
    /// Spawn the mixer task and return the stream + its control handles.
    ///
    /// Both `mic_stream` and `sys_stream` **must** be live capture streams
    /// (calling `next_frame()` should not immediately return `None`).
    ///
    /// `target_format` is the format the mixer will emit. Each source is
    /// downmixed to mono (`channels = 1`) and resampled from its native
    /// rate to `target_format.sample_rate_hz`. The target format is
    /// typically [`AudioFormat::WHISPER`] (16 kHz mono) so downstream
    /// resampling is a no-op.
    ///
    /// Returns an error if a per-source resampler cannot be built (e.g.
    /// when the source format has zero sample rate). When that happens
    /// neither stream is started — callers can fall back to single-source
    /// capture.
    pub fn new(
        mic_stream: Box<dyn AudioStream>,
        sys_stream: Box<dyn AudioStream>,
        target_format: AudioFormat,
    ) -> Result<(Self, MixControls), DomainError> {
        let mic_format = mic_stream.format();
        let sys_format = sys_stream.format();

        // Build the per-source normalizers up front so we surface any
        // resampler-construction error synchronously instead of panicking
        // inside the spawned task.
        let mic_norm = normalizer("mic", mic_format, target_format)?;
        let sys_norm = normalizer("sys", sys_format, target_format)?;

        debug!(
            target = ?target_format,
            mic = ?mic_format,
            sys = ?sys_format,
            "mixed-stream: starting with per-source normalization"
        );

        let controls = MixControls::default();
        let (tx, rx) = mpsc::channel::<AudioFrame>(CHANNEL_CAPACITY);
        let (stop_tx, stop_rx) = oneshot::channel::<()>();

        let mic_active = controls.mic_active.clone();
        let sys_active = controls.sys_active.clone();

        let join = tokio::spawn(run_mixer(
            mic_stream,
            sys_stream,
            mic_norm,
            sys_norm,
            target_format,
            tx,
            stop_rx,
            mic_active,
            sys_active,
        ));

        let stream = Self {
            rx,
            stop_tx: Some(stop_tx),
            join: Some(join),
            format: target_format,
        };

        Ok((stream, controls))
    }
}

#[async_trait]
impl AudioStream for MixedStream {
    fn format(&self) -> AudioFormat {
        self.format
    }

    async fn next_frame(&mut self) -> Option<AudioFrame> {
        self.rx.recv().await
    }

    async fn stop(&mut self) -> Result<(), DomainError> {
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }
        if let Some(join) = self.join.take() {
            join.abort();
        }
        self.rx.close();
        Ok(())
    }
}

impl Drop for MixedStream {
    fn drop(&mut self) {
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }
        if let Some(join) = self.join.take() {
            join.abort();
        }
    }
}

// ---------------------------------------------------------------------------
// Per-source normalizer
// ---------------------------------------------------------------------------

/// Builds the stateful converter of one source to the mono target
/// rate. It follows the format each frame carries, so a source that
/// reconnects with another configuration keeps its speed and timing.
fn normalizer(
    source: &str,
    native: AudioFormat,
    target: AudioFormat,
) -> Result<StreamResampler, DomainError> {
    let built = if target.channels == 1 {
        StreamResampler::for_input(native, target.sample_rate_hz)
    } else {
        Err(ResampleError::InvalidFormat(target))
    };
    built.map_err(|e| {
        DomainError::AudioCaptureFailed(format!(
            "{source} normalizer ({native:?} -> {target:?}): {e}"
        ))
    })
}

/// Appends `frame`, converted to the target format, to `buf`.
fn normalize(norm: &mut StreamResampler, frame: &AudioFrame, buf: &mut Vec<Sample>, source: &str) {
    if let Err(e) = norm.push(&frame.samples, frame.format, buf) {
        warn!(source, error = %e, format = ?frame.format, "mixed-stream: cannot convert frame; dropping it");
    }
}

// ---------------------------------------------------------------------------
// Mixer task
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
async fn run_mixer(
    mut mic: Box<dyn AudioStream>,
    mut sys: Box<dyn AudioStream>,
    mut mic_norm: StreamResampler,
    mut sys_norm: StreamResampler,
    target_format: AudioFormat,
    tx: mpsc::Sender<AudioFrame>,
    mut stop_rx: oneshot::Receiver<()>,
    mic_active: Arc<AtomicBool>,
    sys_active: Arc<AtomicBool>,
) {
    debug!("mixed-stream mixer started");

    let mut mic_buf: Vec<Sample> = Vec::with_capacity(MIX_THRESHOLD * 4);
    let mut sys_buf: Vec<Sample> = Vec::with_capacity(MIX_THRESHOLD * 4);
    let mut elapsed_ns: u64 = 0;

    // Nanoseconds per (mono) sample at the declared target rate.
    let ns_per_sample = if target_format.sample_rate_hz > 0 {
        1_000_000_000u64 / u64::from(target_format.sample_rate_hz)
    } else {
        0
    };

    loop {
        // Emit every frame both sources already cover. A frame takes the
        // same time span from each buffer, so the two sides stay aligned;
        // a source that stops delivering is padded with silence once the
        // other leads by `STALL_SKEW_SAMPLES`, so the mix keeps real time.
        loop {
            let n = MIX_THRESHOLD;
            let (take_mic, take_sys) = match (mic_buf.len() >= n, sys_buf.len() >= n) {
                (true, true) => (true, true),
                (true, false) if mic_buf.len() >= STALL_SKEW_SAMPLES => (true, false),
                (false, true) if sys_buf.len() >= STALL_SKEW_SAMPLES => (false, true),
                _ => break,
            };
            let use_mic = mic_active.load(Ordering::Relaxed);
            let use_sys = sys_active.load(Ordering::Relaxed);

            // Gain follows the controls, not availability: a stalled source
            // counts as silence at the same mix gain, so the level never
            // jumps while both sources are enabled.
            let mut mixed = Vec::with_capacity(n);
            for i in 0..n {
                let mic_s = if take_mic { mic_buf[i] } else { 0.0 };
                let sys_s = if take_sys { sys_buf[i] } else { 0.0 };
                let sample = match (use_mic, use_sys) {
                    (true, true) => (mic_s + sys_s) * 0.5,
                    (true, false) => mic_s,
                    (false, true) => sys_s,
                    (false, false) => 0.0_f32,
                };
                mixed.push(sample.clamp(-1.0, 1.0));
            }

            if take_mic {
                mic_buf.drain(..n);
            }
            if take_sys {
                sys_buf.drain(..n);
            }

            let frame = AudioFrame {
                samples: mixed,
                format: target_format,
                captured_at_ns: elapsed_ns,
            };
            elapsed_ns = elapsed_ns.saturating_add(ns_per_sample * n as u64);

            if tx.send(frame).await.is_err() {
                debug!("mixed-stream: consumer dropped, stopping mixer");
                return;
            }
        }

        // Wait for new samples from either source.
        tokio::select! {
            _ = &mut stop_rx => {
                debug!("mixed-stream: stop signal received");
                break;
            }
            frame = mic.next_frame() => {
                match frame {
                    Some(f) => normalize(&mut mic_norm, &f, &mut mic_buf, "mic"),
                    None => {
                        debug!("mixed-stream: mic stream ended");
                        break;
                    }
                }
            }
            frame = sys.next_frame() => {
                match frame {
                    Some(f) => normalize(&mut sys_norm, &f, &mut sys_buf, "sys"),
                    None => {
                        debug!("mixed-stream: system stream ended");
                        break;
                    }
                }
            }
        }

        // Trim buffers that have drifted too far ahead.
        if mic_buf.len() > MAX_DRIFT_SAMPLES {
            let excess = mic_buf.len() - MAX_DRIFT_SAMPLES;
            warn!(
                excess,
                "mixed-stream: mic buffer drifted; dropping old samples"
            );
            mic_buf.drain(..excess);
        }
        if sys_buf.len() > MAX_DRIFT_SAMPLES {
            let excess = sys_buf.len() - MAX_DRIFT_SAMPLES;
            warn!(
                excess,
                "mixed-stream: sys buffer drifted; dropping old samples"
            );
            sys_buf.drain(..excess);
        }
    }

    drop(mic_norm);
    drop(sys_norm);
    debug!("mixed-stream mixer stopped");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::time::Duration;
    use tokio::sync::Mutex as AsyncMutex;

    /// Test stream that yields a queue of pre-built frames then ends.
    struct ScriptedStream {
        format: AudioFormat,
        frames: AsyncMutex<std::collections::VecDeque<AudioFrame>>,
    }

    impl ScriptedStream {
        fn boxed(format: AudioFormat, frames: Vec<AudioFrame>) -> Box<dyn AudioStream> {
            Box::new(Self {
                format,
                frames: AsyncMutex::new(frames.into()),
            })
        }
    }

    #[async_trait]
    impl AudioStream for ScriptedStream {
        fn format(&self) -> AudioFormat {
            self.format
        }
        async fn next_frame(&mut self) -> Option<AudioFrame> {
            let mut q = self.frames.lock().await;
            if let Some(f) = q.pop_front() {
                Some(f)
            } else {
                // Block forever — emulates a live stream that hasn't ended.
                drop(q);
                std::future::pending::<()>().await;
                None
            }
        }
        async fn stop(&mut self) -> Result<(), DomainError> {
            Ok(())
        }
    }

    fn sine(samples: usize, freq_hz: f32, rate_hz: u32, channels: u16) -> Vec<f32> {
        let mut v = Vec::with_capacity(samples);
        let frames = samples / channels as usize;
        let two_pi_f_over_sr = 2.0 * std::f32::consts::PI * freq_hz / rate_hz as f32;
        for i in 0..frames {
            let s = (two_pi_f_over_sr * i as f32).sin() * 0.5;
            for _ in 0..channels {
                v.push(s);
            }
        }
        v
    }

    fn rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum: f32 = samples.iter().map(|s| s * s).sum();
        (sum / samples.len() as f32).sqrt()
    }

    /// Reproduces the original bug: mic at mono 48 kHz, sys at stereo
    /// 48 kHz, mixer normalizes both to 16 kHz mono so neither source
    /// dominates the other.
    #[tokio::test]
    async fn mismatched_native_formats_normalize_to_target() {
        let mic_format = AudioFormat {
            sample_rate_hz: 48_000,
            channels: 1,
        };
        let sys_format = AudioFormat {
            sample_rate_hz: 48_000,
            channels: 2,
        };

        // 100 ms of audio in each native format.
        let mic_frame = AudioFrame {
            samples: sine(4_800, 440.0, 48_000, 1),
            format: mic_format,
            captured_at_ns: 0,
        };
        let sys_frame = AudioFrame {
            samples: sine(9_600, 880.0, 48_000, 2),
            format: sys_format,
            captured_at_ns: 0,
        };

        let mic = ScriptedStream::boxed(mic_format, vec![mic_frame]);
        let sys = ScriptedStream::boxed(sys_format, vec![sys_frame]);

        let (mut stream, _ctrl) =
            MixedStream::new(mic, sys, AudioFormat::WHISPER).expect("build mixed stream");

        assert_eq!(stream.format(), AudioFormat::WHISPER);

        // Drain a few frames and check they have meaningful RMS — proves
        // the mic contribution survives even when the sys stream pushes
        // twice as many native samples per millisecond.
        let mut received = Vec::new();
        for _ in 0..4 {
            let next = tokio::time::timeout(Duration::from_millis(200), stream.next_frame()).await;
            if let Ok(Some(f)) = next {
                received.extend_from_slice(&f.samples);
            }
        }

        assert!(
            !received.is_empty(),
            "mixer must produce frames from mismatched-format sources"
        );
        let r = rms(&received);
        assert!(
            r > 0.05,
            "expected non-trivial mixed RMS, got {r} — mic likely lost in the mix"
        );
    }

    /// When the system source is muted via `MixControls`, the output must
    /// still contain mic samples at full energy (no halving from the
    /// `(mic+sys)*0.5` average).
    #[tokio::test]
    async fn muted_sys_passes_mic_through_unattenuated() {
        let mic_format = AudioFormat {
            sample_rate_hz: 48_000,
            channels: 1,
        };
        let sys_format = AudioFormat {
            sample_rate_hz: 48_000,
            channels: 2,
        };

        let mic_frame = AudioFrame {
            samples: sine(4_800, 440.0, 48_000, 1),
            format: mic_format,
            captured_at_ns: 0,
        };
        let sys_frame = AudioFrame {
            samples: vec![0.0; 9_600],
            format: sys_format,
            captured_at_ns: 0,
        };

        let mic = ScriptedStream::boxed(mic_format, vec![mic_frame]);
        let sys = ScriptedStream::boxed(sys_format, vec![sys_frame]);

        let (mut stream, ctrl) =
            MixedStream::new(mic, sys, AudioFormat::WHISPER).expect("build mixed stream");

        // Mute sys before any frames are produced.
        ctrl.sys_active.store(false, Ordering::Relaxed);

        // Drain every frame the mixer emits for the single mic frame.
        // The scripted sources block after their first frame, so a
        // timeout with no frame means the mixer has flushed its buffers.
        let mut frames = Vec::new();
        while let Ok(Some(f)) =
            tokio::time::timeout(Duration::from_millis(200), stream.next_frame()).await
        {
            frames.push(f);
        }

        assert!(
            !frames.is_empty(),
            "mixer must keep producing with sys muted"
        );

        // "Unattenuated" means at least one emitted frame carries the mic
        // sine at (near) full energy. We check the PEAK per-frame RMS,
        // not the RMS of the whole concatenation: when the muted sys
        // buffer crosses the frame threshold before the mic buffer does,
        // the mixer emits leading silence frames, and how many is
        // scheduling-dependent — averaging them in is what made the old
        // concatenated-RMS assertion flaky. A 440 Hz sine at amplitude
        // 0.5 has RMS ≈ 0.354; the old halving bug ((mic+0)*0.5) would
        // cap every frame at ≈0.177, so a 0.3 floor cleanly separates
        // "fixed" from "regressed" regardless of interleaved silence.
        let peak_rms = frames
            .iter()
            .map(|f| rms(&f.samples))
            .fold(0.0_f32, f32::max);
        assert!(
            peak_rms > 0.3,
            "muted sys still attenuating: peak per-frame RMS {peak_rms} (expected ≈0.354)"
        );
    }

    /// Power of `freq_hz` in `samples` (Goertzel), normalised by length.
    fn tone_power(samples: &[f32], freq_hz: f32, rate_hz: u32) -> f32 {
        let w = 2.0 * std::f32::consts::PI * freq_hz / rate_hz as f32;
        let coeff = 2.0 * w.cos();
        let (mut s1, mut s2) = (0.0_f32, 0.0_f32);
        for &x in samples {
            let s0 = x + coeff * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        (s1 * s1 + s2 * s2 - coeff * s1 * s2) / samples.len().max(1) as f32
    }

    /// `n` samples of a sine split into frames of `frame` samples.
    fn frames_of(n: usize, frame: usize, freq_hz: f32, format: AudioFormat) -> Vec<AudioFrame> {
        let all = sine(n, freq_hz, format.sample_rate_hz, format.channels);
        all.chunks(frame)
            .map(|c| AudioFrame {
                samples: c.to_vec(),
                format,
                captured_at_ns: 0,
            })
            .collect()
    }

    async fn drain(stream: &mut MixedStream, idle: Duration) -> Vec<AudioFrame> {
        let mut frames = Vec::new();
        while let Ok(Some(f)) = tokio::time::timeout(idle, stream.next_frame()).await {
            frames.push(f);
        }
        frames
    }

    /// Mic and system audio that arrive interleaved in time must be
    /// summed sample by sample, not emitted one after the other.
    #[tokio::test]
    async fn both_sources_are_mixed_into_the_same_frames() {
        let fmt = AudioFormat::WHISPER;
        let total = 3_200;
        let mic = ScriptedStream::boxed(fmt, frames_of(total, 800, 440.0, fmt));
        let sys = ScriptedStream::boxed(fmt, frames_of(total, 800, 1_000.0, fmt));
        let (mut stream, _ctrl) = MixedStream::new(mic, sys, fmt).expect("build mixed stream");

        let frames = drain(&mut stream, Duration::from_millis(200)).await;
        let emitted: usize = frames.iter().map(|f| f.samples.len()).sum();
        assert_eq!(emitted, total, "one output sample per aligned input sample");
        for (i, f) in frames.iter().enumerate() {
            let p440 = tone_power(&f.samples, 440.0, fmt.sample_rate_hz);
            let p1000 = tone_power(&f.samples, 1_000.0, fmt.sample_rate_hz);
            let silence = tone_power(&f.samples, 2_500.0, fmt.sample_rate_hz);
            assert!(
                p440 > 10.0 * silence && p1000 > 10.0 * silence,
                "frame {i} must carry both tones: 440 Hz {p440}, 1 kHz {p1000}, floor {silence}"
            );
        }
    }

    /// A mic that reconnects with another configuration tags its frames
    /// with the new format; the mixer must follow it so the mic keeps
    /// its speed and stays aligned with system audio.
    #[tokio::test]
    async fn mic_format_change_after_reconnect_keeps_time() {
        let mono_48k = AudioFormat {
            sample_rate_hz: 48_000,
            channels: 1,
        };
        let stereo_16k = AudioFormat {
            sample_rate_hz: 16_000,
            channels: 2,
        };
        // 100 ms before the reconnect, 100 ms after it.
        let mut mic_frames = frames_of(4_800, 960, 440.0, mono_48k);
        mic_frames.extend(frames_of(3_200, 640, 440.0, stereo_16k));
        let mic = ScriptedStream::boxed(mono_48k, mic_frames);
        let fmt = AudioFormat::WHISPER;
        let sys = ScriptedStream::boxed(fmt, frames_of(3_200, 320, 1_000.0, fmt));
        let (mut stream, _ctrl) = MixedStream::new(mic, sys, fmt).expect("build mixed stream");

        let frames = drain(&mut stream, Duration::from_millis(200)).await;
        let all: Vec<f32> = frames
            .iter()
            .flat_map(|f| f.samples.iter().copied())
            .collect();
        assert_eq!(all.len(), 3_200, "200 ms in from each source, 200 ms out");
        let after = &all[1_600..];
        let p440 = tone_power(after, 440.0, fmt.sample_rate_hz);
        let p1000 = tone_power(after, 1_000.0, fmt.sample_rate_hz);
        let floor = tone_power(after, 2_500.0, fmt.sample_rate_hz);
        assert!(
            p440 > 10.0 * floor && p1000 > 10.0 * floor,
            "after the reconnect both tones must survive: 440 Hz {p440}, 1 kHz {p1000}, floor {floor}"
        );
    }

    /// A source that stops delivering must not stall the other: after a
    /// bounded skew the missing side is padded with silence, at the same
    /// gain as when both sources flow.
    #[tokio::test]
    async fn a_stalled_source_is_padded_with_silence() {
        let fmt = AudioFormat::WHISPER;
        let total = 6_400;
        let mic = ScriptedStream::boxed(fmt, frames_of(total, 800, 440.0, fmt));
        let sys = ScriptedStream::boxed(fmt, Vec::new());
        let (mut stream, _ctrl) = MixedStream::new(mic, sys, fmt).expect("build mixed stream");

        let frames = drain(&mut stream, Duration::from_millis(200)).await;
        let emitted: usize = frames.iter().map(|f| f.samples.len()).sum();
        assert!(
            emitted >= total - STALL_SKEW_SAMPLES && emitted <= total,
            "mic must keep flowing once the skew bound is reached: {emitted} of {total}"
        );
        let all: Vec<f32> = frames
            .iter()
            .flat_map(|f| f.samples.iter().copied())
            .collect();
        let r = rms(&all);
        // 0.5 amplitude sine ⇒ RMS 0.354; mixed with silence at 0.5 gain ⇒ 0.177.
        assert!(
            (r - 0.177).abs() < 0.03,
            "stalled sys must be silence at mix gain: rms {r}"
        );
    }
}
