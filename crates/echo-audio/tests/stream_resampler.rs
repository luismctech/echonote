//! Contract of the stateful resampler behind the `Resampler` port:
//! exact output length, block-size independence, time alignment and
//! recovery after a bad format, across the device rates EchoNote meets.

use echo_audio::{resample_to_whisper, RubatoResamplerAdapter, StreamResampler};
use echo_domain::{AudioFormat, Resampler};

fn fmt(r: u32, c: u16) -> AudioFormat {
    AudioFormat {
        sample_rate_hz: r,
        channels: c,
    }
}

fn noise(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s as f32 / u32::MAX as f32) - 0.5
        })
        .collect()
}

fn expected(frames: usize, rate: u32) -> usize {
    (frames as f64 * 16_000.0 / rate as f64).round() as usize
}

#[test]
fn one_shot_length_is_round_frames_times_ratio_for_every_rate() {
    for &rate in &[
        8_000u32, 11_025, 22_050, 32_000, 44_100, 48_000, 96_000, 192_000, 16_000,
    ] {
        for &ch in &[1u16, 2, 6] {
            for &frames in &[
                0usize, 1, 2, 3, 100, 1023, 1024, 1025, 4799, 48_000, 100_003,
            ] {
                let x = noise(frames * ch as usize, frames as u32);
                let out = resample_to_whisper(&x, fmt(rate, ch)).unwrap();
                assert_eq!(
                    out.len(),
                    expected(frames, rate),
                    "rate {rate} ch {ch} frames {frames}"
                );
                let a = RubatoResamplerAdapter::new()
                    .to_whisper(&x, fmt(rate, ch))
                    .unwrap();
                assert_eq!(a, out);
            }
        }
    }
}

#[test]
fn irregular_blocks_match_one_shot() {
    for &rate in &[8_000u32, 22_050, 44_100, 48_000, 96_000] {
        let frames = 37_777usize;
        let x = noise(frames * 2, rate);
        let whole = resample_to_whisper(&x, fmt(rate, 2)).unwrap();
        let mut s = StreamResampler::new(16_000);
        let mut out = Vec::new();
        let sizes = [0usize, 1, 7, 480, 0, 1024, 2048, 13, 5000, 1];
        let mut i = 0;
        let mut k = 0;
        while i < frames {
            let n = sizes[k % sizes.len()].min(frames - i);
            s.push(&x[i * 2..(i + n) * 2], fmt(rate, 2), &mut out)
                .unwrap();
            i += n;
            k += 1;
        }
        s.flush(&mut out).unwrap();
        assert_eq!(out.len(), whole.len(), "rate {rate}");
        let d = out
            .iter()
            .zip(&whole)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(d < 1e-5, "rate {rate} diff {d}");
    }
}

#[test]
fn output_never_runs_ahead_of_input() {
    // Output may lag but never exceed round(frames*ratio) mid-run.
    for &rate in &[8_000u32, 11_025, 44_100, 48_000] {
        let mut s = StreamResampler::new(16_000);
        let mut out = Vec::new();
        let mut frames = 0usize;
        for b in 0..400 {
            let n = 1 + (b * 37) % 900;
            s.push(&noise(n, b as u32), fmt(rate, 1), &mut out).unwrap();
            frames += n;
            assert!(
                out.len() <= expected(frames, rate),
                "rate {rate} overshoot {} > {}",
                out.len(),
                expected(frames, rate)
            );
        }
        s.flush(&mut out).unwrap();
        assert_eq!(out.len(), expected(frames, rate));
    }
}

#[test]
fn second_flush_adds_nothing_and_next_run_starts_fresh() {
    let mut s = StreamResampler::new(16_000);
    let mut out = Vec::new();
    s.push(&noise(4800, 1), fmt(48_000, 1), &mut out).unwrap();
    s.flush(&mut out).unwrap();
    assert_eq!(out.len(), 1600);
    s.flush(&mut out).unwrap();
    assert_eq!(out.len(), 1600, "second flush adds nothing");
    // A new run after flush equals a fresh one-shot.
    let y = noise(4800, 2);
    let mut run2 = Vec::new();
    s.push(&y, fmt(48_000, 1), &mut run2).unwrap();
    s.flush(&mut run2).unwrap();
    assert_eq!(run2, resample_to_whisper(&y, fmt(48_000, 1)).unwrap());
}

#[test]
fn empty_push_in_a_new_format_keeps_the_length() {
    let mut s = StreamResampler::new(16_000);
    let mut out = Vec::new();
    s.push(&noise(4800, 1), fmt(48_000, 1), &mut out).unwrap();
    s.push(&[], fmt(44_100, 2), &mut out).unwrap(); // format change with no audio
    s.push(&noise(4410 * 2, 3), fmt(44_100, 2), &mut out)
        .unwrap();
    s.flush(&mut out).unwrap();
    assert_eq!(out.len(), 1600 + 1600);
}

#[test]
fn invalid_format_mid_stream_flushes_and_stays_usable() {
    let mut s = StreamResampler::new(16_000);
    let mut out = Vec::new();
    s.push(&noise(4800, 1), fmt(48_000, 1), &mut out).unwrap();
    let before = out.len();
    let err = s.push(&[0.0; 10], fmt(0, 1), &mut out);
    assert!(err.is_err());
    // The old run was flushed into `out` before the error.
    assert_eq!(out.len(), 1600, "before {before}");
    s.push(&noise(4800, 2), fmt(48_000, 1), &mut out).unwrap();
    s.flush(&mut out).unwrap();
    assert_eq!(out.len(), 3200);
}

#[test]
fn one_shot_rejects_invalid_formats() {
    assert!(resample_to_whisper(&[0.0; 4], fmt(0, 1)).is_err());
    assert!(resample_to_whisper(&[0.0; 4], fmt(16_000, 0)).is_err());
    assert!(resample_to_whisper(&[], fmt(0, 1)).is_err());
}

#[test]
fn impulse_stays_in_place_at_every_rate() {
    for &rate in &[8_000u32, 22_050, 44_100, 48_000, 96_000] {
        let n = rate as usize; // 1 s
        let mut x = vec![0.0f32; n];
        x[n / 2] = 1.0;
        let out = resample_to_whisper(&x, fmt(rate, 1)).unwrap();
        let pk = (0..out.len())
            .max_by(|&a, &b| out[a].abs().total_cmp(&out[b].abs()))
            .unwrap();
        assert!(pk.abs_diff(8000) <= 1, "rate {rate} peak at {pk}");
    }
}
