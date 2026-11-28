//! Writing frames whose format differs from the file's (a device that
//! reconnected in another configuration) and the one-shot adapter's
//! cached filter: exact durations, placement, recovery after a bad
//! format, block-size independence and concurrent callers.

use echo_audio::{
    resample_to_whisper, FormatConverter, RubatoResamplerAdapter, WavSink, WriteOptions,
};
use echo_domain::{AudioFormat, AudioFrame, Resampler};
use std::sync::Arc;

fn f(rate: u32, ch: u16) -> AudioFormat {
    AudioFormat {
        sample_rate_hz: rate,
        channels: ch,
    }
}
fn fr(samples: Vec<f32>, format: AudioFormat) -> AudioFrame {
    AudioFrame {
        samples,
        format,
        captured_at_ns: 0,
    }
}
fn tone(rate: u32, hz: f32, secs: f32, ch: u16) -> Vec<f32> {
    let n = (rate as f32 * secs).round() as usize;
    (0..n)
        .flat_map(|i| {
            let s = 0.5 * (2.0 * std::f32::consts::PI * hz * i as f32 / rate as f32).sin();
            std::iter::repeat_n(s, ch as usize)
        })
        .collect()
}
/// A scratch WAV path, removed with the returned directory.
fn tmp() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("out.wav");
    (dir, path)
}
fn read(p: &std::path::Path) -> (hound::WavSpec, Vec<i16>) {
    let mut r = hound::WavReader::open(p).unwrap();
    let spec = r.spec();
    (spec, r.samples::<i16>().map(|s| s.unwrap()).collect())
}

/// native -> other -> native; other-format audio lands between the native runs (tail drained before native resumes).
#[test]
fn converted_run_sits_between_native_runs() {
    let file = f(48_000, 2);
    let (_dir, p) = tmp();
    let mut sink = WavSink::create(&p, file, WriteOptions::default()).unwrap();
    sink.write_converting(&fr(vec![0.0; 9_600], file)).unwrap();
    // 100 ms tone at 16 kHz mono in 10 ms frames
    let t = tone(16_000, 440.0, 0.1, 1);
    for c in t.chunks(160) {
        sink.write_converting(&fr(c.to_vec(), f(16_000, 1)))
            .unwrap();
    }
    sink.write_converting(&fr(vec![0.0; 9_600], file)).unwrap();
    let sw = sink.samples_written();
    let p2 = sink.finalize().unwrap();
    let (spec, s) = read(&p2);
    assert_eq!((spec.sample_rate, spec.channels), (48_000, 2));
    assert_eq!(s.len(), 3 * 9_600);
    assert_eq!(sw, s.len() as u64);
    let energy = |a: &[i16]| a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>();
    assert_eq!(energy(&s[..9_600]), 0.0, "nothing before");
    assert_eq!(
        energy(&s[19_200..]),
        0.0,
        "tail must not leak after native resumes"
    );
    let mid = &s[9_600..19_200];
    let peak = mid.iter().map(|x| x.abs()).max().unwrap();
    assert!(peak > 14_000 && peak < 18_000, "peak {peak}");
    assert!(mid.chunks(2).all(|c| c[0] == c[1]));
}

/// switching between two non-native formats and ending on a non-native run (finalize drains the tail).
#[test]
fn two_non_native_formats_then_finalize() {
    let file = f(48_000, 2);
    let (_dir, p) = tmp();
    let mut sink = WavSink::create(&p, file, WriteOptions::default()).unwrap();
    for c in tone(16_000, 300.0, 0.5, 1).chunks(160) {
        sink.write_converting(&fr(c.to_vec(), f(16_000, 1)))
            .unwrap();
    }
    for c in tone(44_100, 300.0, 0.5, 2).chunks(882) {
        sink.write_converting(&fr(c.to_vec(), f(44_100, 2)))
            .unwrap();
    }
    for c in tone(8_000, 300.0, 0.5, 6).chunks(480) {
        sink.write_converting(&fr(c.to_vec(), f(8_000, 6))).unwrap();
    }
    let sw = sink.samples_written();
    let p2 = sink.finalize().unwrap();
    let (_, s) = read(&p2);
    assert_eq!(s.len(), 3 * 24_000 * 2, "1.5 s at 48 kHz stereo");
    assert!(
        sw < s.len() as u64,
        "samples_written before finalize excludes the tail: {sw}"
    );
    // last 10 ms carry signal (tail drained, not zero-padded silence)
    let last = &s[s.len() - 960..];
    assert!(last.iter().any(|&x| x.abs() > 1_000));
}

/// many alternations keep real time within rounding.
#[test]
fn rapid_alternation_keeps_duration() {
    let file = f(48_000, 1);
    let (_dir, p) = tmp();
    let mut sink = WavSink::create(&p, file, WriteOptions::default()).unwrap();
    for _ in 0..200 {
        sink.write_converting(&fr(vec![0.1; 480], file)).unwrap();
        sink.write_converting(&fr(vec![0.1; 440], f(44_100, 1)))
            .unwrap(); // 9.977 ms
    }
    let p2 = sink.finalize().unwrap();
    let (_, s) = read(&p2);
    let ideal = 200.0 * (480.0 + 440.0 * 48_000.0 / 44_100.0);
    assert!(
        (s.len() as f64 - ideal).abs() <= 200.0,
        "drift {}",
        s.len() as f64 - ideal
    );
}

/// an invalid frame format errors, keeps the flushed audio, and the sink stays usable.
#[test]
fn invalid_frame_keeps_previous_audio() {
    let file = f(48_000, 1);
    let (_dir, p) = tmp();
    let mut sink = WavSink::create(&p, file, WriteOptions::default()).unwrap();
    sink.write_converting(&fr(vec![0.2; 1_600], f(16_000, 1)))
        .unwrap();
    assert!(sink.write_converting(&fr(vec![0.2; 10], f(0, 1))).is_err());
    assert_eq!(
        sink.samples_written(),
        4_800,
        "16 kHz run flushed before the error"
    );
    sink.write_converting(&fr(vec![0.2; 1_600], f(16_000, 1)))
        .unwrap();
    sink.write_converting(&fr(vec![0.2; 4_800], file)).unwrap();
    let p2 = sink.finalize().unwrap();
    assert_eq!(read(&p2).1.len(), 3 * 4_800);
}

/// Dropping the sink without `finalize` still writes the audio the
/// converter holds.
#[test]
fn drop_without_finalize_keeps_the_converted_tail() {
    let file = f(48_000, 1);
    let (_dir, p) = tmp();
    {
        let mut sink = WavSink::create(&p, file, WriteOptions::default()).unwrap();
        sink.write_converting(&fr(vec![0.2; 16_000], f(16_000, 1)))
            .unwrap();
    }
    assert_eq!(read(&p).1.len(), 48_000, "1 s in, 1 s on disk");
}

/// non-native frames identical to the file's rate but other channel count pass through without resampling.
#[test]
fn same_rate_other_channels() {
    let file = f(48_000, 2);
    let (_dir, p) = tmp();
    let mut sink = WavSink::create(&p, file, WriteOptions::default()).unwrap();
    sink.write_converting(&fr(vec![0.5; 480], f(48_000, 1)))
        .unwrap();
    assert_eq!(sink.samples_written(), 960);
    sink.write_converting(&fr(vec![0.25; 960 * 3], f(48_000, 6)))
        .unwrap();
    let p2 = sink.finalize().unwrap();
    assert_eq!(read(&p2).1.len(), 960 + 960);
}

/// block size never changes FormatConverter output.
#[test]
fn converter_block_invariance() {
    let x = tone(44_100, 523.0, 1.0, 2);
    let mut a = FormatConverter::new(f(48_000, 2)).unwrap();
    let mut whole = Vec::new();
    a.push(&x, f(44_100, 2), &mut whole).unwrap();
    a.flush(&mut whole).unwrap();
    let mut b = FormatConverter::new(f(48_000, 2)).unwrap();
    let mut parts = Vec::new();
    for c in x.chunks(2 * 137) {
        b.push(c, f(44_100, 2), &mut parts).unwrap();
    }
    b.flush(&mut parts).unwrap();
    assert_eq!(whole.len(), 96_000);
    assert_eq!(whole, parts);
    // reused after flush equals fresh
    let mut again = Vec::new();
    b.push(&x, f(44_100, 2), &mut again).unwrap();
    b.flush(&mut again).unwrap();
    assert_eq!(again, whole);
}

#[test]
fn converter_rejects_invalid_target() {
    assert!(FormatConverter::new(f(0, 2)).is_err());
    assert!(FormatConverter::new(f(48_000, 0)).is_err());
}

/// cache is reset between calls and after errors, any order of formats.
#[test]
fn adapter_matches_fresh_in_any_order() {
    let ad = RubatoResamplerAdapter::new();
    let cases: Vec<(Vec<f32>, AudioFormat)> = vec![
        (tone(48_000, 440.0, 0.25, 1), f(48_000, 1)),
        (tone(16_000, 440.0, 0.25, 1), f(16_000, 1)),
        (vec![], f(48_000, 1)),
        (tone(16_000, 440.0, 0.25, 2), f(16_000, 2)),
        (tone(44_100, 440.0, 0.25, 2), f(44_100, 2)),
        (vec![1.0], f(48_000, 1)),
        (tone(8_000, 440.0, 0.25, 1), f(8_000, 1)),
        (tone(48_000, 440.0, 0.25, 1), f(48_000, 1)),
        (tone(48_000, 440.0, 0.25, 1), f(48_000, 1)),
    ];
    for (i, (x, fmt)) in cases.iter().enumerate() {
        assert_eq!(
            ad.to_whisper(x, *fmt).unwrap(),
            resample_to_whisper(x, *fmt).unwrap(),
            "case {i}"
        );
        if i == 4 {
            assert!(ad.to_whisper(&[0.0; 4], f(0, 1)).is_err());
            assert!(ad.to_whisper(&[0.0; 4], f(16_000, 0)).is_err());
        }
    }
}

/// S6c / risk 4: shared Arc<dyn Resampler>, concurrent callers all get exact results (cache or fallback).
#[test]
fn concurrent_callers_get_exact_results() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<RubatoResamplerAdapter>();
    let ad: Arc<dyn Resampler> = Arc::new(RubatoResamplerAdapter::new());
    let fmts = [f(48_000, 1), f(44_100, 2), f(16_000, 1), f(22_050, 1)];
    let inputs: Vec<(Vec<f32>, AudioFormat)> = fmts
        .iter()
        .map(|&fm| (tone(fm.sample_rate_hz, 330.0, 0.2, fm.channels), fm))
        .collect();
    let expected: Vec<Vec<f32>> = inputs
        .iter()
        .map(|(x, fm)| resample_to_whisper(x, *fm).unwrap())
        .collect();
    let inputs = Arc::new(inputs);
    let expected = Arc::new(expected);
    let hs: Vec<_> = (0..8)
        .map(|t| {
            let ad = ad.clone();
            let inp = inputs.clone();
            let exp = expected.clone();
            std::thread::spawn(move || {
                for k in 0..40 {
                    let i = (t + k) % 4;
                    assert_eq!(
                        ad.to_whisper(&inp[i].0, inp[i].1).unwrap(),
                        exp[i],
                        "thread {t} iter {k}"
                    );
                }
            })
        })
        .collect();
    for h in hs {
        h.join().unwrap();
    }
}
