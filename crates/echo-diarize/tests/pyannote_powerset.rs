//! Model-gated check that the pyannote segmenter decodes the powerset
//! head correctly: on a real two-speaker recording it must return
//! speaker spans (the previous log-prob thresholding never fired).

use std::collections::BTreeSet;
use std::path::PathBuf;

use echo_diarize::pyannote_segmenter::PyannoteSegmenter;
use echo_domain::Segmenter;

fn workspace_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p
}

fn load_mono_16k(path: &PathBuf) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("open fixture wav");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16_000, "fixture must be 16 kHz");
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1_i32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f32 / max)
                .collect()
        }
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
    };
    if spec.channels == 1 {
        raw
    } else {
        raw.chunks_exact(spec.channels as usize)
            .map(|c| c.iter().sum::<f32>() / c.len() as f32)
            .collect()
    }
}

#[test]
fn segmenter_returns_spans_on_two_speaker_fixture() {
    let root = workspace_root();
    let model = root.join("models/segmenter/pyannote_segmentation_3_tract.onnx");
    let fixture = root.join("crates/echo-diarize/tests/fixtures/two_speakers_en.wav");
    if !model.exists() || !fixture.exists() {
        eprintln!(
            "skipping: missing assets (model={}, fixture={})",
            model.exists(),
            fixture.exists()
        );
        return;
    }

    let mut seg = PyannoteSegmenter::new(&model).expect("patched segmenter must load");
    let audio = load_mono_16k(&fixture);

    // Mirror the streaming pipeline: feed consecutive 5 s windows.
    let mut speakers = BTreeSet::new();
    let mut total = 0usize;
    for window in audio.chunks(80_000) {
        if window.len() < 16_000 {
            break;
        }
        let segs = seg.segment(window).expect("segment() must not error");
        total += segs.len();
        speakers.extend(segs.iter().map(|s| s.local_speaker));
    }

    eprintln!("powerset decode: {total} segment(s), local speakers {speakers:?}");
    assert!(
        total > 0,
        "segmenter must return spans on real two-speaker speech"
    );
    assert!(
        speakers.len() >= 2,
        "two-speaker fixture should yield at least two local speakers, got {speakers:?}"
    );
}
