//! Verifies that the InstanceNorm-decomposed pyannote segmenter export
//! (`pyannote_segmentation_3_tract.onnx`, produced by
//! `scripts/patch-pyannote-segmenter.py`) actually loads and runs in the
//! pure-Rust tract engine — the upstream sherpa export does not
//! (`InstanceNormalization` fails `into_optimized()` on tract 0.22).
//!
//! Gated on the patched file being present.

use std::path::PathBuf;

use echo_diarize::pyannote_segmenter::PyannoteSegmenter;
use echo_domain::Segmenter;

fn workspace_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p
}

#[test]
fn patched_segmenter_loads_and_runs_in_tract() {
    let patched = workspace_root().join("models/segmenter/pyannote_segmentation_3_tract.onnx");
    if !patched.exists() {
        eprintln!(
            "skipping: patched segmenter absent at {}. Produce it with \
             `python3 scripts/patch-pyannote-segmenter.py \
             models/segmenter/pyannote_segmentation_3.onnx {}`.",
            patched.display(),
            patched.display()
        );
        return;
    }

    // The upstream export fails here; the patched one must succeed.
    let mut seg =
        PyannoteSegmenter::new(&patched).expect("patched pyannote segmenter must load in tract");

    // Run it on a real-length chunk of silence: must not error and must
    // return a (possibly empty) segment list.
    let silence = vec![0.0_f32; echo_diarize::pyannote_segmenter::PYANNOTE_CHUNK_SAMPLES];
    let segments = seg.segment(&silence).expect("segment() must not error");
    eprintln!(
        "patched segmenter OK: loaded in tract, produced {} segment(s) on silence",
        segments.len()
    );
}
