//! End-to-end check that the **multilingual CAM++** embedder works
//! through the real diarization path the streaming pipeline drives:
//! `assign_spans` per chunk (segmenter → embed → cluster) followed by
//! the end-of-session `refine_offline` pass.
//!
//! This is the pipeline-level proof that the 192-dim multilingual export
//! (`campplus_zh_en_common_advanced.onnx`) loads and runs — the unit
//! test in `cam_plus_plus.rs` only covers the embedder in isolation.
//!
//! Gated on the model + segmenter + fixture being present, so CI without
//! the downloaded assets skips cleanly. Fetch them with:
//!   scripts/download-models.sh embed-multi
//!   scripts/download-models.sh segmenter
//! (the two-speaker fixture ships with the CAM++/ERes2Net downloads).

use std::path::PathBuf;

use echo_diarize::cam_plus_plus::CamPlusPlusEmbedder;
use echo_diarize::pyannote_segmenter::PyannoteSegmenter;
use echo_diarize::{OnlineClusterConfig, OnlineDiarizer};
use echo_domain::Diarizer;

fn workspace_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p
}

/// Load a mono 16 kHz WAV into f32 samples.
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
fn multilingual_camplus_runs_through_full_diarization_path() {
    let root = workspace_root();
    let model = root.join("models/embedder/campplus_zh_en_common_advanced.onnx");
    // Prefer the tract-compatible (InstanceNorm-decomposed) export; fall
    // back to the upstream one, which does NOT load in tract 0.22.
    let seg_tract = root.join("models/segmenter/pyannote_segmentation_3_tract.onnx");
    let seg_upstream = root.join("models/segmenter/pyannote_segmentation_3.onnx");
    let segmenter_model = if seg_tract.exists() {
        seg_tract
    } else {
        seg_upstream
    };
    let fixture = root.join("crates/echo-diarize/tests/fixtures/two_speakers_en.wav");

    if !model.exists() || !fixture.exists() {
        eprintln!(
            "skipping multilingual e2e: missing assets (model={}, fixture={}). \
             Run `scripts/download-models.sh embed-multi`.",
            model.exists(),
            fixture.exists()
        );
        return;
    }

    let embedder = CamPlusPlusEmbedder::new(&model).expect("load multilingual CAM++");

    // Mirror the production CAM++ cluster tuning: split at 0.85, online
    // merge disabled (the offline pass handles merging).
    let mut diarizer = OnlineDiarizer::new(
        Box::new(embedder),
        OnlineClusterConfig {
            similarity_threshold: 0.85,
            merge_threshold: 1.0,
            ..Default::default()
        },
    )
    .with_offline_refinement();

    // The pyannote segmenter is best-effort — exactly as `build_diarizer`
    // treats it. NOTE: on tract 0.22 this export fails to load
    // (`InstanceNormalization` node), so in practice the diarizer runs
    // the whole-chunk path. We attach it if (and only if) it loads, and
    // record which path this run actually exercised.
    let segmenter_active = if segmenter_model.exists() {
        match PyannoteSegmenter::new(&segmenter_model) {
            Ok(seg) => {
                diarizer = diarizer.with_segmenter(Box::new(seg));
                true
            }
            Err(e) => {
                eprintln!("pyannote segmenter did not load ({e}); running whole-chunk path");
                false
            }
        }
    } else {
        false
    };

    let audio = load_mono_16k(&fixture);
    assert!(audio.len() >= 6 * 16_000, "fixture too short for the test");

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    // Feed sequential ~3 s windows through assign_spans — exactly the
    // call the streaming pipeline makes per chunk.
    let win = 3 * 16_000;
    let mut total_spans = 0usize;
    for chunk in audio.chunks(win) {
        if chunk.len() < 16_000 {
            break; // skip a sub-1s tail
        }
        let spans = rt
            .block_on(diarizer.assign_spans(chunk))
            .expect("assign_spans must not error with the 192-dim model");
        for s in &spans {
            // Every span's speaker must exist in the diarizer's view —
            // proves the id/slot bookkeeping is consistent.
            assert!(
                diarizer.speakers().iter().any(|sp| sp.id == s.speaker_id),
                "span references unknown speaker id"
            );
        }
        total_spans += spans.len();
    }

    let speakers = diarizer.speakers();
    assert!(
        !speakers.is_empty() && total_spans > 0,
        "expected the multilingual embedder to label at least one span (speakers={}, spans={})",
        speakers.len(),
        total_spans
    );

    // The end-of-session offline refinement must run without panicking
    // and return a (possibly empty) list of merges. Call the *trait*
    // method (disambiguated from the inherent `refine_offline(config)`)
    // because that's the one the streaming pipeline invokes.
    let merges = Diarizer::refine_offline(&diarizer);

    eprintln!(
        "multilingual e2e OK: segmenter_active={} · {} speaker(s) · {} span(s) · {} offline merge(s)",
        segmenter_active,
        speakers.len(),
        total_spans,
        merges.len()
    );
}
