//! Memory and time of the VBx re-segmentation at the embedding-log caps.
//!
//! ADR 0001 (Q3) targets about 35 MB for the log plus about 35 MB of
//! transient memory at stop, whatever the meeting length. This test fills
//! the log to both caps (16 384 entries, 16 384 dense windows) with a
//! cheap stub embedder and measures the bytes the end-of-session refine
//! allocates on top of the log.
//!
//! It is ignored by default: the worst case takes minutes in a debug
//! build. Run it with
//! `cargo test -p echo-diarize --release --test resegment_memory -- --ignored --nocapture`.
//! The counting allocator is process-wide, so this file holds one test.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use echo_diarize::{
    EmbedderFamily, OfflineRefineMode, OnlineClusterConfig, OnlineDiarizer, SpeakerEmbedder,
    EMBEDDING_LOG_CAP,
};
use echo_domain::{ChunkContext, Diarizer, DomainError, RefineBudget, RefineOutcome, Sample};

struct Counting;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// SAFETY: forwards every call to the system allocator unchanged and only
// keeps byte counters on the side.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: same contract as the caller's.
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            let now = CURRENT.fetch_add(layout.size(), Ordering::SeqCst) + layout.size();
            PEAK.fetch_max(now, Ordering::SeqCst);
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: same contract as the caller's.
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            let now = CURRENT.fetch_add(layout.size(), Ordering::SeqCst) + layout.size();
            PEAK.fetch_max(now, Ordering::SeqCst);
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: same contract as the caller's.
        unsafe { System.dealloc(ptr, layout) };
        CURRENT.fetch_sub(layout.size(), Ordering::SeqCst);
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// 192-dim embedder with six "voices": a voice direction plus a small
/// deterministic per-call jitter.
struct VoiceEmbedder {
    calls: u64,
}

impl SpeakerEmbedder for VoiceEmbedder {
    fn sample_rate_hz(&self) -> u32 {
        16_000
    }

    fn dim(&self) -> usize {
        192
    }

    fn embed(&mut self, samples: &[Sample]) -> Result<Option<Vec<f32>>, DomainError> {
        self.calls += 1;
        let voice = (samples[0] * 10.0) as usize % 6;
        let mut v: Vec<f32> = (0..192)
            .map(|i| {
                let h = (self.calls.wrapping_mul(2_654_435_761) ^ i as u64) % 1_000;
                h as f32 / 1_000.0 * 0.3
            })
            .collect();
        for x in &mut v[voice * 32..voice * 32 + 32] {
            *x += 1.0;
        }
        Ok(Some(v))
    }
}

#[tokio::test]
#[ignore = "worst case at both log caps; run in release (see module docs)"]
async fn resegment_at_the_log_caps_stays_within_the_transient_budget() {
    let family = EmbedderFamily::CamPlusPlusMultilingual;
    let mut d = OnlineDiarizer::new(
        Box::new(VoiceEmbedder { calls: 0 }),
        OnlineClusterConfig {
            similarity_threshold: 0.5,
            max_speakers: 8,
            ..OnlineClusterConfig::default()
        },
    )
    .with_plda(family.plda().expect("bundled PLDA"))
    .with_vbx_config(family.vbx_config())
    .with_dense_windows(true)
    .with_refine_mode(OfflineRefineMode::Resegment);

    // 4.5 s chunks (three dense windows each), voices taking turns.
    let chunks: Vec<Vec<Sample>> = (0..6)
        .map(|v| vec![v as f32 / 10.0 + 0.01; 72_000])
        .collect();
    for i in 0..(EMBEDDING_LOG_CAP as u32 + 64) {
        let ctx = ChunkContext {
            chunk_index: i,
            offset_ms: i * 4_500,
            epoch: 0,
        };
        d.assign_spans_at(&chunks[(i as usize / 3) % 6], ctx)
            .await
            .unwrap();
    }

    let base = CURRENT.load(Ordering::SeqCst);
    PEAK.store(base, Ordering::SeqCst);
    let started = Instant::now();
    let outcome = d
        .refine_plan(RefineBudget {
            deadline: Instant::now() + Duration::from_secs(600),
        })
        .unwrap();
    let elapsed = started.elapsed();
    let transient = PEAK.load(Ordering::SeqCst) - base;
    println!(
        "log {:.1} MB, refine transient {:.1} MB in {:.2} s ({})",
        base as f64 / 1_048_576.0,
        transient as f64 / 1_048_576.0,
        elapsed.as_secs_f64(),
        match outcome {
            RefineOutcome::Unchanged => "unchanged",
            RefineOutcome::Merges(_) => "merges",
            RefineOutcome::Relabel(_) => "relabel",
        }
    );
    assert!(
        transient <= 35 * 1024 * 1024,
        "re-segmentation allocated {transient} bytes at the log caps"
    );
}
