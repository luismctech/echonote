//! Peak memory of the merge-only refine at the embedding-log cap.
//!
//! ADR 0001 (Q3) bounds the end-of-session transient memory to about
//! 35 MB whatever the meeting length. A full `n x n` similarity matrix at
//! the 16 384-entry cap would take about 1 GB, so this test counts the
//! bytes the refine allocates and fails well before that.
//!
//! The counting allocator is process-wide, so this file holds a single
//! test: nothing else allocates concurrently while it measures.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use echo_diarize::{refine_speakers, OfflineRefineConfig, EMBEDDING_LOG_CAP};
use echo_domain::Speaker;

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

#[test]
fn merge_only_refine_at_the_log_cap_stays_within_the_transient_budget() {
    const DIM: usize = 8;
    let speakers: Vec<Speaker> = (0..4).map(Speaker::anonymous).collect();
    let records: Vec<(Vec<f32>, _)> = (0..EMBEDDING_LOG_CAP)
        .map(|i| {
            let s = i % speakers.len();
            let mut v = vec![0.0_f32; DIM];
            v[s] = 1.0;
            v[DIM - 1] = (i % 7) as f32 * 0.01;
            (v, speakers[s].id)
        })
        .collect();

    let base = CURRENT.load(Ordering::SeqCst);
    PEAK.store(base, Ordering::SeqCst);
    let merges = refine_speakers(
        &records,
        &speakers,
        OfflineRefineConfig {
            merge_threshold: 0.5,
            min_embeddings: 2,
        },
    );
    let transient = PEAK.load(Ordering::SeqCst) - base;

    assert!(merges.is_empty(), "orthogonal speakers must not merge");
    assert!(
        transient <= 35 * 1024 * 1024,
        "merge-only refine allocated {transient} bytes at the {EMBEDDING_LOG_CAP}-entry cap"
    );
}
