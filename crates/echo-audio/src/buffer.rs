//! Pre-allocated audio buffer pool for the streaming pipeline.
//!
//! Every 5-second chunk in the capture loop needs a `Vec<f32>` to hold
//! the raw samples extracted from the ring buffer. Without pooling each
//! chunk allocates ~320 KB (80 000 samples × 4 bytes at 16 kHz mono;
//! larger at 44.1 kHz stereo) and frees it a few hundred milliseconds
//! later — a classic allocator-thrash pattern.
//!
//! [`SamplePool`] keeps a small stack of previously-used buffers and
//! hands them back (cleared but capacity-preserved) on the next
//! [`checkout`](SamplePool::checkout). When the caller is done it
//! returns the buffer via [`checkin`](SamplePool::checkin), which
//! pushes it back into the pool automatically.
//!
//! The pool is intentionally **not** thread-safe (`!Send`/`!Sync`).
//! The streaming pipeline runs on a single Tokio task, so interior
//! mutability would add overhead for no benefit.

use echo_domain::Sample;

/// Maximum number of idle buffers kept in the pool.  Two is enough for
/// the streaming pipeline (one being filled while one is processed),
/// but we keep four to absorb small bursts.
const MAX_IDLE: usize = 4;

/// A pool of reusable `Vec<Sample>` buffers.
///
/// # Usage
///
/// ```ignore
/// let mut pool = SamplePool::new(80_000);
/// let mut buf = pool.checkout();         // zero-alloc if pool has spares
/// buf.extend_from_slice(&incoming);
/// // … process buf.as_slice() …
/// pool.checkin(buf);                     // returns capacity to pool
/// ```
pub struct SamplePool {
    /// Pre-allocated buffers waiting to be reused.
    idle: Vec<Vec<Sample>>,
    /// Capacity hint for new buffers (number of samples).
    capacity: usize,
}

impl SamplePool {
    /// Create a pool. `capacity` is the number of samples each buffer
    /// will be pre-allocated for (e.g. `sample_rate * chunk_secs *
    /// channels`).
    pub fn new(capacity: usize) -> Self {
        Self {
            idle: Vec::with_capacity(MAX_IDLE),
            capacity,
        }
    }

    /// Take a buffer from the pool, or allocate a fresh one if the
    /// pool is empty.  The returned buffer is always empty (`len == 0`)
    /// but retains its previous capacity.
    pub fn checkout(&mut self) -> Vec<Sample> {
        self.idle
            .pop()
            .unwrap_or_else(|| Vec::with_capacity(self.capacity))
    }

    /// Return a buffer to the pool for reuse.  If the pool is full the
    /// buffer is simply dropped (its memory is freed).
    pub fn checkin(&mut self, mut buf: Vec<Sample>) {
        buf.clear();
        if self.idle.len() < MAX_IDLE {
            self.idle.push(buf);
        }
        // else: silently drop — pool is full
    }

    /// Number of idle buffers currently in the pool.
    #[cfg(test)]
    pub fn idle_count(&self) -> usize {
        self.idle.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkout_returns_empty_buffer_with_capacity() {
        let mut pool = SamplePool::new(1024);
        let buf = pool.checkout();
        assert!(buf.is_empty());
        assert!(buf.capacity() >= 1024);
    }

    #[test]
    fn checkin_then_checkout_reuses_allocation() {
        let mut pool = SamplePool::new(1024);
        let mut buf = pool.checkout();
        buf.extend_from_slice(&[1.0; 512]);
        let ptr = buf.as_ptr();
        pool.checkin(buf);

        let reused = pool.checkout();
        assert!(reused.is_empty(), "checkin must clear the buffer");
        assert_eq!(reused.as_ptr(), ptr, "must reuse the same allocation");
    }

    #[test]
    fn pool_caps_at_max_idle() {
        let mut pool = SamplePool::new(64);
        for _ in 0..10 {
            let buf = pool.checkout();
            pool.checkin(buf);
        }
        // Even after 10 checkins the pool never exceeds MAX_IDLE
        assert!(pool.idle_count() <= super::MAX_IDLE);
    }

    #[test]
    fn checkout_from_empty_pool_allocates_fresh() {
        let mut pool = SamplePool::new(256);
        let a = pool.checkout();
        let b = pool.checkout();
        // Both are independent allocations
        assert_ne!(a.as_ptr(), b.as_ptr());
    }
}
