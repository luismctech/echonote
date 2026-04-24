//! Pre-allocated sample buffer pool for the streaming pipeline.
//!
//! Recycles `Vec<Sample>` allocations across chunks so the hot loop
//! does not hit the allocator on every 5-second window.

use echo_domain::Sample;

/// Maximum number of idle buffers retained.
const MAX_IDLE: usize = 4;

/// A stack-based pool of reusable `Vec<Sample>` buffers.
pub(super) struct SamplePool {
    idle: Vec<Vec<Sample>>,
    capacity: usize,
}

impl SamplePool {
    /// Create a pool.  `capacity` is the pre-allocation hint (samples).
    pub fn new(capacity: usize) -> Self {
        Self {
            idle: Vec::with_capacity(MAX_IDLE),
            capacity,
        }
    }

    /// Take a buffer from the pool (or allocate a fresh one).
    /// The returned `Vec` is always empty but retains capacity.
    pub fn checkout(&mut self) -> Vec<Sample> {
        self.idle
            .pop()
            .unwrap_or_else(|| Vec::with_capacity(self.capacity))
    }

    /// Return a buffer to the pool for reuse.
    pub fn checkin(&mut self, mut buf: Vec<Sample>) {
        buf.clear();
        if self.idle.len() < MAX_IDLE {
            self.idle.push(buf);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkout_returns_empty_with_capacity() {
        let mut pool = SamplePool::new(1024);
        let buf = pool.checkout();
        assert!(buf.is_empty());
        assert!(buf.capacity() >= 1024);
    }

    #[test]
    fn checkin_reuses_allocation() {
        let mut pool = SamplePool::new(1024);
        let mut buf = pool.checkout();
        buf.extend_from_slice(&[1.0; 512]);
        let ptr = buf.as_ptr();
        pool.checkin(buf);

        let reused = pool.checkout();
        assert!(reused.is_empty());
        assert_eq!(reused.as_ptr(), ptr, "should reuse same allocation");
    }

    #[test]
    fn pool_caps_idle_count() {
        let mut pool = SamplePool::new(64);
        // Check in more than MAX_IDLE buffers
        let bufs: Vec<_> = (0..MAX_IDLE + 2).map(|_| pool.checkout()).collect();
        for b in bufs {
            pool.checkin(b);
        }
        assert!(pool.idle.len() <= MAX_IDLE);
    }
}
