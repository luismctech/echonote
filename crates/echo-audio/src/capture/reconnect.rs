//! Retry policy shared by the cpal-backed adapters that rebuild their
//! stream after the device goes away (microphone, Windows loopback).

use std::time::Duration;

use tokio::sync::oneshot;
use tracing::{info, warn};

use echo_domain::DomainError;

/// How often and how long to retry opening a device.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Backoff {
    /// Attempts before giving up.
    pub attempts: u32,
    /// Wait before the first attempt; doubles after every failure.
    pub base: Duration,
    /// Longest wait between attempts.
    pub max: Duration,
    /// How often a stop request is checked while waiting.
    pub poll: Duration,
}

impl Backoff {
    /// Policy for audio devices: 10 attempts, 300 ms doubling up to 5 s
    /// (about 30 s in total), stop checked every 50 ms.
    pub(crate) const DEVICE: Self = Self {
        attempts: 10,
        base: Duration::from_millis(300),
        max: Duration::from_secs(5),
        poll: Duration::from_millis(50),
    };
}

/// True once stop was sent or its sender dropped.
pub(crate) fn stop_requested(stop_rx: &mut oneshot::Receiver<()>) -> bool {
    !matches!(stop_rx.try_recv(), Err(oneshot::error::TryRecvError::Empty))
}

/// Calls `attempt` (with its 1-based number) after each backoff wait
/// until it succeeds. Returns `None` when stop is requested or every
/// attempt failed.
pub(crate) fn retry<T>(
    policy: Backoff,
    stop_rx: &mut oneshot::Receiver<()>,
    mut attempt: impl FnMut(u32) -> Result<T, DomainError>,
) -> Option<T> {
    let mut delay = policy.base;
    for n in 1..=policy.attempts {
        if stop_requested(stop_rx) {
            return None;
        }
        info!(
            attempt = n,
            max = policy.attempts,
            ?delay,
            "reconnect attempt"
        );
        let mut remaining = delay;
        while remaining > Duration::ZERO {
            let step = remaining.min(policy.poll);
            std::thread::sleep(step);
            remaining = remaining.saturating_sub(step);
            if stop_requested(stop_rx) {
                return None;
            }
        }
        match attempt(n) {
            Ok(value) => return Some(value),
            Err(e) => {
                warn!(attempt = n, error = %e, "reconnect attempt failed");
                delay = (delay * 2).min(policy.max);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    const FAST: Backoff = Backoff {
        attempts: 4,
        base: Duration::from_millis(2),
        max: Duration::from_millis(5),
        poll: Duration::from_millis(1),
    };

    fn unavailable() -> DomainError {
        DomainError::AudioDeviceUnavailable("gone".into())
    }

    #[test]
    fn returns_the_first_successful_attempt() {
        let (_tx, mut rx) = oneshot::channel();
        let mut calls = Vec::new();
        let got = retry(FAST, &mut rx, |n| {
            calls.push(n);
            if n < 3 {
                Err(unavailable())
            } else {
                Ok("stream")
            }
        });
        assert_eq!(got, Some("stream"));
        assert_eq!(calls, vec![1, 2, 3]);
    }

    #[test]
    fn gives_up_after_the_last_attempt() {
        let (_tx, mut rx) = oneshot::channel();
        let mut calls = 0;
        let got: Option<()> = retry(FAST, &mut rx, |_| {
            calls += 1;
            Err(unavailable())
        });
        assert_eq!(got, None);
        assert_eq!(calls, FAST.attempts);
    }

    #[test]
    fn waits_longer_after_each_failure_up_to_the_cap() {
        let (_tx, mut rx) = oneshot::channel();
        let begin = Instant::now();
        let mut at = Vec::new();
        let _: Option<()> = retry(FAST, &mut rx, |_| {
            at.push(begin.elapsed());
            Err(unavailable())
        });
        // Waits: 2, 4, 5 (capped), 5 ms.
        let min_gaps = [2, 4, 5, 5].map(Duration::from_millis);
        let mut prev = Duration::ZERO;
        for (t, gap) in at.iter().zip(min_gaps) {
            assert!(
                *t - prev >= gap,
                "waited {:?}, expected at least {gap:?}",
                *t - prev
            );
            prev = *t;
        }
    }

    #[test]
    fn a_stop_sent_before_retrying_skips_every_attempt() {
        let (tx, mut rx) = oneshot::channel();
        tx.send(()).unwrap();
        let mut calls = 0;
        let got: Option<()> = retry(FAST, &mut rx, |_| {
            calls += 1;
            Ok(())
        });
        assert_eq!(got, None);
        assert_eq!(calls, 0);
    }

    #[test]
    fn a_stop_sent_while_retrying_ends_the_retries() {
        let (tx, mut rx) = oneshot::channel();
        let mut tx = Some(tx);
        let mut calls = 0;
        let got: Option<()> = retry(FAST, &mut rx, |_| {
            calls += 1;
            tx.take().unwrap().send(()).unwrap();
            Err(unavailable())
        });
        assert_eq!(got, None);
        assert_eq!(calls, 1);
    }

    #[test]
    fn a_dropped_stop_sender_counts_as_stop() {
        let (tx, mut rx) = oneshot::channel::<()>();
        drop(tx);
        assert!(stop_requested(&mut rx));
    }
}
