//! Dedicated inference worker thread.
//!
//! A `llama_cpp_2::context::LlamaContext` borrows its model and is not
//! `Send`, so a context that outlives one request has to live on a single
//! OS thread for its whole life. [`Worker`] is that thread: requests
//! ([`Job`]s) are queued and served one at a time against one persistent
//! [`Engine`] (the context), instead of building and dropping a ~1.2 GB KV
//! cache per request.
//!
//! ## Scheduling
//!
//! The queue has two classes ([`Priority`]). The worker always takes the
//! oldest queued [`Priority::Interactive`] job (chat) before any queued
//! [`Priority::Background`] job (summaries), FIFO within a class. A running
//! job is never preempted mid-decode: a map-reduce summary is many separate
//! `generate` calls, so a chat turn gets in between them and waits for at
//! most the call in flight.
//!
//! Trade-off: a continuous stream of chat turns can delay a queued summary
//! indefinitely (starvation). That is acceptable for a single-user desktop
//! app, where one person is either chatting or waiting for a summary, and
//! it keeps a single context (one KV cache) instead of a pool.
//!
//! ## Lifecycle
//!
//! * The thread is spawned lazily by the first [`Worker::submit`], and the
//!   engine is created lazily *inside* it by the [`EngineSource`], so the
//!   engine's borrow of the model is valid for as long as the engine
//!   exists.
//! * After `idle` without a job the engine is dropped (freeing the KV
//!   cache); the thread stays parked on the queue and the next job
//!   recreates the engine.
//! * Dropping the [`Worker`] raises a shutdown flag, wakes the queue and
//!   joins the thread. The job in hand stops at its next token (its
//!   [`Sink`] reports it as closed and its output is failed with a
//!   shutdown error), queued jobs fail fast with the same error, the
//!   engine is dropped and the thread exits. The join is therefore bounded
//!   by one decode step, and it must stay: a detached thread could still
//!   hold the GPU/KV resources while the process tears down its Metal
//!   device, which aborts the process (`ggml_metal_device_free`).
//! * A job whose consumer is already gone is skipped, and a consumer that
//!   goes away mid-job is observed through [`PieceSink::is_closed`] so
//!   decoding stops at the next token instead of blocking the queue.
//! * A consumer that is alive but stops draining its channel cannot block
//!   the queue either: sinks send through [`send_bounded`], which gives up
//!   after a stall timeout. The channel is full at that point, so the stream
//!   is marked ([`StallMark`]) and [`drain_stream`] ends it with an error
//!   once the consumer resumes: truncated output is never a clean end.
//! * Consumer-side code (the sinks) cannot kill the thread: every call into
//!   a sink is contained, and a [`DeathGuard`] covers any other panic by
//!   marking the queue dead and failing the queued jobs; the next
//!   [`Worker::submit`] starts a fresh thread.
//! * Prefix reuse across jobs is the engine's business (it owns the KV
//!   cache); the worker only guarantees jobs reach the engine one at a
//!   time, in order, and that a rebuilt engine starts with an empty cache.
//!
//! The worker is generic over [`EngineSource`]/[`Engine`] so the queueing,
//! idle-release, shutdown and cancellation policy is tested with a fake
//! engine, no model required.

use std::collections::VecDeque;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

use futures::stream::{BoxStream, StreamExt};
use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::mpsc::{Receiver, Sender};
use tracing::{debug, error};

use echo_domain::{ChatOptions, DomainError, GenerateOptions};

/// How long a sink waits for a consumer to make room in a full channel
/// before it gives up and cancels the job.
pub(crate) const CONSUMER_STALL_TIMEOUT: Duration = Duration::from_secs(30);

/// Poll interval while a channel is full.
const SEND_POLL: Duration = Duration::from_millis(5);

/// Error every job gets when the worker is dropped before serving it.
fn shutdown_error() -> DomainError {
    DomainError::LlmFailed("llm worker is shutting down".into())
}

/// Error for a job that can no longer be served because the worker thread
/// died unexpectedly.
fn worker_died_error() -> DomainError {
    DomainError::LlmFailed("llm worker stopped unexpectedly".into())
}

/// Whether the producer should keep generating after a piece was handed
/// to a [`PieceSink`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Flow {
    /// Keep decoding.
    Continue,
    /// A stop condition was met or the consumer is gone: end this job.
    Stop,
}

/// Cooperative cancellation flag raised when the worker shuts down.
///
/// Cloning shares the flag.
#[derive(Debug, Clone, Default)]
pub(crate) struct Cancel(Arc<AtomicBool>);

impl Cancel {
    /// `true` once cancellation was requested.
    pub(crate) fn is_set(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }

    pub(crate) fn set(&self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// Outcome of [`send_bounded`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SendOutcome {
    /// The item is in the channel.
    Sent,
    /// The receiver is gone.
    Closed,
    /// The job was cancelled (worker shutdown) while the channel was full.
    Cancelled,
    /// The channel stayed full for the whole stall timeout.
    Stalled,
}

/// Send `item` without ever blocking forever: `try_send` in a loop with
/// short sleeps, giving up when the receiver is gone, the job is
/// cancelled, or the channel stays full for `stall`.
///
/// This replaces `blocking_send`, which would park the worker (and, through
/// `Worker::drop`'s join, possibly a runtime thread) for as long as a stuck
/// consumer holds a full buffer.
pub(crate) fn send_bounded<T>(
    tx: &Sender<T>,
    mut item: T,
    cancel: &Cancel,
    stall: Duration,
) -> SendOutcome {
    let started = Instant::now();
    loop {
        match tx.try_send(item) {
            Ok(()) => return SendOutcome::Sent,
            Err(TrySendError::Closed(_)) => return SendOutcome::Closed,
            Err(TrySendError::Full(back)) => item = back,
        }
        if cancel.is_set() {
            return SendOutcome::Cancelled;
        }
        let waited = started.elapsed();
        if waited >= stall {
            return SendOutcome::Stalled;
        }
        thread::sleep(SEND_POLL.min(stall - waited));
    }
}

/// Streaming half of a [`Sink`]: a bounded channel to the consumer plus the
/// stall policy of [`send_bounded`], shared by the stream-shaped sinks.
pub(crate) struct Outlet<T> {
    tx: Sender<Result<T, DomainError>>,
    cancel: Cancel,
    stall: Duration,
    stalled: bool,
    mark: StallMark,
}

impl<T> Outlet<T> {
    /// Outlet over `tx` with the default [`CONSUMER_STALL_TIMEOUT`].
    pub(crate) fn new(tx: Sender<Result<T, DomainError>>) -> Self {
        Self {
            tx,
            cancel: Cancel::default(),
            stall: CONSUMER_STALL_TIMEOUT,
            stalled: false,
            mark: StallMark::default(),
        }
    }

    /// Override the stall timeout (tests use a short one).
    #[cfg(test)]
    pub(crate) fn with_stall(mut self, stall: Duration) -> Self {
        self.stall = stall;
        self
    }

    /// Adopt the worker's cancellation flag.
    pub(crate) fn bind_cancel(&mut self, cancel: Cancel) {
        self.cancel = cancel;
    }

    /// Forward one item; [`Flow::Stop`] when the consumer is gone, the job
    /// was cancelled, or the consumer stalled past the timeout (the job is
    /// then abandoned so the queue moves on).
    pub(crate) fn push(&mut self, item: T) -> Flow {
        match send_bounded(&self.tx, Ok(item), &self.cancel, self.stall) {
            SendOutcome::Sent => Flow::Continue,
            SendOutcome::Closed | SendOutcome::Cancelled => Flow::Stop,
            SendOutcome::Stalled => {
                self.stalled = true;
                // The channel is still full, so the failure cannot travel
                // through it; the stream reads this flag once it drains.
                self.mark.store(true, Ordering::SeqCst);
                tracing::warn!(
                    stall_secs = self.stall.as_secs_f32(),
                    "stream consumer stopped draining; cancelling the llm job"
                );
                Flow::Stop
            }
        }
    }

    /// `true` once the receiver is gone.
    pub(crate) fn is_closed(&self) -> bool {
        self.tx.is_closed()
    }

    /// Handle for [`drain_stream`] to learn that the stream was cut off.
    pub(crate) fn stall_mark(&self) -> StallMark {
        Arc::clone(&self.mark)
    }

    /// Deliver `error` as the stream's final item. Skipped after a stall:
    /// the buffer is still full and waiting again would only delay the
    /// queue a second time.
    pub(crate) fn fail(&mut self, error: DomainError) {
        if self.stalled {
            tracing::warn!(%error, "dropping the final error of a stalled stream");
            return;
        }
        let _ = send_bounded(&self.tx, Err(error), &self.cancel, self.stall);
    }
}

/// Shared flag raised when an [`Outlet`] cuts a stalled consumer off.
pub(crate) type StallMark = Arc<AtomicBool>;

/// Turn the receiving half of an [`Outlet`] channel into the stream handed
/// to the caller.
///
/// A consumer that stalled past the timeout was cut off while the channel
/// was full, so nothing could tell it the output is incomplete. Once it
/// drains the buffered items, the stream ends with an explicit error
/// instead of a clean end, so truncated output is never mistaken for a
/// complete answer.
pub(crate) fn drain_stream<T: Send + 'static>(
    rx: Receiver<Result<T, DomainError>>,
    mark: StallMark,
) -> BoxStream<'static, Result<T, DomainError>> {
    futures::stream::unfold(Some(rx), move |state| {
        let mark = Arc::clone(&mark);
        async move {
            let mut rx = state?;
            match rx.recv().await {
                Some(item) => Some((item, Some(rx))),
                None if mark.load(Ordering::SeqCst) => Some((
                    Err(DomainError::LlmFailed(
                        "consumer stalled; output truncated".into(),
                    )),
                    None,
                )),
                None => None,
            }
        }
    })
    .boxed()
}

/// Sampling knobs for one request, shared by every entry point.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SamplingParams {
    /// Hard cap on generated tokens.
    pub max_tokens: u32,
    /// Sampling temperature; `0.0` selects greedy decoding.
    pub temperature: f32,
    /// Nucleus-sampling parameter.
    pub top_p: f32,
    /// RNG seed; a fixed default is used when `None`.
    pub seed: Option<u64>,
    /// Add a repetition penalty to the non-greedy sampler chain (chat).
    pub penalize_repeats: bool,
}

impl SamplingParams {
    /// Knobs of a one-shot generation request.
    pub(crate) fn from_generate(options: &GenerateOptions) -> Self {
        Self {
            max_tokens: options.max_tokens,
            temperature: options.temperature,
            top_p: options.top_p,
            seed: options.seed,
            penalize_repeats: false,
        }
    }

    /// Knobs of a chat turn, which also penalises repetition.
    pub(crate) fn from_chat(options: &ChatOptions) -> Self {
        Self {
            max_tokens: options.max_tokens,
            temperature: options.temperature,
            top_p: options.top_p,
            seed: options.seed,
            penalize_repeats: true,
        }
    }
}

/// The part of a job's output the decode loop talks to.
pub(crate) trait PieceSink {
    /// Handle one decoded text piece; stop decoding when it returns
    /// [`Flow::Stop`].
    fn on_piece(&mut self, piece: &str) -> Flow;

    /// `true` once nobody is listening any more (stream receiver or
    /// result future dropped) or the worker is shutting down. Polled
    /// between tokens and before a queued job starts.
    fn is_closed(&self) -> bool;
}

/// Receives the output of one job and owns its completion.
pub(crate) trait Sink: PieceSink + Send {
    /// Hand the sink the worker's cancellation flag before the job is
    /// served, so a bounded send can notice a shutdown. Sinks that never
    /// wait on a consumer may ignore it.
    fn bind_cancel(&mut self, _cancel: Cancel) {}

    /// Report how the job ended. Not called for jobs skipped because the
    /// sink was already closed.
    fn finish(self: Box<Self>, result: Result<(), DomainError>);
}

/// Scheduling class of a [`Job`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Priority {
    /// A person is waiting on the reply (chat).
    Interactive,
    /// Bulk work that can wait a turn (summaries, map-reduce steps).
    Background,
}

/// One queued request.
pub(crate) struct Job {
    /// Scheduling class.
    pub priority: Priority,
    /// Fully rendered prompt (chat template already applied).
    pub prompt: String,
    /// Sampling knobs.
    pub params: SamplingParams,
    /// Where the output goes.
    pub sink: Box<dyn Sink>,
}

/// A live inference context, used on the worker thread only.
pub(crate) trait Engine {
    /// Run one request to completion, feeding pieces to `sink`. Must poll
    /// [`PieceSink::is_closed`] between tokens.
    fn serve(
        &mut self,
        prompt: &str,
        params: &SamplingParams,
        sink: &mut dyn PieceSink,
    ) -> Result<(), DomainError>;
}

/// Factory for the engine, shared with the worker thread.
pub(crate) trait EngineSource: Send + Sync + 'static {
    /// Build a fresh engine. Called on the worker thread, and again after
    /// each idle release.
    fn create(&self) -> Result<Box<dyn Engine + '_>, DomainError>;
}

/// Jobs waiting for the worker, one FIFO per [`Priority`].
#[derive(Default)]
struct Pending {
    interactive: VecDeque<Job>,
    background: VecDeque<Job>,
    /// The worker thread died unexpectedly; cleared when a new one is
    /// spawned.
    dead: bool,
}

/// What the worker should do next.
enum Next {
    Job(Job),
    /// Nothing arrived within the idle period.
    Idle,
    Shutdown,
}

/// Two-class job queue shared between submitters and the worker thread.
struct Queue {
    pending: Mutex<Pending>,
    ready: Condvar,
    cancel: Cancel,
}

impl Queue {
    fn new() -> Self {
        Self {
            pending: Mutex::new(Pending::default()),
            ready: Condvar::new(),
            cancel: Cancel::default(),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Pending> {
        self.pending.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Queue `job` in its priority class. A dead queue (its thread died
    /// and drained it) refuses the job and hands it back, so the caller can
    /// start a new thread instead of stranding it. Checking and pushing
    /// under one lock is what closes the window in which the thread's
    /// [`DeathGuard`] could drain the queue between a liveness check and
    /// the push.
    fn push(&self, job: Job) -> Result<(), Job> {
        {
            let mut pending = self.lock();
            if pending.dead {
                return Err(job);
            }
            match job.priority {
                Priority::Interactive => pending.interactive.push_back(job),
                Priority::Background => pending.background.push_back(job),
            }
        }
        self.ready.notify_one();
        Ok(())
    }

    /// Block until a job is available (interactive first), `idle` passes
    /// (when given) or shutdown is requested.
    fn next(&self, idle: Option<Duration>) -> Next {
        let deadline = idle.map(|idle| Instant::now() + idle);
        let mut pending = self.lock();
        loop {
            if self.cancel.is_set() {
                return Next::Shutdown;
            }
            if let Some(job) = pending
                .interactive
                .pop_front()
                .or_else(|| pending.background.pop_front())
            {
                return Next::Job(job);
            }
            pending = match deadline {
                None => self
                    .ready
                    .wait(pending)
                    .unwrap_or_else(PoisonError::into_inner),
                Some(deadline) => {
                    let left = deadline.saturating_duration_since(Instant::now());
                    if left.is_zero() {
                        return Next::Idle;
                    }
                    self.ready
                        .wait_timeout(pending, left)
                        .unwrap_or_else(PoisonError::into_inner)
                        .0
                }
            };
        }
    }

    fn is_dead(&self) -> bool {
        self.lock().dead
    }

    /// Clear the dead mark before a replacement thread starts.
    fn revive(&self) {
        self.lock().dead = false;
    }

    /// The worker thread is unwinding: mark the queue dead and fail every
    /// queued job, so no caller waits for a thread that is gone.
    fn fail_dead(&self) {
        self.mark_dead();
        self.drain(worker_died_error);
    }

    /// Make [`Queue::push`] refuse every later job until [`Queue::revive`].
    /// Without this, a job pushed after the thread's last drain would wait
    /// for a thread that is gone. Callers pair it with a drain; the mark is
    /// set under the lock, so a push either lands before it (and the drain
    /// fails the job) or is refused.
    fn mark_dead(&self) {
        self.lock().dead = true;
    }

    /// Raise the shutdown flag and wake the worker. The flag is set before
    /// taking the lock so a worker about to wait still sees it.
    fn shutdown(&self) {
        self.cancel.set();
        drop(self.lock());
        self.ready.notify_all();
    }

    /// Fail every still-queued job with the shutdown error.
    fn fail_pending(&self) {
        self.drain(shutdown_error);
    }

    /// Remove every queued job (interactive first) and finish each with the
    /// error `error` builds. The sinks run outside the lock.
    fn drain(&self, error: fn() -> DomainError) {
        let jobs: Vec<Job> = {
            let mut pending = self.lock();
            let interactive = std::mem::take(&mut pending.interactive);
            let background = std::mem::take(&mut pending.background);
            interactive.into_iter().chain(background).collect()
        };
        for mut job in jobs {
            shielded("bind_cancel", || job.sink.bind_cancel(self.cancel.clone()));
            shielded("finish", move || job.sink.finish(Err(error())));
        }
    }
}

/// Run a call into a sink, containing a panic: sinks are consumer-side
/// code and must not be able to take the worker thread down. `None` when
/// it panicked.
fn shielded<R>(what: &str, call: impl FnOnce() -> R) -> Option<R> {
    panic::catch_unwind(AssertUnwindSafe(call))
        .map_err(|_| error!(call = what, "llm sink panicked"))
        .ok()
}

/// Backstop for a panic that escapes every [`shielded`]/`catch_unwind`
/// site of the thread body: while unwinding it marks the queue dead and
/// fails the queued jobs, which would otherwise wait for a thread that is
/// gone. The next [`Worker::submit`] then starts a clean thread.
struct DeathGuard<'a>(&'a Queue);

impl<'a> DeathGuard<'a> {
    fn new(queue: &'a Queue) -> Self {
        Self(queue)
    }
}

impl Drop for DeathGuard<'_> {
    fn drop(&mut self) {
        if thread::panicking() {
            error!("llm worker thread panicked; failing the queued jobs");
            self.0.fail_dead();
        }
    }
}

/// See [`Worker::before_push`].
#[cfg(test)]
type BeforePush = Box<dyn FnOnce(&Queue) + Send>;

/// Handle to the worker thread. Dropping it shuts the thread down.
pub(crate) struct Worker {
    source: Arc<dyn EngineSource>,
    idle: Duration,
    queue: Arc<Queue>,
    thread: Mutex<Option<thread::JoinHandle<()>>>,
    /// Test seam: runs once between the liveness check and the push, to
    /// force the interleaving where the thread dies in that window.
    #[cfg(test)]
    before_push: Mutex<Option<BeforePush>>,
}

impl Worker {
    /// Build a worker that releases its engine after `idle` without jobs.
    /// No thread is started until the first [`Self::submit`].
    pub(crate) fn new(source: Arc<dyn EngineSource>, idle: Duration) -> Self {
        Self {
            source,
            idle,
            queue: Arc::new(Queue::new()),
            thread: Mutex::new(None),
            #[cfg(test)]
            before_push: Mutex::new(None),
        }
    }

    /// Queue `job` in its priority class, starting the thread if it is not
    /// running (first use, or it died).
    ///
    /// The thread only exits on shutdown, so a finished handle (or a queue
    /// its [`DeathGuard`] marked dead) means it died unexpectedly; it is
    /// replaced, and a spawn failure is reported instead of retried. If it
    /// dies after that check, [`Queue::push`] refuses the job and the
    /// replacement is attempted once more.
    pub(crate) fn submit(&self, job: Job) -> Result<(), DomainError> {
        let mut thread = self.thread.lock().unwrap_or_else(PoisonError::into_inner);
        if thread.as_ref().is_none_or(thread::JoinHandle::is_finished) || self.queue.is_dead() {
            self.respawn(&mut thread)?;
        }
        #[cfg(test)]
        if let Some(hook) = self.before_push.lock().unwrap().take() {
            hook(&self.queue);
        }
        // The thread can still die between the check above and this push;
        // the queue then refuses the job and one more thread is started.
        if let Err(job) = self.queue.push(job) {
            self.respawn(&mut thread)?;
            self.queue.push(job).map_err(|_| worker_died_error())?;
        }
        Ok(())
    }

    /// Replace the (dead or missing) worker thread. Called with the thread
    /// lock held.
    fn respawn(&self, thread: &mut Option<thread::JoinHandle<()>>) -> Result<(), DomainError> {
        // A thread that died is at most a few instructions from done (its
        // guard has already run), so joining it is brief.
        if let Some(dead) = thread.take() {
            let _ = dead.join();
        }
        self.queue.revive();
        let source = Arc::clone(&self.source);
        let queue = Arc::clone(&self.queue);
        let idle = self.idle;
        let handle = thread::Builder::new()
            .name("echo-llm-worker".into())
            .spawn(move || run(&*source, &queue, idle))
            .map_err(|e| DomainError::LlmFailed(format!("spawn llm worker: {e}")))?;
        *thread = Some(handle);
        Ok(())
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.queue.shutdown();
        let thread = self
            .thread
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(thread) = thread {
            let _ = thread.join();
        }
        // Covers jobs queued behind a thread that died on its own.
        self.queue.fail_pending();
    }
}

/// Reports a worker shutdown to the engine as a closed consumer, so the
/// decode loop stops at its next token.
struct Guarded<'a> {
    inner: &'a mut dyn PieceSink,
    cancel: &'a Cancel,
}

impl PieceSink for Guarded<'_> {
    fn on_piece(&mut self, piece: &str) -> Flow {
        if self.cancel.is_set() {
            return Flow::Stop;
        }
        self.inner.on_piece(piece)
    }

    fn is_closed(&self) -> bool {
        self.cancel.is_set() || self.inner.is_closed()
    }
}

/// Worker thread body: serve jobs (interactive first) until shutdown, then
/// fail whatever is still queued.
fn run(source: &dyn EngineSource, queue: &Queue, idle: Duration) {
    // Declared first so it drops last, once the engine is already gone.
    let _guard = DeathGuard::new(queue);
    // Declared after `source` is borrowed so the engine drops first.
    let mut engine: Option<Box<dyn Engine + '_>> = None;
    loop {
        let mut job = match queue.next(engine.is_some().then_some(idle)) {
            Next::Job(job) => job,
            Next::Idle => {
                debug!("llm worker idle: releasing the context");
                engine = None;
                continue;
            }
            Next::Shutdown => break,
        };
        let bound = shielded("bind_cancel", || {
            job.sink.bind_cancel(queue.cancel.clone());
        });
        // A sink that panics while being inspected is treated as gone.
        if bound.is_none() || shielded("is_closed", || job.sink.is_closed()).unwrap_or(true) {
            debug!("skipping a queued llm job whose consumer is gone");
            continue;
        }
        if engine.is_none() {
            match source.create() {
                Ok(created) => engine = Some(created),
                Err(err) => {
                    shielded("finish", move || job.sink.finish(Err(err)));
                    continue;
                }
            }
        }
        let Some(eng) = engine.as_mut() else {
            continue;
        };
        let Job {
            priority: _,
            prompt,
            params,
            mut sink,
        } = job;
        // A panic in the engine must not take the queue down with it; the
        // context may be left in an unknown state, so it is rebuilt.
        let mut guarded = Guarded {
            inner: sink.as_mut(),
            cancel: &queue.cancel,
        };
        let outcome = panic::catch_unwind(AssertUnwindSafe(|| {
            eng.serve(&prompt, &params, &mut guarded)
        }));
        let result = match outcome {
            // An abort by shutdown ends the engine loop "successfully";
            // the consumer must not mistake the truncated output for a
            // complete one.
            Ok(Ok(())) if queue.cancel.is_set() => Err(shutdown_error()),
            Ok(result) => result,
            Err(_) => {
                error!("llm engine panicked; discarding the context");
                engine = None;
                Err(DomainError::LlmFailed("inference worker panicked".into()))
            }
        };
        shielded("finish", move || sink.finish(result));
    }
    drop(engine);
    queue.fail_pending();
    debug!("llm worker stopped");
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::mpsc::{channel, Receiver, Sender};
    use std::time::Instant;

    use super::*;

    /// Poll `cond` until it holds or two seconds pass.
    fn eventually(mut cond: impl FnMut() -> bool) -> bool {
        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            if cond() {
                return true;
            }
            thread::sleep(Duration::from_millis(5));
        }
        cond()
    }

    #[derive(Default)]
    struct Probe {
        created: AtomicUsize,
        dropped: AtomicUsize,
        served: Mutex<Vec<String>>,
    }

    struct FakeSource {
        probe: Arc<Probe>,
        /// Jobs wait for a message here before finishing, when set.
        gate: Mutex<Option<Receiver<()>>>,
        fail_create: AtomicBool,
    }

    impl FakeSource {
        fn new(probe: &Arc<Probe>) -> Self {
            Self {
                probe: Arc::clone(probe),
                gate: Mutex::new(None),
                fail_create: AtomicBool::new(false),
            }
        }
    }

    struct FakeEngine<'a> {
        source: &'a FakeSource,
    }

    impl EngineSource for FakeSource {
        fn create(&self) -> Result<Box<dyn Engine + '_>, DomainError> {
            if self.fail_create.load(Ordering::SeqCst) {
                return Err(DomainError::LlmFailed("create context: boom".into()));
            }
            self.probe.created.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(FakeEngine { source: self }))
        }
    }

    impl Drop for FakeEngine<'_> {
        fn drop(&mut self) {
            self.source.probe.dropped.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl Engine for FakeEngine<'_> {
        fn serve(
            &mut self,
            prompt: &str,
            _params: &SamplingParams,
            sink: &mut dyn PieceSink,
        ) -> Result<(), DomainError> {
            self.source.probe.served.lock().unwrap().push(prompt.into());
            if let Some(gate) = self.source.gate.lock().unwrap().take() {
                let _ = gate.recv_timeout(Duration::from_secs(2));
            }
            if prompt == "panic" {
                panic!("engine blew up");
            }
            if prompt.starts_with("spin") {
                // A long decode that only ends when told to stop.
                let deadline = Instant::now() + Duration::from_secs(5);
                while !sink.is_closed() && Instant::now() < deadline {
                    let _ = sink.on_piece("x");
                    thread::sleep(Duration::from_millis(1));
                }
                return Ok(());
            }
            for piece in ["a", "b", "c"] {
                if sink.on_piece(piece) == Flow::Stop {
                    break;
                }
            }
            Ok(())
        }
    }

    struct TestSink {
        pieces: Arc<Mutex<String>>,
        closed: Arc<AtomicBool>,
        done: Sender<Result<(), String>>,
    }

    impl PieceSink for TestSink {
        fn on_piece(&mut self, piece: &str) -> Flow {
            self.pieces.lock().unwrap().push_str(piece);
            Flow::Continue
        }
        fn is_closed(&self) -> bool {
            self.closed.load(Ordering::SeqCst)
        }
    }

    impl Sink for TestSink {
        fn finish(self: Box<Self>, result: Result<(), DomainError>) {
            let _ = self.done.send(result.map_err(|e| e.to_string()));
        }
    }

    struct Submitted {
        done: Receiver<Result<(), String>>,
        pieces: Arc<Mutex<String>>,
        closed: Arc<AtomicBool>,
    }

    impl Submitted {
        fn wait(&self) -> Result<(), String> {
            self.done
                .recv_timeout(Duration::from_secs(5))
                .expect("job finished")
        }
    }

    fn submit(worker: &Worker, prompt: &str) -> Submitted {
        submit_with(worker, prompt, Priority::Background)
    }

    fn submit_with(worker: &Worker, prompt: &str, priority: Priority) -> Submitted {
        let (done_tx, done) = channel();
        let pieces = Arc::new(Mutex::new(String::new()));
        let closed = Arc::new(AtomicBool::new(false));
        worker
            .submit(Job {
                priority,
                prompt: prompt.into(),
                params: SamplingParams {
                    max_tokens: 8,
                    temperature: 0.0,
                    top_p: 1.0,
                    seed: None,
                    penalize_repeats: false,
                },
                sink: Box::new(TestSink {
                    pieces: Arc::clone(&pieces),
                    closed: Arc::clone(&closed),
                    done: done_tx,
                }),
            })
            .expect("submit");
        Submitted {
            done,
            pieces,
            closed,
        }
    }

    fn worker(source: FakeSource, idle: Duration) -> Worker {
        Worker::new(Arc::new(source), idle)
    }

    #[test]
    fn jobs_are_served_in_submission_order_on_one_engine() {
        let probe = Arc::new(Probe::default());
        let w = worker(FakeSource::new(&probe), Duration::from_secs(60));
        let jobs: Vec<_> = ["one", "two", "three", "four"]
            .into_iter()
            .map(|p| submit(&w, p))
            .collect();
        for job in &jobs {
            assert_eq!(job.wait(), Ok(()));
            assert_eq!(*job.pieces.lock().unwrap(), "abc");
        }
        assert_eq!(
            *probe.served.lock().unwrap(),
            ["one", "two", "three", "four"]
        );
        assert_eq!(probe.created.load(Ordering::SeqCst), 1, "one engine only");
    }

    #[test]
    fn a_queued_job_whose_consumer_is_gone_is_skipped() {
        let probe = Arc::new(Probe::default());
        let source = FakeSource::new(&probe);
        let (release, gate) = channel();
        *source.gate.lock().unwrap() = Some(gate);
        let w = worker(source, Duration::from_secs(60));

        let first = submit(&w, "first");
        let abandoned = submit(&w, "abandoned");
        abandoned.closed.store(true, Ordering::SeqCst);
        let last = submit(&w, "last");
        release.send(()).unwrap();

        assert_eq!(first.wait(), Ok(()));
        assert_eq!(last.wait(), Ok(()));
        assert_eq!(*probe.served.lock().unwrap(), ["first", "last"]);
    }

    #[test]
    fn the_engine_is_released_after_the_idle_period_and_recreated_on_demand() {
        let probe = Arc::new(Probe::default());
        let w = worker(FakeSource::new(&probe), Duration::from_millis(40));

        assert_eq!(submit(&w, "one").wait(), Ok(()));
        assert!(
            eventually(|| probe.dropped.load(Ordering::SeqCst) == 1),
            "idle engine must be dropped"
        );

        assert_eq!(submit(&w, "two").wait(), Ok(()));
        assert_eq!(probe.created.load(Ordering::SeqCst), 2);
        assert!(eventually(|| probe.dropped.load(Ordering::SeqCst) == 2));
    }

    #[test]
    fn the_engine_is_kept_while_jobs_keep_arriving_within_the_idle_period() {
        let probe = Arc::new(Probe::default());
        let w = worker(FakeSource::new(&probe), Duration::from_secs(60));
        for p in ["one", "two", "three"] {
            assert_eq!(submit(&w, p).wait(), Ok(()));
        }
        assert_eq!(probe.created.load(Ordering::SeqCst), 1);
        assert_eq!(probe.dropped.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn dropping_the_worker_joins_the_thread_and_frees_the_engine() {
        let probe = Arc::new(Probe::default());
        let source = Arc::new(FakeSource::new(&probe));
        let weak_source = Arc::downgrade(&source);
        let w = Worker::new(source, Duration::from_secs(60));
        assert_eq!(submit(&w, "one").wait(), Ok(()));
        drop(w);

        // Synchronous: the drop returned only after the thread exited.
        assert_eq!(probe.dropped.load(Ordering::SeqCst), 1);
        assert!(weak_source.upgrade().is_none(), "source handle released");
    }

    #[test]
    fn an_engine_creation_failure_is_reported_and_the_next_job_retries() {
        let probe = Arc::new(Probe::default());
        let source = FakeSource::new(&probe);
        source.fail_create.store(true, Ordering::SeqCst);
        let source = Arc::new(source);
        let w = Worker::new(
            Arc::clone(&source) as Arc<dyn EngineSource>,
            Duration::from_secs(60),
        );

        let failed = submit(&w, "one").wait();
        assert!(failed.unwrap_err().contains("create context"));

        source.fail_create.store(false, Ordering::SeqCst);
        assert_eq!(submit(&w, "two").wait(), Ok(()));
    }

    #[test]
    fn a_panicking_engine_fails_its_job_and_the_worker_keeps_serving() {
        let probe = Arc::new(Probe::default());
        let w = worker(FakeSource::new(&probe), Duration::from_secs(60));

        let err = submit(&w, "panic").wait().unwrap_err();
        assert!(err.contains("panicked"), "{err}");

        assert_eq!(submit(&w, "after").wait(), Ok(()));
        assert_eq!(probe.created.load(Ordering::SeqCst), 2, "fresh engine");
    }

    /// Wait until the engine has started serving `prompt`.
    fn wait_until_serving(probe: &Probe, prompt: &str) {
        assert!(
            eventually(|| probe.served.lock().unwrap().iter().any(|p| p == prompt)),
            "{prompt} never started"
        );
    }

    #[test]
    fn an_interactive_job_is_served_before_queued_background_jobs() {
        let probe = Arc::new(Probe::default());
        let source = FakeSource::new(&probe);
        let (release, gate) = channel();
        *source.gate.lock().unwrap() = Some(gate);
        let w = worker(source, Duration::from_secs(60));

        let running = submit(&w, "bg-running");
        wait_until_serving(&probe, "bg-running");
        let queued_bg_one = submit(&w, "bg-queued-1");
        let queued_bg_two = submit(&w, "bg-queued-2");
        let chat_one = submit_with(&w, "chat-1", Priority::Interactive);
        let chat_two = submit_with(&w, "chat-2", Priority::Interactive);
        release.send(()).unwrap();

        for job in [
            &running,
            &queued_bg_one,
            &queued_bg_two,
            &chat_one,
            &chat_two,
        ] {
            assert_eq!(job.wait(), Ok(()));
        }
        // The running job is never preempted; interactive jobs then go
        // first (FIFO among themselves), background jobs keep their order.
        assert_eq!(
            *probe.served.lock().unwrap(),
            [
                "bg-running",
                "chat-1",
                "chat-2",
                "bg-queued-1",
                "bg-queued-2"
            ]
        );
    }

    #[test]
    fn the_idle_release_still_works_with_the_priority_queue() {
        let probe = Arc::new(Probe::default());
        let w = worker(FakeSource::new(&probe), Duration::from_millis(40));
        assert_eq!(
            submit_with(&w, "chat", Priority::Interactive).wait(),
            Ok(())
        );
        assert!(eventually(|| probe.dropped.load(Ordering::SeqCst) == 1));
        assert_eq!(submit(&w, "bg").wait(), Ok(()));
        assert_eq!(probe.created.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn dropping_the_worker_aborts_the_running_job_and_fails_the_queued_ones() {
        let probe = Arc::new(Probe::default());
        let w = worker(FakeSource::new(&probe), Duration::from_secs(60));
        let running = submit(&w, "spin");
        wait_until_serving(&probe, "spin");
        let queued_bg = submit(&w, "queued-bg");
        let queued_chat = submit_with(&w, "queued-chat", Priority::Interactive);

        let started = Instant::now();
        drop(w);
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "drop must not wait for the running job: {:?}",
            started.elapsed()
        );

        for job in [&running, &queued_bg, &queued_chat] {
            let err = job.wait().unwrap_err();
            assert!(err.contains("shutting down"), "{err}");
        }
        assert_eq!(*probe.served.lock().unwrap(), ["spin"], "queued never ran");
    }

    /// Sink that streams through an [`Outlet`] nobody reads, so it stalls
    /// as soon as the one-slot buffer is full.
    struct StuckStreamSink {
        out: Outlet<String>,
        done: Sender<Result<(), String>>,
    }

    impl PieceSink for StuckStreamSink {
        fn on_piece(&mut self, piece: &str) -> Flow {
            self.out.push(piece.to_string())
        }
        fn is_closed(&self) -> bool {
            self.out.is_closed()
        }
    }

    impl Sink for StuckStreamSink {
        fn bind_cancel(&mut self, cancel: Cancel) {
            self.out.bind_cancel(cancel);
        }
        fn finish(self: Box<Self>, result: Result<(), DomainError>) {
            let _ = self.done.send(result.map_err(|e| e.to_string()));
        }
    }

    type StuckReceiver = tokio::sync::mpsc::Receiver<Result<String, DomainError>>;

    /// Queue a job whose consumer never drains its one-slot channel; the
    /// receiver is returned so the channel stays open.
    fn submit_stuck(
        worker: &Worker,
        stall: Duration,
    ) -> (Receiver<Result<(), String>>, StuckReceiver) {
        let (done_tx, done) = channel();
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        worker
            .submit(Job {
                priority: Priority::Interactive,
                prompt: "stuck".into(),
                params: SamplingParams::from_generate(&GenerateOptions::default()),
                sink: Box::new(StuckStreamSink {
                    out: Outlet::new(tx).with_stall(stall),
                    done: done_tx,
                }),
            })
            .expect("submit");
        (done, rx)
    }

    #[test]
    fn a_stalled_consumer_is_cut_off_and_the_queue_moves_on() {
        let probe = Arc::new(Probe::default());
        let w = worker(FakeSource::new(&probe), Duration::from_secs(60));
        let (stuck_done, _rx_kept_open) = submit_stuck(&w, Duration::from_millis(50));
        let next = submit(&w, "next");

        // The stuck job ends (its consumer, if it resumed, learns of the
        // truncation through the stream mark) and the following job is served instead of waiting forever.
        assert_eq!(
            stuck_done
                .recv_timeout(Duration::from_secs(5))
                .expect("ended"),
            Ok(())
        );
        assert_eq!(next.wait(), Ok(()));
    }

    #[test]
    fn dropping_the_worker_does_not_wait_for_a_stalled_consumer() {
        let probe = Arc::new(Probe::default());
        let w = worker(FakeSource::new(&probe), Duration::from_secs(60));
        let (stuck_done, _rx_kept_open) = submit_stuck(&w, Duration::from_secs(60));
        wait_until_serving(&probe, "stuck");
        // Let the sink fill its buffer and start waiting.
        thread::sleep(Duration::from_millis(100));

        let started = Instant::now();
        drop(w);
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "drop waited for the consumer: {:?}",
            started.elapsed()
        );
        let err = stuck_done
            .recv_timeout(Duration::from_secs(1))
            .expect("ended")
            .unwrap_err();
        assert!(err.contains("shutting down"), "{err}");
    }

    #[test]
    fn send_bounded_reports_each_way_a_send_can_end() {
        let cancel = Cancel::default();
        let (tx, rx) = tokio::sync::mpsc::channel::<u8>(1);
        let short = Duration::from_millis(20);
        assert_eq!(send_bounded(&tx, 1, &cancel, short), SendOutcome::Sent);
        assert_eq!(send_bounded(&tx, 2, &cancel, short), SendOutcome::Stalled);
        cancel.set();
        assert_eq!(
            send_bounded(&tx, 3, &cancel, Duration::from_secs(60)),
            SendOutcome::Cancelled
        );
        drop(rx);
        assert_eq!(send_bounded(&tx, 4, &cancel, short), SendOutcome::Closed);
    }

    #[tokio::test]
    async fn a_consumer_that_resumes_after_a_stall_sees_an_error_not_a_clean_end() {
        use futures::StreamExt;

        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let mut out = Outlet::new(tx).with_stall(Duration::from_millis(20));
        let mut stream = drain_stream(rx, out.stall_mark());
        assert_eq!(out.push("a".to_string()), Flow::Continue);
        // The buffer is full and nobody reads: the job is cut off.
        assert_eq!(out.push("b".to_string()), Flow::Stop);
        drop(out);

        // The consumer wakes up: it gets what was buffered, then the error.
        assert_eq!(stream.next().await.unwrap().unwrap(), "a");
        let err = stream.next().await.unwrap().unwrap_err();
        assert!(err.to_string().contains("stalled"), "{err}");
        assert!(stream.next().await.is_none(), "error is the last item");
    }

    #[tokio::test]
    async fn a_stream_that_ends_normally_has_no_trailing_error() {
        use futures::StreamExt;

        let (tx, rx) = tokio::sync::mpsc::channel(4);
        let mut out = Outlet::new(tx);
        let stream = drain_stream(rx, out.stall_mark());
        assert_eq!(out.push(1_u8), Flow::Continue);
        drop(out);
        let items: Vec<_> = stream.collect().await;
        assert_eq!(items.len(), 1);
        assert_eq!(*items[0].as_ref().unwrap(), 1);
    }

    /// Sink whose `finish` panics, like a buggy consumer bridge would.
    struct PanickingFinishSink;

    impl PieceSink for PanickingFinishSink {
        fn on_piece(&mut self, _piece: &str) -> Flow {
            Flow::Continue
        }
        fn is_closed(&self) -> bool {
            false
        }
    }

    impl Sink for PanickingFinishSink {
        fn finish(self: Box<Self>, _result: Result<(), DomainError>) {
            panic!("finish blew up");
        }
    }

    fn submit_panicking_finish(worker: &Worker) {
        worker
            .submit(Job {
                priority: Priority::Background,
                prompt: "bad-sink".into(),
                params: SamplingParams::from_generate(&GenerateOptions::default()),
                sink: Box::new(PanickingFinishSink),
            })
            .expect("submit");
    }

    #[test]
    fn a_panic_in_a_sink_does_not_strand_the_queued_jobs() {
        let probe = Arc::new(Probe::default());
        let source = FakeSource::new(&probe);
        let (release, gate) = channel();
        *source.gate.lock().unwrap() = Some(gate);
        let w = worker(source, Duration::from_secs(60));

        submit_panicking_finish(&w);
        let queued = submit(&w, "queued");
        release.send(()).unwrap();

        // Served by the same thread: the panic was contained.
        assert_eq!(queued.wait(), Ok(()));
        assert_eq!(*probe.served.lock().unwrap(), ["bad-sink", "queued"]);
    }

    #[test]
    fn a_thread_that_dies_fails_its_queued_jobs() {
        let queue = Arc::new(Queue::new());
        let (done_tx, done) = channel();
        queue
            .push(Job {
                priority: Priority::Background,
                prompt: "stranded".into(),
                params: SamplingParams::from_generate(&GenerateOptions::default()),
                sink: Box::new(TestSink {
                    pieces: Arc::new(Mutex::new(String::new())),
                    closed: Arc::new(AtomicBool::new(false)),
                    done: done_tx,
                }),
            })
            .map_err(|_| "queue refused the job")
            .unwrap();

        let q = Arc::clone(&queue);
        let dying = thread::spawn(move || {
            let _guard = DeathGuard::new(&q);
            panic!("thread body blew up");
        });
        assert!(dying.join().is_err());

        let err = done
            .recv_timeout(Duration::from_secs(2))
            .expect("stranded job was failed")
            .unwrap_err();
        assert!(err.contains("worker stopped"), "{err}");
    }

    /// Engine whose drop panics once armed: kills the thread outside every
    /// `catch_unwind`, after the idle release. With a `gate` the drop first
    /// blocks until the test sends on it, so the test decides exactly when
    /// the thread dies.
    struct BombSource {
        armed: AtomicBool,
        probe: Arc<Probe>,
        gate: Mutex<Option<Receiver<()>>>,
        dropping: AtomicBool,
    }

    impl BombSource {
        fn new(probe: &Arc<Probe>, gate: Option<Receiver<()>>) -> Arc<Self> {
            Arc::new(Self {
                armed: AtomicBool::new(true),
                probe: Arc::clone(probe),
                gate: Mutex::new(gate),
                dropping: AtomicBool::new(false),
            })
        }
    }

    struct BombEngine<'a>(&'a BombSource);

    impl EngineSource for BombSource {
        fn create(&self) -> Result<Box<dyn Engine + '_>, DomainError> {
            self.probe.created.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(BombEngine(self)))
        }
    }

    impl Drop for BombEngine<'_> {
        fn drop(&mut self) {
            if self.0.armed.swap(false, Ordering::SeqCst) {
                self.0.dropping.store(true, Ordering::SeqCst);
                let gate = self.0.gate.lock().unwrap().take();
                if let Some(gate) = gate {
                    let _ = gate.recv();
                }
                panic!("engine drop blew up");
            }
        }
    }

    impl Engine for BombEngine<'_> {
        fn serve(
            &mut self,
            _prompt: &str,
            _params: &SamplingParams,
            sink: &mut dyn PieceSink,
        ) -> Result<(), DomainError> {
            let _ = sink.on_piece("ok");
            Ok(())
        }
    }

    #[test]
    fn a_worker_whose_thread_died_serves_the_next_submit_on_a_new_thread() {
        let probe = Arc::new(Probe::default());
        let source = BombSource::new(&probe, None);
        let w = Worker::new(
            Arc::clone(&source) as Arc<dyn EngineSource>,
            Duration::from_millis(30),
        );
        assert_eq!(submit(&w, "one").wait(), Ok(()));
        // Idle release drops the engine, which panics and kills the thread.
        assert!(eventually(|| w.queue.is_dead()));

        assert_eq!(submit(&w, "two").wait(), Ok(()));
        assert_eq!(probe.created.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn a_job_submitted_while_the_thread_dies_is_still_served() {
        let probe = Arc::new(Probe::default());
        let (kill, gate) = channel();
        let source = BombSource::new(&probe, Some(gate));
        let w = Worker::new(
            Arc::clone(&source) as Arc<dyn EngineSource>,
            Duration::from_millis(30),
        );
        assert_eq!(submit(&w, "one").wait(), Ok(()));
        // The idle release starts dropping the engine, which then blocks on
        // the gate: the thread is alive, so `submit` passes its liveness
        // check, until the test opens the gate.
        assert!(eventually(|| source.dropping.load(Ordering::SeqCst)));
        // Between the check and the push, let the thread die: the push then
        // lands in a queue nobody will ever drain unless it refuses the job.
        *w.before_push.lock().unwrap() = Some(Box::new(move |queue| {
            kill.send(()).unwrap();
            assert!(eventually(|| queue.is_dead()));
        }));

        let raced = submit(&w, "raced");

        assert_eq!(raced.wait(), Ok(()));
        assert_eq!(probe.created.load(Ordering::SeqCst), 2);
    }
}
