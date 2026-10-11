//! Dedicated inference worker thread.
//!
//! A `llama_cpp_2::context::LlamaContext` borrows its model and is not
//! `Send`, so a context that outlives one request has to live on a single
//! OS thread for its whole life. [`Worker`] is that thread: requests
//! ([`Job`]s) are queued over a channel and served one at a time against
//! one persistent [`Engine`] (the context), instead of building and
//! dropping a ~1.2 GB KV cache per request.
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
//! * Dropping the [`Worker`] closes the queue and joins the thread: it
//!   finishes (or, its consumer being gone, abandons) the job in hand,
//!   drops the engine and exits, so the GPU/KV resources are released
//!   before the drop returns (a detached thread could still hold them
//!   while the process tears down its Metal device).
//! * A job whose consumer is already gone is skipped, and a consumer that
//!   goes away mid-job is observed through [`Sink`] so decoding stops at
//!   the next token instead of blocking the queue.
//!
//! The worker is generic over [`EngineSource`]/[`Engine`] so the queueing,
//! idle-release and cancellation policy is tested with a fake engine, no
//! model required.

use std::panic::{self, AssertUnwindSafe};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread;
use std::time::Duration;

use tracing::{debug, error};

use echo_domain::{ChatOptions, DomainError, GenerateOptions};

/// Whether the producer should keep generating after a piece was handed
/// to a [`Sink`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Flow {
    /// Keep decoding.
    Continue,
    /// A stop condition was met or the consumer is gone: end this job.
    Stop,
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

/// Receives the output of one job and owns its completion.
pub(crate) trait Sink: Send {
    /// Handle one decoded text piece; stop decoding when it returns
    /// [`Flow::Stop`].
    fn on_piece(&mut self, piece: &str) -> Flow;

    /// `true` once nobody is listening any more (stream receiver or
    /// result future dropped). Polled between tokens and before a queued
    /// job starts.
    fn is_closed(&self) -> bool;

    /// Report how the job ended. Not called for jobs skipped because the
    /// sink was already closed.
    fn finish(self: Box<Self>, result: Result<(), DomainError>);
}

/// One queued request.
pub(crate) struct Job {
    /// Fully rendered prompt (chat template already applied).
    pub prompt: String,
    /// Sampling knobs.
    pub params: SamplingParams,
    /// Where the output goes.
    pub sink: Box<dyn Sink>,
}

/// A live inference context, used on the worker thread only.
pub(crate) trait Engine {
    /// Run one request to completion, feeding pieces to `sink`.
    fn serve(
        &mut self,
        prompt: &str,
        params: &SamplingParams,
        sink: &mut dyn Sink,
    ) -> Result<(), DomainError>;
}

/// Factory for the engine, shared with the worker thread.
pub(crate) trait EngineSource: Send + Sync + 'static {
    /// Build a fresh engine. Called on the worker thread, and again after
    /// each idle release.
    fn create(&self) -> Result<Box<dyn Engine + '_>, DomainError>;
}

/// Handle to the worker thread. Dropping it shuts the thread down.
pub(crate) struct Worker {
    source: Arc<dyn EngineSource>,
    idle: Duration,
    queue: Mutex<Option<Running>>,
}

/// The queue and thread of a started worker.
struct Running {
    tx: Sender<Job>,
    thread: thread::JoinHandle<()>,
}

impl Worker {
    /// Build a worker that releases its engine after `idle` without jobs.
    /// No thread is started until the first [`Self::submit`].
    pub(crate) fn new(source: Arc<dyn EngineSource>, idle: Duration) -> Self {
        Self {
            source,
            idle,
            queue: Mutex::new(None),
        }
    }

    /// Queue `job` behind the ones already waiting, starting the thread if
    /// it is not running (first use, or it died).
    pub(crate) fn submit(&self, job: Job) -> Result<(), DomainError> {
        let mut queue = self.queue.lock().unwrap_or_else(PoisonError::into_inner);
        let mut job = job;
        // At most one respawn: a send only fails when the thread is gone.
        for _ in 0..2 {
            if queue.is_none() {
                let (tx, rx) = mpsc::channel();
                let source = Arc::clone(&self.source);
                let idle = self.idle;
                let thread = thread::Builder::new()
                    .name("echo-llm-worker".into())
                    .spawn(move || run(&*source, &rx, idle))
                    .map_err(|e| DomainError::LlmFailed(format!("spawn llm worker: {e}")))?;
                *queue = Some(Running { tx, thread });
            }
            let Some(running) = queue.as_ref() else {
                continue;
            };
            match running.tx.send(job) {
                Ok(()) => return Ok(()),
                Err(mpsc::SendError(returned)) => {
                    job = returned;
                    *queue = None;
                }
            }
        }
        Err(DomainError::LlmFailed("llm worker is not running".into()))
    }
}

/// Worker thread body: serve jobs until every [`Worker`] handle is gone.
impl Drop for Worker {
    fn drop(&mut self) {
        let running = self
            .queue
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(Running { tx, thread }) = running {
            drop(tx);
            let _ = thread.join();
        }
    }
}

fn run(source: &dyn EngineSource, rx: &mpsc::Receiver<Job>, idle: Duration) {
    // Declared after `source` is borrowed so the engine drops first.
    let mut engine: Option<Box<dyn Engine + '_>> = None;
    loop {
        let job = if engine.is_some() {
            match rx.recv_timeout(idle) {
                Ok(job) => job,
                Err(RecvTimeoutError::Timeout) => {
                    debug!("llm worker idle: releasing the context");
                    engine = None;
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => break,
            }
        } else {
            match rx.recv() {
                Ok(job) => job,
                Err(_) => break,
            }
        };
        if job.sink.is_closed() {
            debug!("skipping a queued llm job whose consumer is gone");
            continue;
        }
        if engine.is_none() {
            match source.create() {
                Ok(created) => engine = Some(created),
                Err(err) => {
                    job.sink.finish(Err(err));
                    continue;
                }
            }
        }
        let Some(eng) = engine.as_mut() else {
            continue;
        };
        let Job {
            prompt,
            params,
            mut sink,
        } = job;
        // A panic in the engine must not take the queue down with it; the
        // context may be left in an unknown state, so it is rebuilt.
        match panic::catch_unwind(AssertUnwindSafe(|| {
            eng.serve(&prompt, &params, sink.as_mut())
        })) {
            Ok(result) => sink.finish(result),
            Err(_) => {
                error!("llm engine panicked; discarding the context");
                engine = None;
                sink.finish(Err(DomainError::LlmFailed(
                    "inference worker panicked".into(),
                )));
            }
        }
    }
    debug!("llm worker stopped");
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::mpsc::{channel, Receiver};
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
            sink: &mut dyn Sink,
        ) -> Result<(), DomainError> {
            self.source.probe.served.lock().unwrap().push(prompt.into());
            if let Some(gate) = self.source.gate.lock().unwrap().take() {
                let _ = gate.recv_timeout(Duration::from_secs(2));
            }
            if prompt == "panic" {
                panic!("engine blew up");
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

    impl Sink for TestSink {
        fn on_piece(&mut self, piece: &str) -> Flow {
            self.pieces.lock().unwrap().push_str(piece);
            Flow::Continue
        }
        fn is_closed(&self) -> bool {
            self.closed.load(Ordering::SeqCst)
        }
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
        let (done_tx, done) = channel();
        let pieces = Arc::new(Mutex::new(String::new()));
        let closed = Arc::new(AtomicBool::new(false));
        worker
            .submit(Job {
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
}
