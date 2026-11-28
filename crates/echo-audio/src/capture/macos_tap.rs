//! macOS system-audio capture through a Core Audio process tap
//! (macOS 14.2+).
//!
//! A *global* tap mixes what every process sends to the output, minus
//! the processes it excludes (EchoNote itself, so playing a saved
//! session is not transcribed again). The tap is read through a private
//! aggregate device clocked by the default output, whose IOProc hands
//! us Float32 buffers. Unlike ScreenCaptureKit there is no video stream
//! and no Screen Recording permission: macOS asks for "System Audio
//! Recording" (`NSAudioCaptureUsageDescription`) the first time.
//!
//! ## Availability
//!
//! `AudioHardwareCreateProcessTap` / `AudioHardwareDestroyProcessTap`
//! only exist from macOS 14.2, and Rust `extern` declarations are not
//! weak: the linker binds them lazily (`lazy-bind` in `dyld_info
//! -fixups`), so a call on an older system would abort the process (the
//! app supports macOS 11). They are resolved at runtime with `dlsym`
//! instead ([`TapApi`]); a missing symbol means "no taps here" and the
//! caller falls back to ScreenCaptureKit. (cpal 0.18 also imports both,
//! lazily, for a loopback path EchoNote does not use on macOS.)
//!
//! ## Permission
//!
//! macOS has no public way to read the "System Audio Recording"
//! permission, and a tap without it starts fine but delivers only zeros
//! (observed on macOS 27.2). The adapter therefore reads it through the
//! private TCC SPI (`TCCAccessPreflight`, resolved with `dlopen`, as
//! AudioCap does) and starts a tap only when it reports "granted". In any
//! other case the start fails, so [`super::fallback::FallbackAudioCapture`]
//! uses ScreenCaptureKit for that session, and macOS is asked once
//! (`TCCAccessRequest`, non-blocking) so the next session can use the tap.
//! If the SPI is missing or changes meaning, only `0` counts as granted:
//! the worst case is ScreenCaptureKit, never a silent tap (ADR 0002).
//!
//! ## Silence
//!
//! The IOProc only runs while some process plays audio: in silence the
//! tap delivers nothing at all. The capture thread fills those gaps with
//! silent frames ([`silence_to_emit`]) so the stream keeps the session's
//! timeline, like a microphone that is always delivering samples.
//!
//! ## Threading
//!
//! The tap, the aggregate device and the IOProc belong to a dedicated
//! `std::thread` (mirroring [`super::macos`]); the IOProc runs on Core
//! Audio's I/O thread and only touches [`Shared`] (a channel sender and
//! atomics).

use std::ffi::{c_char, c_void, CStr};
use std::mem::size_of;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc as sync_mpsc;
use std::sync::{Arc, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, error, info, warn};

use echo_domain::{
    AudioCapture, AudioFormat, AudioFrame, AudioSource, AudioStream, CaptureSpec, DeviceInfo,
    DomainError, Sample,
};

use block2::{Block, RcBlock};
use objc2::rc::Retained;
use objc2::runtime::{AnyClass, AnyObject};
use objc2::{AnyThread, Message};
use objc2_core_audio::{
    kAudioAggregateDeviceIsPrivateKey, kAudioAggregateDeviceIsStackedKey,
    kAudioAggregateDeviceMainSubDeviceKey, kAudioAggregateDeviceNameKey,
    kAudioAggregateDeviceSubDeviceListKey, kAudioAggregateDeviceTapAutoStartKey,
    kAudioAggregateDeviceTapListKey, kAudioAggregateDeviceUIDKey,
    kAudioDevicePropertyDeviceIsAlive, kAudioDevicePropertyDeviceUID,
    kAudioHardwarePropertyDefaultOutputDevice, kAudioHardwarePropertyTranslatePIDToProcessObject,
    kAudioObjectPropertyElementMain, kAudioObjectPropertyScopeGlobal, kAudioObjectSystemObject,
    kAudioSubDeviceUIDKey, kAudioSubTapDriftCompensationKey, kAudioSubTapUIDKey,
    kAudioTapPropertyFormat, AudioDeviceCreateIOProcID, AudioDeviceDestroyIOProcID,
    AudioDeviceIOProcID, AudioDeviceStart, AudioDeviceStop, AudioHardwareCreateAggregateDevice,
    AudioHardwareDestroyAggregateDevice, AudioObjectGetPropertyData, AudioObjectID,
    AudioObjectPropertyAddress, CATapDescription, CATapMuteBehavior,
};
use objc2_core_audio_types::{
    kAudioFormatFlagIsFloat, kAudioFormatFlagIsNonInterleaved, AudioBuffer, AudioBufferList,
    AudioStreamBasicDescription, AudioTimeStamp,
};
use objc2_core_foundation::CFDictionary;
use objc2_foundation::{NSArray, NSDictionary, NSNumber, NSString};

use super::interleave::{interleave, Plane};
use super::reconnect::{retry, stop_requested, Backoff};

/// Synthetic device id reported by [`CoreAudioTapCapture::list_devices`].
pub const TAP_DEVICE_ID: &str = "macos:coreaudio-tap:default-output";

/// Same bound as the other adapters (~5 s of 48 kHz stereo).
const CHANNEL_CAPACITY_HINT: usize = 512;

/// How often the capture thread checks for stop and fills silence.
const SUPERVISE_EVERY: Duration = Duration::from_millis(50);

/// Without a buffer for this long the output is considered silent and
/// the timeline is filled with zeros. Longer than one I/O cycle (~10 ms)
/// with margin, short enough to keep the stream responsive.
const QUIET_AFTER: Duration = Duration::from_millis(200);

/// Longest silent frame sent at once.
const MAX_SILENT_FRAME: Duration = Duration::from_millis(100);

/// `noErr`.
const NO_ERR: i32 = 0;

/// Core Audio process-tap adapter for system audio.
#[derive(Debug, Default, Clone)]
pub struct CoreAudioTapCapture;

impl CoreAudioTapCapture {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Whether this macOS has process taps (14.2+).
    #[must_use]
    pub fn is_supported() -> bool {
        TapApi::get().is_some()
    }

    /// Whether the "System Audio Recording" permission is granted, as
    /// the private TCC call reports it (`false` when unknown).
    #[must_use]
    pub fn permission_granted() -> bool {
        TccStatus::current() == TccStatus::Granted
    }
}

#[async_trait]
impl AudioCapture for CoreAudioTapCapture {
    async fn list_devices(&self, source: AudioSource) -> Result<Vec<DeviceInfo>, DomainError> {
        reject_microphone(source)?;
        TapApi::get().ok_or_else(unsupported)?;
        tap_permission(TccStatus::current(), || {})?;
        Ok(vec![DeviceInfo {
            id: TAP_DEVICE_ID.to_string(),
            name: "System Output (Core Audio tap)".to_string(),
            is_default: true,
        }])
    }

    async fn start(&self, spec: CaptureSpec) -> Result<Box<dyn AudioStream>, DomainError> {
        reject_microphone(spec.source)?;
        let api = TapApi::get().ok_or_else(unsupported)?;
        tap_permission(TccStatus::current(), request_permission_once)?;
        let started = tokio::task::spawn_blocking(move || start_capture(api))
            .await
            .map_err(|e| DomainError::AudioCaptureFailed(format!("start join: {e}")))??;
        Ok(Box::new(started))
    }
}

fn reject_microphone(source: AudioSource) -> Result<(), DomainError> {
    if source == AudioSource::SystemOutput {
        return Ok(());
    }
    Err(DomainError::AudioDeviceUnavailable(format!(
        "{source:?} is not handled by the Core Audio tap adapter; \
         use the cpal microphone adapter instead"
    )))
}

fn unsupported() -> DomainError {
    DomainError::AudioDeviceUnavailable("Core Audio process taps need macOS 14.2 or later".into())
}

// ---------------------------------------------------------------------------
// System Audio Recording permission (private TCC SPI)
// ---------------------------------------------------------------------------

/// The "System Audio Recording" permission as TCC reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TccStatus {
    Granted,
    /// Denied, or not asked yet: TCC answers 2 for both.
    NotGranted,
    /// The private TCC call is unavailable.
    Unknown,
}

impl TccStatus {
    /// Only `0` means granted; any other code is treated as not granted.
    fn from_preflight(code: Option<i32>) -> Self {
        match code {
            Some(0) => Self::Granted,
            Some(_) => Self::NotGranted,
            None => Self::Unknown,
        }
    }

    fn current() -> Self {
        Self::from_preflight(Tcc::get().map(Tcc::preflight))
    }
}

/// Lets the tap start only with the permission granted. When it is not,
/// calls `request` (macOS prompt for the next session) and fails so the
/// fallback capture takes this session; when the state is unknown, fails
/// without asking.
fn tap_permission(status: TccStatus, request: impl FnOnce()) -> Result<(), DomainError> {
    match status {
        TccStatus::Granted => Ok(()),
        TccStatus::NotGranted => {
            request();
            Err(DomainError::AudioDeviceUnavailable(
                "the Core Audio tap needs the System Audio Recording permission; \
                 macOS was asked to prompt, using the fallback capture meanwhile"
                    .into(),
            ))
        }
        TccStatus::Unknown => Err(DomainError::AudioDeviceUnavailable(
            "cannot read the System Audio Recording permission; using the fallback capture".into(),
        )),
    }
}

/// Asks macOS for the permission at most once per process.
fn request_permission_once() {
    static REQUESTED: AtomicBool = AtomicBool::new(false);
    if REQUESTED.swap(true, Ordering::AcqRel) {
        return;
    }
    if let Some(tcc) = Tcc::get() {
        tcc.request();
    }
}

type TccPreflight = unsafe extern "C" fn(*const NSString, *const c_void) -> i32;
type TccRequest = unsafe extern "C" fn(*const NSString, *const c_void, *const Block<dyn Fn(u8)>);

extern "C" {
    fn dlopen(path: *const c_char, mode: i32) -> *mut c_void;
}

/// `RTLD_LAZY`.
const RTLD_LAZY: i32 = 1;

/// `TCCAccessPreflight` / `TCCAccessRequest` from the private TCC
/// framework, looked up once.
#[derive(Clone, Copy)]
struct Tcc {
    preflight: TccPreflight,
    request: TccRequest,
}

impl Tcc {
    const SERVICE: &str = "kTCCServiceAudioCapture";

    fn get() -> Option<Self> {
        static TCC: OnceLock<Option<Tcc>> = OnceLock::new();
        *TCC.get_or_init(|| {
            // SAFETY: dlopen/dlsym with NUL-terminated strings; the
            // symbols have the signatures AudioCap uses (CFStringRef
            // service, CFDictionaryRef options, block callback).
            unsafe {
                let handle = dlopen(
                    c"/System/Library/PrivateFrameworks/TCC.framework/Versions/A/TCC".as_ptr(),
                    RTLD_LAZY,
                );
                if handle.is_null() {
                    return None;
                }
                let preflight = dlsym(handle, c"TCCAccessPreflight".as_ptr());
                let request = dlsym(handle, c"TCCAccessRequest".as_ptr());
                if preflight.is_null() || request.is_null() {
                    return None;
                }
                Some(Tcc {
                    preflight: std::mem::transmute::<*mut c_void, TccPreflight>(preflight),
                    request: std::mem::transmute::<*mut c_void, TccRequest>(request),
                })
            }
        })
    }

    fn preflight(self) -> i32 {
        let service = NSString::from_str(Self::SERVICE);
        // SAFETY: NSString is toll-free bridged to CFStringRef; no options.
        unsafe { (self.preflight)(Retained::as_ptr(&service), std::ptr::null()) }
    }

    /// Shows the macOS prompt if the user was never asked; returns at once.
    fn request(self) {
        let service = NSString::from_str(Self::SERVICE);
        let done = RcBlock::new(|granted: u8| {
            info!(
                granted = granted != 0,
                "System Audio Recording permission answered"
            );
        });
        // SAFETY: as above; TCC copies the block before returning.
        unsafe { (self.request)(Retained::as_ptr(&service), std::ptr::null(), &*done) };
    }
}

// ---------------------------------------------------------------------------
// Runtime-resolved tap API
// ---------------------------------------------------------------------------

type CreateProcessTap =
    unsafe extern "C-unwind" fn(Option<&CATapDescription>, *mut AudioObjectID) -> i32;
type DestroyProcessTap = unsafe extern "C-unwind" fn(AudioObjectID) -> i32;

extern "C" {
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

/// `RTLD_DEFAULT` on Apple platforms: search every loaded image.
const RTLD_DEFAULT: *mut c_void = -2isize as *mut c_void;

/// The macOS 14.2+ entry points, looked up once.
#[derive(Clone, Copy)]
struct TapApi {
    create: CreateProcessTap,
    destroy: DestroyProcessTap,
}

impl TapApi {
    fn get() -> Option<Self> {
        static API: OnceLock<Option<TapApi>> = OnceLock::new();
        *API.get_or_init(|| {
            // SAFETY: dlsym with RTLD_DEFAULT and NUL-terminated names;
            // the returned pointers have the C signatures declared in
            // <CoreAudio/AudioHardware.h>.
            let create = unsafe { dlsym(RTLD_DEFAULT, c"AudioHardwareCreateProcessTap".as_ptr()) };
            let destroy =
                unsafe { dlsym(RTLD_DEFAULT, c"AudioHardwareDestroyProcessTap".as_ptr()) };
            let class = AnyClass::get(c"CATapDescription");
            if create.is_null() || destroy.is_null() || class.is_none() {
                return None;
            }
            // SAFETY: see above; non-null symbols of those functions.
            Some(unsafe {
                TapApi {
                    create: std::mem::transmute::<*mut c_void, CreateProcessTap>(create),
                    destroy: std::mem::transmute::<*mut c_void, DestroyProcessTap>(destroy),
                }
            })
        })
    }
}

// ---------------------------------------------------------------------------
// Stream handle
// ---------------------------------------------------------------------------

struct StartedStream {
    rx: mpsc::Receiver<AudioFrame>,
    stop: Option<oneshot::Sender<()>>,
    format: AudioFormat,
    join: Option<thread::JoinHandle<()>>,
}

#[async_trait]
impl AudioStream for StartedStream {
    fn format(&self) -> AudioFormat {
        self.format
    }

    async fn next_frame(&mut self) -> Option<AudioFrame> {
        self.rx.recv().await
    }

    async fn stop(&mut self) -> Result<(), DomainError> {
        if let Some(tx) = self.stop.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.join.take() {
            tokio::task::spawn_blocking(move || handle.join())
                .await
                .map_err(|e| DomainError::AudioCaptureFailed(format!("join task: {e}")))?
                .map_err(|e| {
                    DomainError::AudioCaptureFailed(format!("tap thread panicked: {e:?}"))
                })?;
        }
        self.rx.close();
        Ok(())
    }
}

impl Drop for StartedStream {
    fn drop(&mut self) {
        if let Some(tx) = self.stop.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.join.take() {
            let _ = handle.join();
        }
    }
}

fn start_capture(api: TapApi) -> Result<StartedStream, DomainError> {
    let (tx, rx) = mpsc::channel::<AudioFrame>(CHANNEL_CAPACITY_HINT);
    let (stop_tx, stop_rx) = oneshot::channel::<()>();
    let (ready_tx, ready_rx) = sync_mpsc::channel::<Result<AudioFormat, DomainError>>();

    let join = thread::Builder::new()
        .name("echo-audio-tap".into())
        .spawn(move || run_capture_thread(api, tx, stop_rx, ready_tx))
        .map_err(|e| DomainError::AudioCaptureFailed(format!("spawn tap thread: {e}")))?;

    match ready_rx.recv() {
        Ok(Ok(format)) => Ok(StartedStream {
            rx,
            stop: Some(stop_tx),
            format,
            join: Some(join),
        }),
        Ok(Err(e)) => {
            let _ = join.join();
            Err(e)
        }
        Err(e) => Err(DomainError::AudioCaptureFailed(format!(
            "tap thread died before signalling readiness: {e}"
        ))),
    }
}

fn run_capture_thread(
    api: TapApi,
    tx: mpsc::Sender<AudioFrame>,
    mut stop_rx: oneshot::Receiver<()>,
    ready_tx: sync_mpsc::Sender<Result<AudioFormat, DomainError>>,
) {
    let timeline = Arc::new(Timeline::new());
    let tap = match Tap::open(api, tx.clone(), Arc::clone(&timeline)) {
        Ok(tap) => tap,
        Err(e) => {
            let _ = ready_tx.send(Err(e));
            return;
        }
    };
    let format = tap.format;
    info!(
        rate = format.sample_rate_hz,
        ch = format.channels,
        "Core Audio tap capture live"
    );
    timeline.mark_live();
    let _ = ready_tx.send(Ok(format));

    let mut format = format;
    let mut current = Some(tap);
    while !stop_requested(&mut stop_rx) {
        // Switching the default output keeps a global tap working, but
        // losing the device that clocks the aggregate (headphones
        // unplugged) stops its IOProc: rebuild on the new default output.
        if current.as_ref().is_some_and(|t| !device_alive(t.clock)) {
            current = None;
            warn!("Core Audio tap clock device went away — rebuilding the tap");
            let rebuilt = retry(Backoff::DEVICE, &mut stop_rx, |_| {
                Tap::open(api, tx.clone(), Arc::clone(&timeline))
            });
            match rebuilt {
                Some(tap) => {
                    info!(
                        rate = tap.format.sample_rate_hz,
                        ch = tap.format.channels,
                        "Core Audio tap rebuilt"
                    );
                    format = tap.format;
                    current = Some(tap);
                }
                None if stop_requested(&mut stop_rx) => break,
                None => {
                    warn!("Core Audio tap could not be rebuilt — ending capture");
                    break;
                }
            }
        }
        fill_silence(&timeline, format, &tx);
        thread::sleep(SUPERVISE_EVERY);
    }

    drop(current);
    info!("Core Audio tap capture stopped");
}

/// Sends the silent frames [`silence_to_emit`] asks for.
fn fill_silence(timeline: &Timeline, format: AudioFormat, tx: &mpsc::Sender<AudioFrame>) {
    let mut missing = silence_to_emit(
        timeline.elapsed(),
        timeline.sent(),
        format.sample_rate_hz,
        timeline.since_last_buffer(),
        QUIET_AFTER,
    );
    let max_frames = frames_in(MAX_SILENT_FRAME, format.sample_rate_hz).max(1);
    while missing > 0 {
        let frames = missing.min(max_frames);
        let at = timeline.advance(frames, format.sample_rate_hz);
        let samples = vec![0.0; frames as usize * usize::from(format.channels)];
        send_frame(tx, samples, format, at);
        missing -= frames;
    }
}

/// Frames of silence (at `rate_hz`) to append so the stream keeps up
/// with wall-clock time: zero while buffers keep arriving (less than
/// `quiet_after` since the last one), otherwise whatever separates the
/// audio already `sent` from `elapsed`.
fn silence_to_emit(
    elapsed: Duration,
    sent: Duration,
    rate_hz: u32,
    since_last_buffer: Duration,
    quiet_after: Duration,
) -> u64 {
    if since_last_buffer < quiet_after {
        return 0;
    }
    frames_in(elapsed.saturating_sub(sent), rate_hz)
}

fn frames_in(span: Duration, rate_hz: u32) -> u64 {
    (span.as_nanos() * u128::from(rate_hz) / 1_000_000_000) as u64
}

fn send_frame(
    tx: &mpsc::Sender<AudioFrame>,
    samples: Vec<Sample>,
    format: AudioFormat,
    captured_at_ns: u64,
) {
    let frame = AudioFrame {
        samples,
        format,
        captured_at_ns,
    };
    if let Err(mpsc::error::TrySendError::Full(_)) = tx.try_send(frame) {
        warn!("audio frame dropped — consumer is slower than the Core Audio tap");
    }
}

// ---------------------------------------------------------------------------
// Timeline shared between the IOProc and the capture thread
// ---------------------------------------------------------------------------

/// Audio sent so far (real or silent) in nanoseconds — rate-agnostic,
/// so a tap rebuilt at another sample rate continues the same timeline —
/// plus when the tap went live and when its last buffer arrived, both as
/// nanoseconds since `start` (`last_buffer_ns == u64::MAX`: none yet).
struct Timeline {
    start: Instant,
    sent_ns: AtomicU64,
    live_ns: AtomicU64,
    last_buffer_ns: AtomicU64,
}

impl Timeline {
    fn new() -> Self {
        Self {
            start: Instant::now(),
            sent_ns: AtomicU64::new(0),
            live_ns: AtomicU64::new(0),
            last_buffer_ns: AtomicU64::new(u64::MAX),
        }
    }

    /// Marks the moment the tap started: the stream's time 0. Setting
    /// up the tap takes ~0.6 s, which must not become leading silence
    /// (it would put system audio behind the microphone in a mix).
    fn mark_live(&self) {
        let now = self.start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
        self.live_ns.store(now, Ordering::Release);
    }

    /// Audio sent so far.
    fn sent(&self) -> Duration {
        Duration::from_nanos(self.sent_ns.load(Ordering::Acquire))
    }

    /// Accounts `frames` at `rate_hz` as sent; returns where they start
    /// on the stream's timeline (their `captured_at_ns`).
    fn advance(&self, frames: u64, rate_hz: u32) -> u64 {
        let ns = u128::from(frames) * 1_000_000_000 / u128::from(rate_hz.max(1));
        self.sent_ns
            .fetch_add(ns.min(u128::from(u64::MAX)) as u64, Ordering::AcqRel)
    }

    /// Time since the tap went live.
    fn elapsed(&self) -> Duration {
        self.start
            .elapsed()
            .saturating_sub(Duration::from_nanos(self.live_ns.load(Ordering::Acquire)))
    }

    fn mark_buffer(&self) {
        let now = self
            .start
            .elapsed()
            .as_nanos()
            .min(u128::from(u64::MAX - 1)) as u64;
        self.last_buffer_ns.store(now, Ordering::Release);
    }

    fn since_last_buffer(&self) -> Duration {
        match self.last_buffer_ns.load(Ordering::Acquire) {
            u64::MAX => self.elapsed(),
            ns => self
                .start
                .elapsed()
                .saturating_sub(Duration::from_nanos(ns)),
        }
    }
}

// ---------------------------------------------------------------------------
// Tap + aggregate device + IOProc
// ---------------------------------------------------------------------------

/// What the IOProc needs, owned by [`Tap`] and lent to Core Audio as
/// the IOProc's client data.
struct Shared {
    tx: mpsc::Sender<AudioFrame>,
    format: AudioFormat,
    timeline: Arc<Timeline>,
}

/// A running tap. Fields are filled as each step succeeds, and `Drop`
/// releases whatever exists in reverse order: IOProc stop and destroy,
/// aggregate device, tap.
struct Tap {
    api: TapApi,
    tap: Option<AudioObjectID>,
    aggregate: Option<AudioObjectID>,
    io_proc: AudioDeviceIOProcID,
    started: bool,
    shared: Option<Box<Shared>>,
    /// The output device clocking the aggregate (default output at open).
    clock: AudioObjectID,
    format: AudioFormat,
}

impl Tap {
    fn open(
        api: TapApi,
        tx: mpsc::Sender<AudioFrame>,
        timeline: Arc<Timeline>,
    ) -> Result<Self, DomainError> {
        let mut tap = Tap {
            api,
            tap: None,
            aggregate: None,
            io_proc: None,
            started: false,
            shared: None,
            clock: 0,
            format: AudioFormat {
                sample_rate_hz: 0,
                channels: 0,
            },
        };

        // 1. Global tap of every process but this one, left audible.
        let own = own_process_object();
        let excluded: Retained<NSArray<NSNumber>> = match own {
            Some(id) => NSArray::from_retained_slice(&[NSNumber::new_u32(id)]),
            None => {
                warn!("cannot resolve EchoNote's audio process; its own audio will be captured");
                NSArray::new()
            }
        };
        // SAFETY: plain Objective-C initialisers and setters on a fresh
        // object; the class exists (checked by TapApi::get).
        let description = unsafe {
            let d = CATapDescription::initStereoGlobalTapButExcludeProcesses(
                CATapDescription::alloc(),
                &excluded,
            );
            d.setPrivate(true);
            d.setMuteBehavior(CATapMuteBehavior::Unmuted);
            d
        };
        let mut tap_id: AudioObjectID = 0;
        // SAFETY: valid description and out pointer.
        check(
            unsafe { (api.create)(Some(&description), &mut tap_id) },
            "AudioHardwareCreateProcessTap",
        )?;
        tap.tap = Some(tap_id);

        // 2. Its format.
        let asbd: AudioStreamBasicDescription = get_property(tap_id, kAudioTapPropertyFormat)
            .map_err(|st| status_error("kAudioTapPropertyFormat", st))?;
        let layout = TapLayout::from_asbd(&asbd)?;
        tap.format = layout.format;

        // 3. Private aggregate device that reads the tap, clocked by the
        //    default output.
        let (clock, output_uid) = default_output()?;
        tap.clock = clock;
        // SAFETY: UUID() of an initialised description.
        let tap_uid = unsafe { description.UUID() }.UUIDString();
        let aggregate_uid = NSString::from_str(&format!(
            "app.echonote.tap.{}.{}",
            std::process::id(),
            tap_id
        ));
        let dict = aggregate_description(&aggregate_uid, &output_uid, &tap_uid);
        let mut aggregate_id: AudioObjectID = 0;
        // SAFETY: NSDictionary is toll-free bridged to CFDictionary.
        let cf: &CFDictionary = unsafe { &*(Retained::as_ptr(&dict).cast::<CFDictionary>()) };
        check(
            unsafe { AudioHardwareCreateAggregateDevice(cf, NonNull::from(&mut aggregate_id)) },
            "AudioHardwareCreateAggregateDevice",
        )?;
        tap.aggregate = Some(aggregate_id);

        // 4. IOProc on the aggregate device.
        let shared = Box::new(Shared {
            tx,
            format: layout.format,
            timeline,
        });
        let client = std::ptr::from_ref::<Shared>(&shared)
            .cast_mut()
            .cast::<c_void>();
        tap.shared = Some(shared);
        let proc: AudioDeviceIOProcID = if layout.non_interleaved {
            Some(io_proc::<true>)
        } else {
            Some(io_proc::<false>)
        };
        let mut io_proc_id: AudioDeviceIOProcID = None;
        // SAFETY: `client` points into the Box owned by `tap`, which
        // outlives the IOProc (destroyed first in Drop).
        check(
            unsafe {
                AudioDeviceCreateIOProcID(
                    aggregate_id,
                    proc,
                    client,
                    NonNull::from(&mut io_proc_id),
                )
            },
            "AudioDeviceCreateIOProcID",
        )?;
        tap.io_proc = io_proc_id;

        // SAFETY: valid device and IOProc id.
        check(
            unsafe { AudioDeviceStart(aggregate_id, tap.io_proc) },
            "AudioDeviceStart",
        )?;
        tap.started = true;
        Ok(tap)
    }
}

impl Drop for Tap {
    fn drop(&mut self) {
        // SAFETY: each id was created by the matching call in `open` and
        // is released once, in reverse order of creation.
        unsafe {
            if let Some(aggregate) = self.aggregate {
                if self.started {
                    log_status(AudioDeviceStop(aggregate, self.io_proc), "AudioDeviceStop");
                }
                if self.io_proc.is_some() {
                    log_status(
                        AudioDeviceDestroyIOProcID(aggregate, self.io_proc),
                        "AudioDeviceDestroyIOProcID",
                    );
                }
                log_status(
                    AudioHardwareDestroyAggregateDevice(aggregate),
                    "AudioHardwareDestroyAggregateDevice",
                );
            }
            if let Some(tap) = self.tap {
                log_status((self.api.destroy)(tap), "AudioHardwareDestroyProcessTap");
            }
        }
        // The IOProc is gone; its client data can go too.
        self.shared.take();
    }
}

/// IOProc: copies the tap's input buffers into one interleaved frame.
/// `NON_INTERLEAVED` selects the buffer layout reported by the tap.
unsafe extern "C-unwind" fn io_proc<const NON_INTERLEAVED: bool>(
    _device: AudioObjectID,
    _now: NonNull<AudioTimeStamp>,
    input: NonNull<AudioBufferList>,
    _input_time: NonNull<AudioTimeStamp>,
    _output: NonNull<AudioBufferList>,
    _output_time: NonNull<AudioTimeStamp>,
    client: *mut c_void,
) -> i32 {
    // SAFETY: `client` is the `Shared` registered with this IOProc and
    // lives until after the IOProc is destroyed.
    let shared = unsafe { &*client.cast::<Shared>() };
    // SAFETY: Core Audio passes a valid list of `mNumberBuffers` buffers.
    // The Rust type declares `mBuffers: [AudioBuffer; 1]`, so the slice
    // is built from a raw pointer to the field, not from that array.
    let buffers = unsafe {
        let list = input.as_ptr();
        std::slice::from_raw_parts(
            std::ptr::addr_of!((*list).mBuffers).cast::<AudioBuffer>(),
            (*list).mNumberBuffers as usize,
        )
    };
    let planes: Vec<Plane<'_>> = buffers
        .iter()
        .filter(|b| !b.mData.is_null())
        .map(|b| Plane {
            channels: b.mNumberChannels,
            // SAFETY: Float32 data of `mDataByteSize` bytes (format
            // checked in TapLayout::from_asbd).
            samples: unsafe {
                std::slice::from_raw_parts(
                    b.mData.cast::<f32>(),
                    b.mDataByteSize as usize / size_of::<f32>(),
                )
            },
        })
        .collect();
    let samples = if NON_INTERLEAVED || planes.len() != 1 {
        interleave(&planes, shared.format.channels).samples
    } else {
        planes[0].samples.to_vec()
    };
    let channels = usize::from(shared.format.channels.max(1));
    let frames = (samples.len() / channels) as u64;
    if frames == 0 {
        return NO_ERR;
    }
    shared.timeline.mark_buffer();
    let at = shared
        .timeline
        .advance(frames, shared.format.sample_rate_hz);
    send_frame(&shared.tx, samples, shared.format, at);
    NO_ERR
}

/// How the tap lays out its samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TapLayout {
    format: AudioFormat,
    non_interleaved: bool,
}

impl TapLayout {
    /// Accepts the Float32 PCM a tap delivers; anything else is refused
    /// rather than misread.
    fn from_asbd(asbd: &AudioStreamBasicDescription) -> Result<Self, DomainError> {
        let is_float = asbd.mFormatFlags & kAudioFormatFlagIsFloat != 0;
        let rate = asbd.mSampleRate;
        if !is_float || asbd.mBitsPerChannel != 32 || asbd.mChannelsPerFrame == 0 || rate < 1.0 {
            return Err(DomainError::AudioFormatUnsupported(format!(
                "Core Audio tap format not supported: {} Hz, {} ch, {} bits, flags {:#x}",
                rate, asbd.mChannelsPerFrame, asbd.mBitsPerChannel, asbd.mFormatFlags
            )));
        }
        let channels = u16::try_from(asbd.mChannelsPerFrame).map_err(|_| {
            DomainError::AudioFormatUnsupported(format!(
                "Core Audio tap has {} channels",
                asbd.mChannelsPerFrame
            ))
        })?;
        Ok(Self {
            format: AudioFormat {
                sample_rate_hz: rate.round() as u32,
                channels,
            },
            non_interleaved: asbd.mFormatFlags & kAudioFormatFlagIsNonInterleaved != 0,
        })
    }
}

// ---------------------------------------------------------------------------
// Core Audio helpers
// ---------------------------------------------------------------------------

fn address(selector: u32) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress {
        mSelector: selector,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain,
    }
}

/// Reads a fixed-size property, optionally with qualifier data.
fn get_property_with<T: Copy, Q>(
    object: AudioObjectID,
    selector: u32,
    qualifier: Option<&Q>,
) -> Result<T, i32> {
    let mut value = std::mem::MaybeUninit::<T>::zeroed();
    let mut size = size_of::<T>() as u32;
    let mut addr = address(selector);
    let (q_size, q_ptr) = qualifier.map_or((0, std::ptr::null()), |q| {
        (
            size_of::<Q>() as u32,
            std::ptr::from_ref(q).cast::<c_void>(),
        )
    });
    // SAFETY: `value` has room for `size` bytes; qualifier, if any, is a
    // live `Q` of `q_size` bytes.
    let status = unsafe {
        AudioObjectGetPropertyData(
            object,
            NonNull::from(&mut addr),
            q_size,
            q_ptr,
            NonNull::from(&mut size),
            NonNull::new_unchecked(value.as_mut_ptr().cast::<c_void>()),
        )
    };
    if status != NO_ERR {
        return Err(status);
    }
    // SAFETY: Core Audio filled the value (zeroed beforehand regardless).
    Ok(unsafe { value.assume_init() })
}

fn get_property<T: Copy>(object: AudioObjectID, selector: u32) -> Result<T, i32> {
    get_property_with::<T, ()>(object, selector, None)
}

const SYSTEM_OBJECT: AudioObjectID = kAudioObjectSystemObject as AudioObjectID;

/// Core Audio's object for this process, or `None` when it has none.
fn own_process_object() -> Option<AudioObjectID> {
    let pid = std::process::id() as i32;
    match get_property_with::<AudioObjectID, i32>(
        SYSTEM_OBJECT,
        kAudioHardwarePropertyTranslatePIDToProcessObject,
        Some(&pid),
    ) {
        Ok(0) | Err(_) => None,
        Ok(id) => Some(id),
    }
}

/// The default output device and its UID.
fn default_output() -> Result<(AudioObjectID, Retained<NSString>), DomainError> {
    let device: AudioObjectID =
        get_property(SYSTEM_OBJECT, kAudioHardwarePropertyDefaultOutputDevice)
            .map_err(|st| status_error("kAudioHardwarePropertyDefaultOutputDevice", st))?;
    if device == 0 {
        return Err(DomainError::AudioDeviceUnavailable(
            "no default output device to tap".into(),
        ));
    }
    let uid: *mut NSString = get_property(device, kAudioDevicePropertyDeviceUID)
        .map_err(|st| status_error("kAudioDevicePropertyDeviceUID", st))?;
    // SAFETY: the UID property returns a +1 CFString (toll-free bridged).
    let uid = unsafe { Retained::from_raw(uid) }.ok_or_else(|| {
        DomainError::AudioDeviceUnavailable("default output device has no UID".into())
    })?;
    Ok((device, uid))
}

/// Whether Core Audio still reports `device` as present.
fn device_alive(device: AudioObjectID) -> bool {
    matches!(get_property::<u32>(device, kAudioDevicePropertyDeviceIsAlive), Ok(alive) if alive != 0)
}

/// The aggregate-device description AudioCap and Apple's sample use:
/// private, not stacked, the default output as clock, and the tap
/// (drift-compensated) started with the device.
fn aggregate_description(
    uid: &NSString,
    output_uid: &NSString,
    tap_uid: &NSString,
) -> Retained<NSDictionary<NSString, AnyObject>> {
    let sub_device = dictionary(&[(kAudioSubDeviceUIDKey, any(output_uid.retain()))]);
    let sub_tap = dictionary(&[
        (kAudioSubTapUIDKey, any(tap_uid.retain())),
        (
            kAudioSubTapDriftCompensationKey,
            any(NSNumber::new_bool(true)),
        ),
    ]);
    dictionary(&[
        (kAudioAggregateDeviceUIDKey, any(uid.retain())),
        (
            kAudioAggregateDeviceNameKey,
            any(NSString::from_str("EchoNote system audio")),
        ),
        (
            kAudioAggregateDeviceMainSubDeviceKey,
            any(output_uid.retain()),
        ),
        (
            kAudioAggregateDeviceIsPrivateKey,
            any(NSNumber::new_bool(true)),
        ),
        (
            kAudioAggregateDeviceIsStackedKey,
            any(NSNumber::new_bool(false)),
        ),
        (
            kAudioAggregateDeviceTapAutoStartKey,
            any(NSNumber::new_bool(true)),
        ),
        (
            kAudioAggregateDeviceSubDeviceListKey,
            any(NSArray::from_retained_slice(&[sub_device])),
        ),
        (
            kAudioAggregateDeviceTapListKey,
            any(NSArray::from_retained_slice(&[sub_tap])),
        ),
    ])
}

fn any<T: Message>(object: Retained<T>) -> Retained<AnyObject> {
    // SAFETY: every Objective-C object is an AnyObject.
    unsafe { Retained::cast_unchecked(object) }
}

fn dictionary(
    entries: &[(&CStr, Retained<AnyObject>)],
) -> Retained<NSDictionary<NSString, AnyObject>> {
    let keys: Vec<Retained<NSString>> = entries
        .iter()
        .map(|(k, _)| NSString::from_str(&k.to_string_lossy()))
        .collect();
    let key_refs: Vec<&NSString> = keys.iter().map(|k| &**k).collect();
    let values: Vec<Retained<AnyObject>> = entries.iter().map(|(_, v)| v.clone()).collect();
    NSDictionary::from_retained_objects(&key_refs, &values)
}

fn check(status: i32, call: &str) -> Result<(), DomainError> {
    if status == NO_ERR {
        Ok(())
    } else {
        Err(status_error(call, status))
    }
}

fn status_error(call: &str, status: i32) -> DomainError {
    DomainError::AudioCaptureFailed(format!(
        "{call} failed: OSStatus {status} ({})",
        fourcc(status)
    ))
}

fn log_status(status: i32, call: &str) {
    if status == NO_ERR {
        debug!(call, "Core Audio teardown step done");
    } else {
        error!(call, status, code = %fourcc(status), "Core Audio teardown step failed");
    }
}

/// OSStatus codes are often four ASCII characters ('who?', '!dev').
fn fourcc(status: i32) -> String {
    let bytes = status.to_be_bytes();
    if bytes.iter().all(|b| b.is_ascii_graphic() || *b == b' ') {
        format!("'{}'", String::from_utf8_lossy(&bytes))
    } else {
        status.to_string()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    const MS: fn(u64) -> Duration = Duration::from_millis;

    #[test]
    fn no_silence_while_buffers_keep_arriving() {
        assert_eq!(
            silence_to_emit(MS(1_000), MS(0), 48_000, MS(10), QUIET_AFTER),
            0
        );
    }

    #[test]
    fn silence_fills_up_to_wall_clock_once_the_output_goes_quiet() {
        // 1 s elapsed, 0.5 s sent, quiet for 300 ms: 0.5 s of zeros.
        assert_eq!(
            silence_to_emit(MS(1_000), MS(500), 48_000, MS(300), QUIET_AFTER),
            24_000
        );
    }

    #[test]
    fn a_pause_shorter_than_quiet_after_is_not_filled_then_caught_up() {
        // Bounded skew: a gap under QUIET_AFTER stays unfilled until the
        // output has been quiet long enough, then the whole gap is filled.
        assert_eq!(
            silence_to_emit(MS(1_150), MS(1_000), 48_000, MS(150), QUIET_AFTER),
            0
        );
        assert_eq!(
            silence_to_emit(MS(1_250), MS(1_000), 48_000, MS(250), QUIET_AFTER),
            12_000
        );
    }

    #[test]
    fn silence_never_rewinds_a_stream_ahead_of_the_clock() {
        assert_eq!(
            silence_to_emit(MS(1_000), MS(1_041), 48_000, MS(500), QUIET_AFTER),
            0
        );
    }

    #[test]
    fn a_tap_that_never_delivers_is_all_silence() {
        let timeline = Timeline::new();
        assert!(timeline.since_last_buffer() <= timeline.start.elapsed());
        assert_eq!(
            silence_to_emit(MS(400), MS(0), 16_000, MS(400), QUIET_AFTER),
            6_400
        );
    }

    #[test]
    fn the_timeline_continues_across_a_sample_rate_change() {
        // A tap rebuilt on another output may run at another rate.
        let timeline = Timeline::new();
        assert_eq!(timeline.advance(48_000, 48_000), 0);
        assert_eq!(timeline.advance(22_050, 44_100), 1_000_000_000);
        assert_eq!(timeline.sent(), MS(1_500));
        assert_eq!(
            silence_to_emit(MS(2_000), timeline.sent(), 44_100, MS(300), QUIET_AFTER),
            22_050
        );
    }

    #[test]
    fn time_spent_setting_up_the_tap_is_not_filled_with_silence() {
        let (tx, mut rx) = mpsc::channel(64);
        let format = AudioFormat {
            sample_rate_hz: 1_000,
            channels: 1,
        };
        let timeline = Timeline {
            start: Instant::now() - MS(600),
            ..Timeline::new()
        };
        timeline.mark_live();
        fill_silence(&timeline, format, &tx);
        drop(tx);
        let mut filled = 0;
        while let Ok(f) = rx.try_recv() {
            filled += f.frames_per_channel();
        }
        assert!(filled < 50, "filled {filled} ms of setup time");
    }

    #[test]
    fn filled_silence_is_sent_in_bounded_zero_frames_on_the_timeline() {
        let (tx, mut rx) = mpsc::channel(64);
        let format = AudioFormat {
            sample_rate_hz: 1_000,
            channels: 2,
        };
        let timeline = Timeline {
            start: Instant::now() - MS(350),
            sent_ns: AtomicU64::new(100_000_000),
            ..Timeline::new()
        };
        fill_silence(&timeline, format, &tx);
        drop(tx);
        let mut frames = Vec::new();
        while let Ok(f) = rx.try_recv() {
            frames.push(f);
        }
        let total: usize = frames.iter().map(AudioFrame::frames_per_channel).sum();
        assert!((250..=260).contains(&total), "filled {total} frames");
        assert!(frames.iter().all(|f| f.frames_per_channel() <= 100));
        assert!(frames.iter().all(|f| f.samples.iter().all(|s| *s == 0.0)));
        assert_eq!(frames[0].captured_at_ns, 100_000_000);
        assert_eq!(timeline.sent(), MS(100 + total as u64));
    }

    fn asbd(flags: u32, bits: u32, channels: u32, rate: f64) -> AudioStreamBasicDescription {
        AudioStreamBasicDescription {
            mSampleRate: rate,
            mFormatID: u32::from_be_bytes(*b"lpcm"),
            mFormatFlags: flags,
            mBytesPerPacket: 4 * channels,
            mFramesPerPacket: 1,
            mBytesPerFrame: 4 * channels,
            mChannelsPerFrame: channels,
            mBitsPerChannel: bits,
            mReserved: 0,
        }
    }

    #[test]
    fn the_spike_tap_format_maps_to_interleaved_stereo() {
        // Observed on macOS 27.2: 48 kHz, 2 ch, 32-bit, flags 0x9.
        let layout = TapLayout::from_asbd(&asbd(0x9, 32, 2, 48_000.0)).unwrap();
        assert_eq!(
            layout,
            TapLayout {
                format: AudioFormat {
                    sample_rate_hz: 48_000,
                    channels: 2
                },
                non_interleaved: false,
            }
        );
    }

    #[test]
    fn a_non_interleaved_tap_is_flagged() {
        let flags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsNonInterleaved;
        let layout = TapLayout::from_asbd(&asbd(flags, 32, 2, 44_100.0)).unwrap();
        assert!(layout.non_interleaved);
        assert_eq!(layout.format.sample_rate_hz, 44_100);
    }

    #[test]
    fn integer_tap_formats_are_refused() {
        let outcome = TapLayout::from_asbd(&asbd(0xC, 16, 2, 48_000.0));
        assert!(matches!(
            outcome,
            Err(DomainError::AudioFormatUnsupported(_))
        ));
    }

    #[test]
    fn osstatus_codes_are_shown_as_four_characters() {
        assert_eq!(fourcc(i32::from_be_bytes(*b"!dev")), "'!dev'");
        assert_eq!(fourcc(-50), "-50");
    }

    #[test]
    fn the_default_output_is_alive_and_an_unknown_device_is_not() {
        // CI runners may have no output device.
        if let Ok((device, _uid)) = default_output() {
            assert!(device_alive(device));
        }
        assert!(!device_alive(0xDEAD));
    }

    #[test]
    fn a_granted_permission_lets_the_tap_start_without_asking() {
        let mut asked = false;
        assert!(tap_permission(TccStatus::Granted, || asked = true).is_ok());
        assert!(!asked);
    }

    #[test]
    fn a_missing_permission_is_requested_once_and_hands_over_to_the_fallback() {
        let mut asks = 0;
        let outcome = tap_permission(TccStatus::NotGranted, || asks += 1);
        assert_eq!(asks, 1);
        match outcome {
            Err(DomainError::AudioDeviceUnavailable(msg)) => {
                assert!(msg.contains("System Audio Recording"), "{msg}")
            }
            other => panic!("unexpected outcome: {other:?}"),
        }
    }

    #[test]
    fn an_unknown_permission_state_hands_over_without_asking() {
        // The private TCC call is gone or changed: never risk a tap that
        // would record silence.
        let mut asked = false;
        let outcome = tap_permission(TccStatus::Unknown, || asked = true);
        assert!(matches!(
            outcome,
            Err(DomainError::AudioDeviceUnavailable(_))
        ));
        assert!(!asked);
    }

    #[test]
    fn only_a_zero_preflight_counts_as_granted() {
        // Observed: 0 granted, 2 denied or not asked yet.
        assert_eq!(TccStatus::from_preflight(Some(0)), TccStatus::Granted);
        assert_eq!(TccStatus::from_preflight(Some(1)), TccStatus::NotGranted);
        assert_eq!(TccStatus::from_preflight(Some(2)), TccStatus::NotGranted);
        assert_eq!(TccStatus::from_preflight(None), TccStatus::Unknown);
    }

    #[test]
    fn this_mac_has_process_taps() {
        // CI and dev Macs run macOS 14.2+; older systems take the
        // ScreenCaptureKit fallback instead.
        assert!(CoreAudioTapCapture::is_supported());
    }

    #[tokio::test]
    async fn microphone_source_is_rejected() {
        let cap = CoreAudioTapCapture::new();
        let err = cap
            .list_devices(AudioSource::Microphone)
            .await
            .expect_err("must reject microphone source");
        assert!(matches!(err, DomainError::AudioDeviceUnavailable(_)));
        let outcome = cap.start(CaptureSpec::default_microphone()).await;
        assert!(matches!(
            outcome,
            Err(DomainError::AudioDeviceUnavailable(_))
        ));
    }
}
