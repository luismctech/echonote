# ADR 0002: Core Audio process taps for macOS system audio

- **Status:** Accepted
- **Date:** 2026-11-27
- **Feature:** `core-audio-taps` (capture-audit point 6)
- **Deciders:** project owner

## Context

EchoNote captures the "other side of the call" on macOS through ScreenCaptureKit
(`crates/echo-audio/src/capture/macos.rs`). That works on macOS 13+, but:

- it needs the **Screen Recording** permission, which users read as "this app watches my
  screen", although only audio is used;
- an `SCStream` always captures video, even when it is reduced to 2×2 px at 1 fps;
- the permission check (`CGPreflightScreenCaptureAccess`) can report "not granted" until the
  app restarts after a grant.

macOS 14.2 added **Core Audio process taps** (`CATapDescription`,
`AudioHardwareCreateProcessTap`): a tap of the audio that processes send to the output, read
through a private aggregate device. It asks for **System Audio Recording** only
(`NSAudioCaptureUsageDescription`) and has no video stream. The app's minimum macOS is 11.0.

A spike on macOS 27.2 (`odd/tasks/core-audio-taps.md`, T0) confirmed:

- tap + private aggregate + IOProc work through `objc2-core-audio` 0.3.2, which cpal already
  pulls in;
- the tap delivers 48 kHz stereo Float32, interleaved, in a single buffer;
- excluding a process object removes its audio;
- **while nothing plays the IOProc is not called at all**.

## Decision

1. On macOS 14.2+, system audio (`SystemOutput` and the system half of `Mixed`) comes from a
   **global process tap** that excludes EchoNote's own process and leaves the user's audio
   unmuted (`CoreAudioTapCapture`, `capture/macos_tap.rs`).
2. **ScreenCaptureKit stays as fallback**, through a generic `FallbackAudioCapture`. It is used
   on macOS older than 14.2 and whenever the tap cannot list or start.
   `RoutingAudioCapture::with_default_adapters` composes the two.
3. `AudioHardwareCreateProcessTap` / `AudioHardwareDestroyProcessTap` are resolved with
   **`dlsym` at runtime**. Rust `extern` declarations are strong imports, so naming them would
   stop the app from launching on macOS 11–14.1. A missing symbol doubles as the version gate.
4. The adapter **fills silence** itself. When no buffer arrives for 200 ms, it sends zero frames
   up to wall-clock time, so the stream keeps the session timeline the way a microphone does.
   The timeline counts nanoseconds of audio sent, measured from the moment the tap went live, so
   a rebuilt tap at another rate continues it. The time spent setting up the tap is not counted.
5. Switching the default output keeps a global tap working (observed: about 1 s gap, filled with
   silence). If the device that clocks the aggregate disappears, the tap is rebuilt on the new
   default output with the same backoff as the microphone.

## Consequences

- On macOS 14.2+, system audio no longer needs the Screen Recording permission. macOS asks for
  "System Audio Recording" the first time.
- There is more `unsafe` FFI in `echo-audio` (Core Audio C API, Objective-C via objc2). It is
  contained in one module with ordered teardown in `Drop`.
- No new crates: the `objc2*` crates are direct dependencies now, at the versions already in the
  lock.
- macOS gives no public way to check this permission before use. How a denied permission is
  detected is **open** (see Alternatives); until it is decided, a denied tap may deliver silence
  instead of falling back.
- The Screen Recording usage string stays in `Info.plist` for the fallback.

## Alternatives

| Option | Why not (now) |
|---|---|
| Keep ScreenCaptureKit only | Keeps the Screen Recording prompt and the video stream; this is what the audit flagged. |
| `cidre` bindings | `objc2-core-audio` already covers every call needed and is in the lock. |
| Per-process taps (only the meeting app) | Needs choosing the app; a global tap minus EchoNote matches today's behaviour. |
| Link the 14.2 symbols weakly (`-weak_framework` / `#[linkage]`) | Unstable or build-wide; `dlsym` is local to the module. |
| Detect denial with private TCC SPI (`TCCAccessPreflight`), as AudioCap does | Reliable, but private API. Pending owner decision. |
| Detect denial by a heuristic (callbacks with only zeros) | No private API, but apps that keep the output running with digital silence would look like a denial. |
