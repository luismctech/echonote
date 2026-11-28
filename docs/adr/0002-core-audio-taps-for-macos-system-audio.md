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
   **`dlsym` at runtime**. Rust `extern` declarations are not weak; the linker binds them lazily
   (`dyld_info -fixups`: `lazy-bind`), so calling them on macOS 11–14.1 would abort the process.
   With `dlsym`, a missing symbol is a normal "not supported" and doubles as the version gate.
   cpal 0.18 already imports both (lazily) for a loopback path EchoNote does not use on macOS.
   Launching on macOS older than 14.2 has not been tested.
4. The adapter **fills silence** itself. When no buffer arrives for 200 ms, it sends zero frames
   up to wall-clock time, so the stream keeps the session timeline the way a microphone does.
   The timeline counts nanoseconds of audio sent, measured from the moment the tap went live, so
   a rebuilt tap at another rate continues it. The time spent setting up the tap is not counted.
5. **Permission gate.** A tap without the "System Audio Recording" permission starts without
   error. It then delivers only zeros, observed on macOS 27.2: with the permission denied the
   IOProc ran 260 times in 3 s, with every sample 0.0 while a tone played. While the prompt is
   unanswered, `AudioDeviceStart` blocks for up to 30 s.

   macOS has no public way to read this permission, so the adapter reads it through the private
   TCC SPI (`TCCAccessPreflight`, loaded with `dlopen`). It returned 0 when granted and 2 when
   denied or not yet asked.
   - **Granted (`0`):** the tap starts.
   - **Any other answer:** the start fails, so ScreenCaptureKit takes that session. macOS is
     asked once, through `TCCAccessRequest`, which does not block, so the next session can use
     the tap.
   - **SPI missing (`dlopen` fails):** also falls back to ScreenCaptureKit, without asking.
   Decided by the owner after the denial test.
6. Switching the default output keeps a global tap working (observed: about 1 s gap, filled with
   silence). If the device that clocks the aggregate disappears, the tap is rebuilt on the new
   default output with the same backoff as the microphone.

## Consequences

- On macOS 14.2+, system audio no longer needs the Screen Recording permission. macOS asks for
  "System Audio Recording" the first time.
- There is more `unsafe` FFI in `echo-audio` (Core Audio C API, Objective-C via objc2). It is
  contained in one module with ordered teardown in `Drop`.
- No new crates: the `objc2*` crates are direct dependencies now, at the versions already in the
  lock.
- The permission check uses a **private API**. That rules out the Mac App Store while it exists;
  EchoNote ships with Developer ID and notarization. If Apple removes or changes the call, only
  `0` counts as granted, so the worst case is today's ScreenCaptureKit path. The one bad case is
  the call answering `0` while the permission is denied, which would bring back silent capture.
- When neither permission has been granted, the first session can show two prompts: System Audio
  Recording, requested for the next session, and Screen Recording, for the ScreenCaptureKit
  fallback. A follow-up is to ask for System Audio Recording during onboarding.
- The Screen Recording usage string stays in `Info.plist` for the fallback.
- The fallback applies when a session **starts**. If the tap's clock device disappears and the
  tap cannot be rebuilt, that session's system audio ends; ScreenCaptureKit does not take over
  mid-session. That can happen when a Mac's only output, such as a headset, disconnects.
- If the permission is revoked mid-session, the tap goes quiet and the stream carries silence. A
  debug log marks each switch into and out of the quiet state.

## Alternatives

| Option | Why not (now) |
|---|---|
| Keep ScreenCaptureKit only | Keeps the Screen Recording prompt and the video stream; this is what the audit flagged. |
| `cidre` bindings | `objc2-core-audio` already covers every call needed and is in the lock. |
| Per-process taps (only the meeting app) | Needs choosing the app; a global tap minus EchoNote matches today's behaviour. |
| Link the 14.2 symbols weakly (`-weak_framework` / `#[linkage]`) | Unstable or build-wide; `dlsym` is local to the module. |
| Detect denial by a heuristic (callbacks with only zeros) | No private API, but an app that keeps the output running with digital silence (a call where nobody talks) would look like a denial. The first start can also block for 30 s while the prompt is unanswered. |
| No detection | A denied permission would record silence without any signal. |
