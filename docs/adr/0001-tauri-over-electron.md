# ADR-0001: Tauri 2.x over Electron for the desktop shell

- **Status:** accepted
- **Date:** 2026-04-18
- **Deciders:** Tech Lead
- **Technical story:** Sprint 0 — stack lock-in before any shell code ships.

## Context and problem statement

EchoNote is a privacy-first desktop app that must ship on macOS, Windows and
Linux, embed native C/C++ libraries (whisper.cpp, llama.cpp, ONNX Runtime),
capture system and microphone audio simultaneously, and remain lightweight
enough to sit idle for hours on a laptop without impacting battery.

Which desktop framework best satisfies these constraints while keeping bundle
size, memory footprint and maintenance cost under control?

## Decision drivers

- **Privacy by design.** The shell must not ship with surprise telemetry or
  remote update channels we do not control.
- **Portability.** All three desktops are first-class targets from day one.
- **Footprint.** Installer under 50 MB, RAM idle under 350 MB (Balanced profile
  target from `DEVELOPMENT_PLAN.md` §4.1).
- **Native FFI.** First-class access to Rust/C/C++ libraries without process
  boundaries.
- **Auditable permission model.** Network allowlist, filesystem scoping and
  capability manifests that can be reviewed by security.
- **Future mobile option.** v2 roadmap contemplates iOS/Android; avoiding a
  full rewrite is preferable.

## Considered options

1. **Tauri 2.x** with a Rust backend and a web-based frontend (React).
2. **Electron** with Node.js backend and Chromium frontend.
3. **Qt (QML / Widgets)** with a Rust backend via `cxx-qt`.
4. **Native per OS** (SwiftUI on macOS, WinUI on Windows, GTK on Linux) with a
   shared Rust core.

## Decision outcome

**Chosen option: Tauri 2.x.**

Tauri pairs a Rust backend with the host OS webview, producing installers an
order of magnitude smaller than Electron, memory footprints roughly 5× lower,
direct FFI to our native dependencies, a capability-based permission model
that is trivial to audit, and a clean path to iOS/Android in v2.

## Consequences

### Positive

- Installer footprint matches the target under 50 MB with room to spare.
- Shell process memory idles ~30–50 MB on Apple Silicon.
- Rust backend is the natural host for `whisper-rs`, `llama-cpp-rs`, `ort`
  and `sqlx` without a separate child process.
- Capability manifests (`src-tauri/capabilities/*.json`) make security review
  an append-only diff exercise.
- React frontend leverages the existing UI ecosystem and shadcn/ui design
  system described in `docs/DESIGN.md`.
- Same codebase retargets iOS/Android in Tauri 2 should v2 pursue mobile.

### Negative

- The team must ramp up on Rust for any backend change. Mitigated by a clear
  layering (see ADR-0010) and by limiting unsafe code to adapter crates.
- The plugin ecosystem is smaller than Electron's. We compensate by owning
  our own Rust adapters for audio, ASR, LLM and storage.
- Webview behaviour differs per OS (WKWebView on macOS, WebView2 on Windows,
  WebKitGTK on Linux). Cross-platform snapshot testing will be required.

### Neutral

- Updates use Tauri's signed updater with our own endpoint, avoiding GitHub
  Releases as an infrastructure dependency at runtime.

## Pros and cons of the options

### 1. Tauri 2.x (chosen)

- **Pros.** Tiny footprint; Rust backend; capability-based permissions;
  iOS/Android path; active 2.x release train.
- **Cons.** Rust learning curve; smaller plugin ecosystem; webview drift.

### 2. Electron

- **Pros.** Massive ecosystem; fastest time to market; uniform Chromium on
  every OS.
- **Cons.** 80–150 MB installer and 200–300 MB idle RAM contradict our
  non-functional targets; Node.js adds a second process boundary to reach
  our Rust / C++ adapters; npm supply chain is a broad attack surface for
  a privacy-first app.

### 3. Qt with cxx-qt

- **Pros.** Mature cross-platform toolkit; good native look and feel;
  declarative UI with QML.
- **Cons.** Commercial licensing considerations; QML developer pool is
  small; less idiomatic for web-based designers authoring the system;
  mobile story is complicated for our case.

### 4. Native per OS

- **Pros.** Best possible native integration; smallest binaries.
- **Cons.** Triples the UI surface area of the team; incompatible with a
  two-engineer MVP; duplicates effort on a non-differentiating layer.

## References

- `docs/ARCHITECTURE.md` §3.2.1 — original rationale summary.
- `docs/DEVELOPMENT_PLAN.md` §4.1 — non-functional performance targets.
- Tauri 2.0 release notes — https://tauri.app/blog/tauri-20/
