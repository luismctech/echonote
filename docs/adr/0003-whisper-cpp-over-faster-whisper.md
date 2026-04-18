# ADR-0003: whisper.cpp via `whisper-rs` for on-device ASR

- **Status:** accepted
- **Date:** 2026-04-18
- **Deciders:** Tech Lead, ML Engineer
- **Technical story:** Sprint 0 — confirm the ASR runtime before the audio
  and transcription adapters land in Days 5–6.

## Context and problem statement

EchoNote transcribes meetings entirely on-device. The ASR runtime we pick
governs the distribution story (bundle size, installer complexity),
cross-platform reach, hardware acceleration on Apple Silicon, and whether
the team can extend the pipeline without rewriting a chunk of product code.

The decision window is now: the audio capture adapter (Sprint 0 day 5) and
the ASR adapter (Sprint 0 day 6) both depend on this being final.

## Decision drivers

- **Distribution.** We ship signed, notarised binaries on three platforms.
  Anything that requires users to install Python or a GPU runtime is ruled
  out of the default path.
- **Quality targets** from `DEVELOPMENT_PLAN.md` §4.2:
  WER < 10 % for Spanish and < 8 % for English on clean audio.
- **Performance budget.** Refine 30 minutes of audio in under 90 s on a
  mid-range Apple Silicon laptop (Balanced profile).
- **Hardware acceleration.** Must use the Apple Neural Engine via CoreML,
  CUDA on NVIDIA hosts, and Vulkan on cross-vendor GPUs, all from the same
  API surface.
- **Licensing.** The runtime must allow static linking into a proprietary
  installer without copyleft implications.
- **Operational simplicity.** No sidecar process, no separate lifecycle to
  manage, no surprise dependencies at runtime.

## Considered options

1. **whisper.cpp via `whisper-rs`.** C++ library, statically linkable;
   CoreML/CUDA/Vulkan/Metal backends; MIT licensed.
2. **faster-whisper (CTranslate2).** Python wheel; fastest CPU performance;
   batteries-included inference server capabilities.
3. **openai-whisper in PyTorch.** Reference implementation.
4. **NVIDIA Canary / Qwen-Audio / distil-whisper.** Newer, higher quality
   models with their own runtimes.

## Decision outcome

**Chosen option: whisper.cpp through the `whisper-rs` Rust crate, with the
CoreML backend enabled on macOS and Metal acceleration for non-ANE paths.**

Whisper models themselves (small, medium, large-v3-turbo) remain pluggable:
this ADR is about the runtime, not the weights. The Phase 0 benchmark
(Sprint 0 days 9–10) compares three model sizes to pick the per-profile
defaults declared in `docs/README.md`.

## Consequences

### Positive

- Single static library compiled into the Rust binary; no runtime Python
  and no extra install step for users.
- CoreML backend on Apple Silicon routes attention kernels through the
  Neural Engine, giving real-time factors well under 0.1 on M-series chips.
- First-class Metal acceleration for hosts without ANE (Intel Macs) keeps
  us honest on a single codebase.
- `whisper-rs` exposes timestamps, word-level probabilities and `language`
  auto-detection out of the box — everything the streaming + refinement
  pipeline (ADR-0006) needs.
- MIT licensing for the runtime; all candidate models (Whisper tiny …
  large-v3-turbo) are also MIT.
- The same runtime ships to Windows (x64, ARM64) and Linux unchanged.

### Negative

- Raw CPU performance on hosts without ANE/GPU is 20–30 % slower than
  CTranslate2. Accepted because our Balanced and Quality profiles both
  target machines with hardware acceleration, and the Lite profile
  purposely trades speed for footprint.
- Build complexity: whisper.cpp is compiled from source on first install.
  Mitigated by `sccache` on developer machines and `Swatinem/rust-cache`
  in CI, plus feature-gated backends so we do not ship everything to every
  platform.
- Thread safety caveats: `whisper-rs` contexts are not `Sync`. We wrap
  them behind a dedicated task with a channel, consistent with the
  streaming design in ADR-0006.

### Neutral

- Switching to a different runtime later is a change inside the `echo-asr`
  crate only; the domain port `Transcriber` shields every other layer.

## Pros and cons of the options

### 1. whisper.cpp via whisper-rs (chosen)

- **Pros.** Static link, zero runtime dependencies, CoreML + CUDA + Metal
  + Vulkan backends, mature Rust bindings, MIT licence.
- **Cons.** 20–30 % slower than faster-whisper on raw CPU; each backend is
  its own build flag to manage.

### 2. faster-whisper (CTranslate2)

- **Pros.** Fastest CPU inference; excellent batch scheduling; thriving
  community.
- **Cons.** Requires Python runtime or a shimmed native CTranslate2 build;
  integrating into a signed, notarised desktop app turns into a
  multi-month packaging project (`py2app`, `PyOxidizer`, venv relocation)
  we do not want to own.

### 3. openai-whisper in PyTorch

- **Pros.** Reference implementation; easy research path.
- **Cons.** Heavy runtime (PyTorch + NumPy), cold start multi-second,
  deployment on user machines infeasible.

### 4. NVIDIA Canary / Qwen-Audio / distil-whisper

- **Pros.** Canary-1B tops WER leaderboards for English + multilingual;
  distil-whisper is smaller and fast.
- **Cons.** Canary 1B weights are under a non-commercial licence;
  Qwen-Audio is audio-captioning-oriented; distil-whisper is restricted to
  English. None fits the MVP constraint of ES + EN, permissive licensing,
  runnable in Lite profile.

## References

- `docs/ARCHITECTURE.md` §3.2.5 — initial rationale summary.
- `docs/DEVELOPMENT_PLAN.md` §4.2 and §5.2 — quality targets and Phase 0
  benchmark milestone.
- whisper.cpp repository — https://github.com/ggml-org/whisper.cpp
- whisper-rs crate — https://crates.io/crates/whisper-rs
