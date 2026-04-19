# Sprint 0 — Retrospective

> **Period:** 2026-04-01 → 2026-04-18 (10 working days)
> **Tag:** `v0.1.0-sprint0`
> **Branch:** `develop` → merged to `main` via PR
> **Goal of the sprint:** _"Get from an empty repo to a Tauri app that can stream
> live mic audio through Whisper, persist meetings into SQLite, and ship a
> reproducible WER bench gating CI."_ ✅ Achieved.

---

## 1. What shipped

| Day | Theme | Commit | Highlights |
|---|---|---|---|
| 1 | Workspace scaffolding | `7f2644a` | 9 Rust crates, hexagonal layout, MSRV pinned |
| 2 | CI + ADRs | `675c014` | GitHub Actions (rustfmt + macOS + Linux), Dependabot, ADR 0001–0003 |
| 3 | Onboarding | `386f842` | `CONTRIBUTING.md`, code of conduct, security policy, bootstrap script |
| 4 | Tauri shell | `e424cf1` | Tauri 2 host + React 18 frontend + first typed IPC (`health_check`) |
| 5 | Mic capture | `9549255` | `cpal` adapter, WAV sink, `record` / `record-devices` CLI |
| 6 | Transcription | `15737ca` | `rubato` resampler, `whisper-rs` adapter (Metal), `transcribe` CLI |
| 7 | Streaming | `5d23a4e` | `StreamingPipeline`, energy VAD, `stream` CLI + Tauri events |
| 7 | UI for streaming | `185b624` | Live transcript view wired to `start_streaming`/`stop_streaming` |
| 8 | Persistence | `dbc9990` | `SqliteMeetingStore`, `MeetingRecorder`, meetings CLI + sidebar UI |
| 9 | Benchmarks | `d2fd47f` | WER + LLM bench scaffolding, fixtures contract, `bench.yml` workflow |

**Totals at tag time:**

- 6 247 lines of Rust across 9 library crates + the Tauri shell.
- 835 lines of TypeScript/React.
- **50 unit + integration tests, 0 ignored, 0 failing.**
- 3 ADRs merged (Tauri vs Electron, Rust+React stack, whisper.cpp choice).
- 1 reproducible Phase-0 WER baseline at **8.40 %** (`base.en`, 5 fixtures).

---

## 2. Architecture decisions taken (summary)

These were either captured as ADRs or as in-line decisions worth surfacing
before Sprint 1 planning:

1. **Hexagonal layering enforced from day 1.** `echo-domain` has zero
   non-test deps, `echo-app` only depends on the domain, infra crates
   (`echo-audio`, `echo-asr`, `echo-storage`, `echo-llm`, `echo-diarize`)
   sit on the outside. Tauri (`src-tauri/echo-shell`) is treated as
   _another_ adapter, not as the core.
2. **Async only at the edges.** Domain ports use `async_trait` so adapters
   can be async, but pure domain logic stays synchronous and trivially
   unit-testable (`compute_wer`, normalization, segment math).
3. **Streaming pipeline is bounded and back-pressured.** Channels use
   `mpsc::channel(N)` with `try_send` so a slow consumer never causes
   audio drops mid-meeting; we drop transcript events instead and warn.
4. **Persistence is event-driven, not pipeline-coupled.** `MeetingRecorder`
   subscribes to `TranscriptEvent`s; the pipeline doesn't know SQLite
   exists. This kept Day 7 mergeable before Day 8 was even designed.
5. **Per-chunk durable writes.** Every Whisper chunk is committed inside
   a transaction with `synchronous=NORMAL` + WAL. A crash mid-meeting
   loses ≤ one chunk window (~5 s), never the whole session.
6. **`MeetingId` is UUIDv7.** Time-ordered → free chronological clustering
   in indexes and listings, no extra sort column needed.
7. **Bench fixtures are synthetic and git-ignored.** Only the gold
   transcripts and the `say`+`afconvert` script are committed; CI
   regenerates the WAVs every run. Avoids copyright issues and gives us
   reproducible RTF measurements across runners.
8. **Two profiles already separate.** `cargo` debug for the dev loop,
   `cargo --release` mandatory for any RTF/WER measurement (debug runs
   are roughly 8× slower in Whisper inference).

---

## 3. What worked well

- **Vertical slices per day.** Every day shipped something runnable end-to-end
  (CLI command, IPC roundtrip, or UI flow) instead of horizontal "all the
  domain types first" work. Made each commit independently reviewable.
- **CLI-first, UI-second.** `echo-proto` was always the first consumer of a
  new capability. By the time the Tauri UI needed `start_streaming`, the
  pipeline had been exercised headlessly for a full day.
- **Ports & adapters paid for themselves immediately.** Replacing the file
  WAV sink with the live capture stream on Day 7 was a one-line change in
  `echo-app` because `AudioCapture` was already a trait.
- **Whisper.cpp + Metal is fast enough** to make 5-second windows feel
  almost-live (RTF ≈ 0.08 with `base.en`). No need for `tiny.en` in Phase 0.
- **Energy-based VAD is a good-enough placeholder.** Roughly 30 % of chunks
  during a real call get gated as silence, saving compute. Silero will only
  upgrade quality, not unlock anything new architecturally.
- **CI gates were cheap to keep green.** `rustfmt --check`, `clippy -D
  warnings` and `cargo test --workspace` ran in <2 min on every push.

## 4. What was harder than expected

- **Tauri 2 capability manifests** are noisier than the docs suggest;
  every new IPC command needs both a `#[tauri::command]` registration
  and a capability entry. We'll codify a snippet in `CONTRIBUTING.md`
  during Sprint 1.
- **`AppState::initialize` had to become `async`.** Opening SQLite +
  running migrations at startup forced a `tauri::async_runtime::block_on`
  in the `setup` hook. Works fine but is worth revisiting if startup
  latency becomes user-visible (>200 ms).
- **`cargo test` for `echo-asr` is slow** because it links Whisper natively.
  Fine in CI, mildly annoying locally — the workaround is `cargo test
  -p <crate>` instead of full-workspace runs.
- **macOS-only fixture script.** `say` + `afconvert` is convenient but it
  means Linux contributors can't regenerate fixtures locally. We accept
  this for Phase 0 (macOS is the primary target) but Sprint 1+ should add
  a Piper-TTS fallback so Linux dev loops aren't blind.

## 5. Risks & open issues carrying into Sprint 1

| # | Risk | Severity | Mitigation planned for Sprint 1 |
|---|---|---|---|
| R1 | System-audio capture (ScreenCaptureKit on macOS, WASAPI loopback on Win) is not yet wired | High | Day 1–3 of Sprint 1 |
| R2 | Diarization completely absent — single-speaker mental model leaks into UI | Medium | ADR 0004 + ONNX Silero/3D-Speaker spike |
| R3 | No FTS5 yet — `meetings list` is a flat scan | Low | Add virtual table in Sprint 1 day 4 |
| R4 | Bench gate hardcoded to `base.en` baseline; quality models (medium/large) unmeasured | Medium | Extend `bench.yml` matrix to medium + medium-q5_0 |
| R5 | Frontend has no error boundary; an IPC failure during streaming greys the live pane | Low | Add ErrorBoundary + toast layer in Sprint 1 day 1 |
| R6 | No telemetry / structured event log surfaced to the user | Low | `echo-telemetry` crate exists but is unused; light it up in Sprint 1 |

## 6. Phase-0 quality gates (current snapshot)

| Gate | Target | Current | Status |
|---|---|---|---|
| Streaming RTF p50 (`base.en`, M1 Pro, 5 s windows) | < 0.5 | **0.08** | ✅ huge headroom |
| WER on synthetic EN fixtures (`base.en`) | < 25 % | **8.40 %** | ✅ |
| Meeting persistence durability (chunks lost on crash) | ≤ 1 chunk (~5 s) | by design | ✅ |
| CI wall-clock (push → green on all 3 jobs) | < 5 min | ~3 min | ✅ |
| Workspace tests | pass + zero ignored | 50 / 50 | ✅ |
| Clippy warnings on workspace | 0 with `-D warnings` | 0 | ✅ |

Detailed numbers: [`docs/benchmarks/PHASE-0.md`](./benchmarks/PHASE-0.md).

## 7. Sprint 1 — proposed scope (entry conditions met)

Pre-conditions to start Sprint 1 are now satisfied:

- [x] Live mic streaming end-to-end on Tauri.
- [x] Persistence layer with reload from disk.
- [x] Reproducible WER bench in CI.
- [x] CLI parity with the UI for every backend capability.
- [x] `develop` is green and tagged `v0.1.0-sprint0`.

Suggested Sprint 1 backlog (each will get a GitHub issue):

1. **System-audio capture (macOS).** ScreenCaptureKit adapter behind the
   `AudioCapture` trait. Two-track meeting model in the domain.
2. **VAD upgrade to Silero v5.** Replace energy gate, keep the same port.
3. **Diarization MVP.** 3D-Speaker ERes2Net, per-track clustering, surface
   `Speaker` in the UI.
4. **FTS5-backed meeting search.** Virtual table + `meetings search "<q>"`
   CLI + search bar in the sidebar.
5. **Summary v1 (template: general).** `echo-llm` + Qwen 2.5 7B Q4_K_M,
   summary stored as a `Meeting` projection.
6. **Frontend hardening.** ErrorBoundary, toast layer, recording state
   machine made explicit (Idle → Recording → Stopping → Persisted).
7. **Bench matrix.** Extend `bench.yml` to also benchmark `small.en` and
   `medium.en`, publish results into `docs/benchmarks/PHASE-1/`.

---

## 8. Numbers worth bragging about

- 0 → live transcription + persistence + bench in **10 working days**.
- **3 ms median IPC latency** for `health_check`, **<50 ms** for
  `start_streaming` setup (measured ad-hoc with `tracing` spans).
- **8.40 % WER** on `base.en` is competitive with cloud baselines from
  ~2 years ago, running 100 % offline on a laptop.
- Repo is **6.3 K LoC of Rust** plus 0.8 K LoC of React; small enough that
  any contributor can read every file in an afternoon.

---

_Approved by:_ Alberto Cruz · _Date:_ 2026-04-18
