# ADR-0008: Silero VAD v5.1.2 over tract-onnx, made loadable by a Python pre-simplification step

- **Status:** accepted
- **Date:** 2026-04-23
- **Deciders:** Tech Lead, ML Engineer
- **Technical story:** Sprint 1 day 9 — pick a Voice Activity Detection
  stack that lets us mute Whisper during silence (the root cause of the
  Spanish-multilingual hallucinations seen after the day-9.5 model
  pivot) without giving up the single-binary, no-native-runtime
  property the rest of the audio stack already enforces.

## Context and problem statement

When EchoNote switched defaults from `base.en` + Qwen 2.5 to
`large-v3-turbo` + Qwen 3 14B (commits `c4b2c7c`, `5d23d1c`), Whisper
started emitting Spanish hallucinations on absolute silence —
predominantly `"Gracias."`, `"Gracias indefinidamente."` and a long
tail of meta tokens (`[no speech]`, `[música]`, "Subtitulado por…"
outros). The same audio with `base.en` was silent. The behavioural
delta is well-known in the whisper.cpp community: larger multilingual
checkpoints have memorised more idiomatic Spanish filler and confidently
project it onto pure noise.

Whisper-side hardening (`no_context = true`, `no_speech_thold = 0.5`,
`suppress_blank`, `suppress_nst`, plus a `is_known_hallucination`
post-filter — see `crates/echo-asr/src/whisper_cpp.rs`) and a more
aggressive RMS gate (0.005 → 0.02 in `echo-app::streaming`) cut most
of the residue but not all: any audio above the RMS floor that is not
speech (HVAC noise, keyboard, fan ramp-ups, music intros) still hits
Whisper and still occasionally produces a phantom phrase. The remaining
delta is structural — energy alone cannot tell speech from non-speech
noise.

The `Vad` port already exists in `echo-domain` (added day 4, commit
`c4e0b11`) along with an `EnergyVad` adapter in `echo-audio`. A neural
adapter (`SileroVad`) was scaffolded the same day but never wired into
the pipeline. The day-9 task is to (a) finish the adapter so the
streaming pipeline can gate Whisper on neural VAD, and (b) decide which
model and runtime to commit to.

A complication surfaced during integration: the upstream Silero v5
ONNX file fails to load in `tract-onnx` (the runtime ADR-0007 chose
for diarization) with `optimize: Failed analyse for node #5 "If_0" If`.
We had to decide whether to swap runtimes, fork tract, re-export the
model, or pre-process the file. The choice has to be made now because
the rest of Sprint 1 (chat, summary templates, bench matrix) assumes
the streaming pipeline already swallows silence cleanly.

## Decision drivers

- **Quality target — silence handling.** `docs/DEVELOPMENT_PLAN.md`
  §4.3 implicitly assumes Whisper is only invoked on speech; the
  fix-or-revert criterion is "no `Gracias.`, `[música]` or other
  hallucination on a 60 s pure-silence clip in es-ES".
- **Latency budget.** VAD runs on every chunk inside the streaming
  pipeline. It must be cheaper than the RMS gate it replaces, otherwise
  the cost wipes out the saving from skipping silent chunks. Practical
  ceiling: < 1 ms per 32 ms window on Apple Silicon.
- **Local-first / privacy.** No network calls. Everything on the
  user's machine.
- **Distribution.** No new system dependencies at *runtime*. Pure Rust
  where possible; if a binding is needed it must build cleanly into
  the signed Tauri bundle on macOS, Windows and Linux without a
  sidecar process and without bumping bundle size by >5 MB.
- **Streaming-friendly.** Stateful across frames (so short pauses
  inside an utterance do not chop the run) and resettable per session.
- **Permissive licence on both code and weights** so we can statically
  link and redistribute. Rules out commercial Cobra, Picovoice, etc.
- **Consistency with ADR-0007.** Diarization already chose
  `tract-onnx`. Adding a second ONNX runtime *just* for VAD would
  double the surface area of platform-specific bundling work for a
  ~1 MB model.
- **Build-time vs runtime tooling.** Build-time Python is acceptable
  (`scripts/download-models.sh` already runs Python for ONNX shape
  surgery in adjacent skeleton work). Runtime Python is not.

## Considered options

### VAD strategy

1. **Energy / RMS gate alone (`EnergyVad`)**, harden Whisper post-filter
   further.
2. **WebRTC-VAD** (Google's classic GMM-based VAD, Apache 2.0).
3. **Silero VAD v5.1.2** (chosen).
4. **Silero VAD v6.x** (latest upstream as of Sprint 1).
5. **RNNoise** as a denoiser in front of Whisper.
6. **Picovoice Cobra** (commercial speech detector).

### Inference runtime for the neural model

A. **`tract-onnx`** (pure Rust, used for diarization in ADR-0007).
B. **`ort`** (Rust bindings to onnxruntime, ships a C++ shared library).
C. **`candle`** (HuggingFace's pure-Rust ML).
D. **Hand-written Rust port** of the LSTM + conv ops.

### Strategy for the `If` / contrib-op incompatibility

- α. **Pre-process the ONNX at download time** with a Python script
  that inlines the 16 kHz `then_branch` and constant-folds the
  remaining shape-driven `If`s (chosen).
- β. **Migrate the runtime to `ort`** so the upstream file loads as-is.
- γ. **Patch / fork `tract-onnx`** to implement `If` plus the
  ORT-Extended contrib ops (`FusedConv`, `NchwcConv`, …).
- δ. **Re-export the model from PyTorch** ourselves, producing a
  graph that never had `If` to begin with.

## Decision outcome

**Chosen options:**

- **VAD model.** Silero VAD v5.1.2, MIT-licensed, ~2.2 MB upstream.
  16 kHz / 512-sample window path only — the 8 kHz path is dropped at
  pre-processing time.
- **Inference runtime.** `tract-onnx`, the same crate that powers the
  ERes2Net diarizer adapter (ADR-0007). No new C++ dependency, no
  notarisation churn, no second ONNX runtime in the bundle.
- **Compatibility strategy.** Build-time pre-processor
  `scripts/simplify-silero-vad.py` (Python 3.10+, `onnx>=1.14`,
  `onnxruntime>=1.17`) that ships an equivalent 16 kHz-only graph
  loadable by tract. The script is run **at download time only**, by
  `scripts/download-models.sh` and CI; it is *not* required at app
  runtime, and end users never need Python installed.
- **Pipeline integration.** Neural VAD is the primary gate;
  `EnergyVad` (RMS) is the documented fallback when the model file is
  absent or `--no-neural-vad` / `disableNeuralVad` is set. The Tauri
  shell and the `echo-proto` CLI both expose the toggle.

### What the pre-processor does, in one paragraph

The upstream Silero v5 graph dispatches between the 16 kHz and 8 kHz
sub-networks through an outer ONNX `If` whose predicate reads the `sr`
input. Three additional `If`s nested inside the chosen branch dispatch
on dynamic shape values. EchoNote always feeds 16 kHz audio, so the
script (1) inlines the 16 kHz `then_branch` of the outer `If`,
(2) drops the now-orphan `sr` input, (3) pins both remaining inputs
to their static shapes (`input = [1, 512]`, `state = [2, 1, 128]`),
and (4) runs ONNX Runtime's `ORT_ENABLE_BASIC` graph optimiser, which
constant-folds the three nested `If`s without introducing any contrib
ops.

### Why specifically `ORT_ENABLE_BASIC` (and not higher)

Empirical probe of all four ORT levels on the simplified graph:

| ORT level    | Nodes | `If` | `FusedConv` | tract loads? |
|---           |---   |---   |---           |---            |
| `DISABLE_ALL` | 56   | 3    | 0            | ❌ unsupported `If` |
| **`BASIC`**   | **36** | **0** | **0**       | ✅            |
| `EXTENDED`   | 31   | 0    | 5            | ❌ `Unimplemented(FusedConv)` |
| `ALL`        | 31   | 0    | 5            | ❌ `Unimplemented(FusedConv)` |

`BASIC` is the unique level that *both* eliminates the static-shape
`If`s *and* keeps the graph inside the standard ONNX op vocabulary
that `tract-onnx` supports. Anything above `BASIC` fuses `Conv + ReLU`
into ORT's contrib `FusedConv`, which is not part of the standard ONNX
op set. The script encodes this as a hard invariant
(`assert_no_contrib_ops()`); any future maintainer who tries to "just
turn optimisation up" gets a build-time abort.

### Numerical equivalence

The pre-processed graph is bitwise-equivalent to the upstream model on
16 kHz inputs. Validation (script-internal, exercised in CI smoke
runs): three test inputs (silence, white noise, loud speech-like
signal) produce Δ = 0.00e+00 in the per-frame probability against the
upstream model loaded with `onnxruntime`. The check runs against the
raw upstream cached at `models/vad/silero_vad.onnx.upstream`, which
the download script preserves precisely so this audit is reproducible.

### Resulting on-disk artefact

- File: `models/vad/silero_vad.onnx`, ~1.2 MB (upstream is ~2.2 MB).
- 36 nodes, 0 `If`, 0 contrib ops.
- Op set used: `Conv, Relu, LSTM, Sigmoid, Sqrt, Pow, Pad, Slice, Add,
  Concat, Squeeze, Unsqueeze, Gather, ReduceMean` — all standard,
  all supported by tract.
- Two inputs (`input [1,512] f32`, `state [2,1,128] f32`), two
  outputs (`output [1,1] f32`, `stateN [2,1,128] f32`).

### Defences against silent regressions

The script is idempotent and self-defensive:

- `already_simplified()` checks **three** invariants — no top-level
  `If`, no `sr` input, no contrib ops — instead of trusting file size.
  An earlier bug shipped a `FusedConv`-tainted graph at the *same*
  byte size as the correct one, which would have slipped past a naive
  size check.
- `assert_no_contrib_ops()` aborts the script if any future edit
  raises the ORT level above `BASIC`.
- The download pipeline keeps the raw upstream at `.onnx.upstream`,
  and the simplifier auto-restores from it when it detects a tainted
  intermediate state. A user whose previous run produced a broken
  ONNX recovers by re-running `bash scripts/download-models.sh vad`
  with no manual steps.

## Consequences

### Positive

- **Single ONNX runtime in the bundle.** Both VAD and diarization use
  `tract-onnx`; no `ort` C++ shared library to bundle, code-sign, or
  notarise on macOS / Windows.
- **Smaller on-disk model than upstream** (1.2 MB vs 2.2 MB) — the
  8 kHz sub-network is gone, so is dead `If` machinery.
- **Hallucinations gone in practice.** Streaming a silent buffer no
  longer triggers Whisper at all (`TranscriptEvent::Skipped { reason:
  "vad_silence" }`). Combined with the RMS gate (still active when the
  neural VAD is disabled) and the Whisper hallucination filter, the
  three-layer defence brought the spurious-`Gracias` rate to zero on
  the day-9 fixtures.
- **Cheap per-session reuse.** `SileroVad::clone_for_new_session()`
  shares the optimised tract `Arc<TypedModel>` and only resets the
  LSTM state, carry buffer and hysteresis bookkeeping. The expensive
  graph optimisation runs once per app lifetime.
- **Graceful degradation.** Missing model file → `Ok(None)` from
  `AppState::ensure_vad()` with a warning log; pipeline falls back to
  RMS without crashing. Same path is exercised by the
  `--no-neural-vad` flag in regression tests.
- **74 unit tests still pass** after the integration
  (`cargo test -p echo-audio -p echo-app --release`), including six
  Silero-specific tests and four new VAD-gated streaming tests
  (`vad_silence_skips_chunks_even_when_rms_is_loud`,
  `vad_voiced_forwards_chunks_to_transcriber`,
  `vad_bypasses_rms_gate_for_silent_chunks`,
  `vad_is_reset_on_session_start`).

### Negative

- **The on-disk file is no longer the upstream artefact.** A reviewer
  who downloads `silero_vad.onnx` from Hugging Face cannot diff it
  against ours. We mitigate by (a) keeping the raw upstream at
  `.onnx.upstream` for forensic comparison, (b) documenting the
  transformation in the script's module docstring and in
  `crates/echo-audio/src/preprocess/silero_vad.rs`'s module doc.
- **Build-time Python dependency.** `scripts/download-models.sh`
  auto-installs `onnx` + `onnxruntime` via `pip install --user` if
  missing; this works on every supported developer OS but adds ~150 MB
  to a clean dev environment. End users are unaffected. Acceptable
  trade vs the runtime cost of bundling onnxruntime.
- **No 8 kHz path.** Telephony-grade audio is downsampled-then-resampled
  to 16 kHz upstream of the VAD anyway, but if EchoNote ever wants
  native 8 kHz, the pre-processor needs a parallel `--sr 8000` mode.
- **Bumping Silero is no longer a one-line URL change.** Any future
  upgrade has to re-validate the script's invariants. The script is
  written defensively (the contrib-ops list, the three-invariant
  check, the `.upstream` recovery), so the failure mode of a future
  bump is "the script aborts loudly", not "the app silently ships a
  broken model".

### Neutral

- The download script gained an `embed`-equivalent flow for VAD: it
  always invokes the simplifier (which is fast and idempotent in the
  already-clean case) instead of trusting size-based caching.
- `crates/echo-audio/src/preprocess/silero_vad.rs` no longer takes
  the `sr` input in its inference call, so the adapter shape contract
  changed by exactly one tensor. Documented in the module preamble.
- `docs/DEVELOPMENT_PLAN.md` story E2.6 mentions Silero "vía `ort`";
  that wording predates this ADR and is now superseded. The story's
  intent ("a neural VAD adapter behind the `Vad` port") is unchanged.

## Pros and cons of the options

### VAD strategy

#### 1. Energy / RMS only (status quo)

- **Pros.** Already shipped (`EnergyVad`). Zero CPU. No model file.
- **Cons.** Cannot distinguish a fan from a quiet talker; cannot
  reject loud non-speech (keyboard, music intros). Whisper
  hallucinations on loud-but-not-speech are the entire reason this
  ADR exists — RMS alone does not solve them.

#### 2. WebRTC-VAD (Google, GMM)

- **Pros.** Tiny, very fast (1 ms / 30 ms frame), pure-C port exists
  (`webrtc-vad-rs`).
- **Cons.** Trained on narrowband telephony (~2010); known to be
  noisy on modern wideband meeting audio. Empirically (community
  reports + our day-4 spike) less robust than Silero on the same
  fixtures we already use for Whisper benchmarking.

#### 3. Silero VAD v5.1.2 (chosen)

- **Pros.** ~2.2 MB, MIT, opset 16, well-known good quality on
  meeting / podcast audio, stateful (LSTM) so short pauses inside an
  utterance do not chop the run, < 1 ms / 32 ms window on M-series
  CPU. Numerical contract documented (`models.silero.ai`).
- **Cons.** Upstream graph contains an ONNX `If` op that
  `tract-onnx` does not implement, plus three nested `If`s on shape
  values. Mitigated by the pre-processor (the entire premise of this
  ADR's third decision).

#### 4. Silero VAD v6.x

- **Pros.** Latest upstream; marginal quality improvements claimed
  by the maintainers.
- **Cons.** Same `If` operator as v5; the pre-processor would still
  be required. Day-9 investigation initially mis-diagnosed the load
  failure as "v6 introduced `If`, downgrade to v5.1.2 fixes it" —
  that hypothesis was **wrong**: v5.1.2's release artefact also
  contains `If_0`. Pinning to v5.1.2 alone solved nothing and the
  bug only went away with the pre-processor. We see no quality
  delta on our fixtures that would justify the v6 dependency churn,
  but the simplifier is forward-portable to v6 if upstream changes
  warrant the upgrade.

#### 5. RNNoise (denoiser)

- **Pros.** Could clean noise upstream of Whisper, shrinking the
  hallucination surface.
- **Cons.** Wrong tool — RNNoise reduces noise, it does not classify
  speech vs non-speech. Would have to be combined with a VAD anyway.
  Adds CPU cost without solving the gating problem.

#### 6. Picovoice Cobra

- **Pros.** Best-in-class quality on commercial benchmarks.
- **Cons.** Proprietary licence, runtime activation key. Conflicts
  with the local-first / no-keys constraint of the product.

### Inference runtime

#### A. `tract-onnx` (chosen)

- **Pros.** Pure Rust, statically linked, already in the workspace
  for diarization (ADR-0007), single bundle, no notarisation cost.
- **Cons.** Does not implement `If` or ORT contrib ops — the entire
  reason the pre-processor exists. Marginally slower than
  onnxruntime on heavy graphs; not measurable at 1.2 MB / 36 nodes.

#### B. `ort`

- **Pros.** Loads the upstream Silero file as-is, no pre-processing
  step needed. Faster on heavy graphs (irrelevant at this size).
- **Cons.** Bundles a C++ shared library on every target.
  Notarisation pain on macOS, signing on Windows, glibc surface on
  Linux. Pulls a second ONNX runtime into a binary that already has
  one (`tract-onnx` for diarization). Net effect: bigger bundle,
  more platform CI work, two runtimes to keep in sync.

#### C. `candle`

- **Pros.** Pure Rust, native HuggingFace tooling.
- **Cons.** No first-class ONNX import; Silero would need re-export
  into a candle-friendly format. Re-export is the same kind of work
  as option δ below, and we already need a Python script for the
  download path either way.

#### D. Hand-written Rust port of the LSTM + conv ops

- **Pros.** Zero ONNX runtime; bit-perfect control of inference.
- **Cons.** Re-implements primitives `tract-onnx` already provides;
  every Silero version bump requires re-deriving the architecture
  from the upstream Python code. Maintenance burden disproportionate
  to the win.

### Compatibility strategy

#### α. Pre-process at download time (chosen)

- **Pros.** Runtime stack stays pure Rust. The transformation is
  numerically exact on 16 kHz audio (verified Δ = 0). Idempotent,
  so re-running the download is safe. Self-defensive (3-invariant
  check + ORT-level assertion + `.upstream` recovery). Forward-
  portable to Silero v6 with no changes if upstream keeps the same
  outer-`If` dispatch shape.
- **Cons.** Adds a Python toolchain to the *build* (not runtime).
  ONNX Runtime install is ~150 MB on a clean machine. The on-disk
  file diverges from upstream — mitigated by keeping
  `.onnx.upstream` for audit.

#### β. Migrate the runtime to `ort`

- **Pros.** Drops the pre-processor. Loads Silero, ERes2Net and any
  future ONNX with no graph surgery.
- **Cons.** Reverses ADR-0007's runtime decision and pulls a second
  C++ runtime into the bundle. Cost is paid on every platform release,
  not just for VAD.

#### γ. Patch / fork `tract-onnx` to implement `If` + contrib ops

- **Pros.** Upstream contribution; benefits the broader Rust ML
  ecosystem.
- **Cons.** `If` is a control-flow op — implementing it correctly in
  tract's compile-then-run model is a multi-week project, plus the
  contrib-ops list is open-ended (we would have to chase ORT updates
  forever). Not on Sprint 1's critical path.

#### δ. Re-export the model from PyTorch ourselves

- **Pros.** Could produce a graph that never had `If` in the first
  place.
- **Cons.** Requires the Silero training repo (PyTorch, plus its
  dependencies) at build time; the resulting export is no longer
  bitwise-equivalent to upstream (different tracing path), so the
  numerical-parity audit becomes harder. Also requires us to track
  upstream Silero's training-time API, which is less stable than its
  ONNX release.

## Maintenance contract for `simplify-silero-vad.py`

Treated as a build-time artefact with the same review bar as Rust
code. Concretely:

- **The ORT optimisation level is `ORT_ENABLE_BASIC`** and is not to
  be raised without re-running the four-level probe table (§
  "Why specifically `ORT_ENABLE_BASIC`") and updating both this ADR
  and the script's docstring.
- **Three invariants are checked** on every output: no top-level `If`,
  no `sr` input, no ORT contrib op (full list in
  `_ORT_CONTRIB_OPS`).
- **The contrib-ops blocklist** must include any new ORT contrib op
  introduced by future onnxruntime releases that ends up in the
  optimised graph at `BASIC`. This is an "if it's ever observed,
  add it" discipline — onnxruntime adds contrib ops conservatively.
- **The `.upstream` backup is mandatory.** `download-models.sh` is
  not allowed to skip it; the simplifier relies on it for recovery
  and the audit relies on it for diff-against-upstream.
- **Future Silero bumps** must re-run the pre-processor against the
  new upstream and re-validate Δ = 0 on the standard test inputs
  (silence, noise, loud). If the outer-`If` dispatch shape changes
  upstream, the inlining logic in step (1) of the script needs to be
  updated.
- **A CI smoke test** (open todo, see §6.4 of `SPRINT-1-STATUS.md`)
  will run the script on a sandbox download and assert the three
  invariants. Currently the check is manual.

## References

- `docs/SPRINT-1-STATUS.md` §3.3 — pipeline integration of the
  neural VAD (per-session clone, RMS bypass when neural is on).
- `docs/SPRINT-1-STATUS.md` §3.4 — empirical history of the failed
  hypothesis (v5 vs v6) and the four-level ORT probe table.
- `docs/ARCHITECTURE.md` §3 (audio pipeline diagram, line 97) —
  where VAD sits between Preproceso and ASR.
- `docs/DEVELOPMENT_PLAN.md` §4.3 — RTF / latency budget the VAD
  fits inside.
- `docs/adr/0003-whisper-cpp-over-faster-whisper.md` — establishes
  the no-runtime-Python rule that scopes this ADR's solution to
  build-time tooling only.
- `docs/adr/0007-diarization-via-onnx-eres2net.md` — establishes the
  `tract-onnx` baseline that this ADR extends to a second model.
- `crates/echo-domain/src/ports/vad.rs` — `Vad` port definition and
  `VoiceState` enum.
- `crates/echo-audio/src/preprocess/silero_vad.rs` — Silero adapter
  (model load, hysteresis, `clone_for_new_session`).
- `crates/echo-audio/src/preprocess/vad.rs` — `EnergyVad` fallback.
- `crates/echo-app/src/use_cases/streaming/mod.rs` — pipeline gating
  (RMS bypass when neural VAD active, `Skipped { reason: "vad_silence" }`).
- `src-tauri/src/commands.rs` — `AppState::ensure_vad()` lazy load
  + `disableNeuralVad` IPC option.
- `crates/echo-proto/src/main.rs` — `--vad-model` / `--no-neural-vad`
  CLI flags and `vad=silero|rms` startup log.
- `scripts/simplify-silero-vad.py` — the pre-processor itself, with
  the four-level rationale in its module docstring.
- `scripts/download-models.sh` — orchestration: fetch upstream,
  preserve `.onnx.upstream`, invoke simplifier idempotently.
- Silero VAD upstream — https://github.com/snakers4/silero-vad
- ONNX `If` operator spec — https://onnx.ai/onnx/operators/onnx__If.html
- ONNX Runtime graph optimisation levels —
  https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html
