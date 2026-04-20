# EchoNote

[![CI](https://github.com/AlbertoMZCruz/echonote/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/AlbertoMZCruz/echonote/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.88%2B-orange.svg?logo=rust)](./rust-toolchain.toml)

> Private, local-first meeting transcription and AI summaries.
> Runs 100% on your device. No cloud, no bots, no subscriptions.

EchoNote is a cross-platform desktop application (Windows, macOS, Linux) that
captures, transcribes and summarizes meetings using open-source AI models that
run entirely on the user's machine. It is the privacy-first alternative to
cloud-based tools like Granola.

**Status:** 🚧 Pre-alpha — Sprint 1 in progress. Sprint 0 (`v0.1.0-sprint0`)
shipped end-to-end live streaming, SQLite persistence and a Phase-0 WER bench
on macOS. Sprint 1 has so far added per-track diarization, FTS5 search, a
refactored React frontend that follows clean-architecture layering, an ordered
shutdown path, on-demand local LLM summaries (Qwen 3 14B by default), and
moved the default ASR to multilingual `large-v3-turbo` for Spanish-first
recordings. Conversational chat with citations lands next.

**What works today (`develop`):**

- 🎙️ Live microphone streaming with 5-second windows, neural Silero VAD (with energy-VAD fallback), and Whisper
  on Metal (RTF ≈ 0.08 on Apple M1 Pro).
- 🗣️ Online speaker diarization via 3D-Speaker ERes2Net embeddings; speakers
  are persisted per-meeting and renameable from the UI.
- 💾 Per-chunk persistence into SQLite — recordings survive app restarts;
  WAL is checkpointed on app close via an ordered shutdown hook.
- 🔍 Full-text search across every meeting (SQLite FTS5, diacritic-insensitive,
  BM25-ranked, with snippet highlights).
- 🖥️ Tauri desktop UI with meetings sidebar, live transcript pane, replay view
  for past meetings, and a hand-rolled toast layer.
- ⌨️ `echo-proto` CLI for headless capture, transcription, streaming, meetings
  inspection and WER benchmarks.
- 📊 Phase-0 WER baseline at **8.40 %** (`base.en`, synthetic fixtures, see
  [`docs/benchmarks/PHASE-0.md`](./docs/benchmarks/PHASE-0.md)).

**Not yet:** local LLM summaries (day 9), system-audio capture, encrypted-at-rest
storage, the setup wizard, and the Lite/Quality hardware profiles.

---

## Highlights

- **100% local processing.** Audio, transcripts, summaries and chat never leave the device.
- **Dual audio capture.** Microphone and system audio captured as separate tracks (no bot joins the call).
- **Hybrid ASR.** Live streaming transcription + high-quality refinement after the meeting.
- **Per-track diarization.** Speakers are clustered per audio track using local ONNX embeddings.
- **6 summary templates.** General, 1:1, sprint review, interview, sales call, class.
- **Chat with your meeting.** Ask questions; the LLM answers with citations back to segments.
- **Full-text search** across every meeting via SQLite FTS5.
- **Three hardware profiles** (Lite / Balanced / Quality) with a setup wizard that picks the right one.

---

## Tech stack

| Layer | Technology |
|---|---|
| Desktop shell | [Tauri 2.x](https://tauri.app/) |
| Backend | Rust 1.88+ (Clean Architecture with ports & adapters) |
| Frontend | React 18 + TypeScript + Tailwind + shadcn/ui + Zustand |
| ASR | [whisper.cpp](https://github.com/ggerganov/whisper.cpp) via `whisper-rs` |
| LLM | [llama.cpp](https://github.com/ggerganov/llama.cpp) via `llama-cpp-rs` |
| VAD / Diarization | ONNX Runtime via `ort` (Silero VAD, 3D-Speaker ERes2Net) |
| Storage | SQLite + FTS5 + optional SQLCipher |
| Capture | `cpal` + platform-specific (WASAPI / ScreenCaptureKit / PulseAudio) |

Full rationale is documented in [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md).

---

## Default models (Balanced profile)

EchoNote targets **Spanish-first** meetings, so the defaults below favour
multilingual ASR and Spanish-capable LLMs out of the box. English-only Whisper
variants and Qwen 2.5 fallbacks remain reachable for benchmarks and back-compat.

| Component | Model | Size on disk |
|---|---|---|
| ASR | Whisper large-v3-turbo (multilingual) | ~1.6 GB |
| ASR (optional, Spanish fine-tune) | whisper-large-v3-turbo-es | ~1.6 GB |
| LLM | Qwen 3 14B Instruct Q4_K_M | ~9 GB |
| LLM (lighter) | Qwen 3 8B Instruct Q4_K_M | ~5 GB |
| LLM (Quality, MoE) | Qwen 3 30B-A3B Instruct Q4_K_M | ~18 GB |
| VAD | Silero VAD v5.1.2 (pre-procesado) | ~1.2 MB on-disk (~2.2 MB upstream) |
| Diarization | 3D-Speaker ERes2Net | ~15 MB |

Lite and Quality profiles, plus benchmarks of alternative models, are tracked
in `docs/benchmarks/` (populated during Phase 0).

---

## Repository layout

```
echonote/
├── Cargo.toml                Rust workspace root
├── package.json              Frontend root (pnpm)
├── vite.config.ts            Vite configuration
├── tailwind.config.ts        Tailwind configuration
├── index.html                Vite entry HTML
├── src/                      React frontend (TypeScript)
│   ├── main.tsx
│   ├── App.tsx
│   ├── ipc/client.ts         Typed Tauri IPC client
│   ├── types/                Pure TS types mirroring Rust DTOs
│   ├── state/                Reducers + context providers
│   ├── hooks/                Reusable behaviour (recording, meeting detail…)
│   ├── lib/                  Dependency-free helpers
│   ├── components/           Reusable UI primitives
│   └── features/             View-level components grouped by feature
├── src-tauri/                Tauri host shell (echo-shell crate)
│   ├── Cargo.toml
│   ├── tauri.conf.json
│   ├── capabilities/         Capability manifests (security)
│   └── src/                  commands.rs + lib.rs + main.rs
├── crates/                   Rust library crates (domain, app, audio, asr, …)
├── tests/                    Fixtures, integration, e2e
├── docs/
│   ├── ARCHITECTURE.md       Technical architecture
│   ├── DESIGN.md             UI/UX system
│   ├── DEVELOPMENT_PLAN.md   Phased roadmap (28 weeks)
│   ├── adr/                  Architecture Decision Records (MADR)
│   ├── benchmarks/           ASR/LLM benchmark results per phase
│   └── mockup-*.html         Interactive mockups
├── scripts/                  Utility scripts (bootstrap, download-models)
└── README.md                 You are here
```

---

## Development

**Primary development platform for Phase 0:** macOS (Apple Silicon).
Windows and Linux are added in Phase 1 (weeks 12–15).

### Prerequisites

- macOS 12.3+ (Monterey) on Apple Silicon or Intel
- Rust 1.88+ (pinned in `rust-toolchain.toml`)
- Node 20+ and pnpm 10+
- CMake, Clang (required by whisper.cpp / llama.cpp build scripts)
  ```sh
  brew install cmake ninja
  xcode-select --install
  ```

### First-time setup

```sh
git clone https://github.com/AlbertoMZCruz/echonote.git
cd echonote
git checkout develop
./scripts/bootstrap.sh          # verifies toolchain and wires up git hooks
pnpm install                    # frontend deps
cargo build --workspace         # backend deps
```

### Day-to-day

```sh
# Launch the desktop shell in dev mode (webview + hot-reload + Rust).
pnpm tauri:dev

# Frontend only (browser, no IPC).
pnpm dev

# Run all backend checks (same as CI).
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
```

### CLI prototype (Phase 0)

`echo-proto` is the headless prototype that grows through Sprint 0 into a
full record → transcribe → summarize pipeline. Subcommands land
incrementally; `--help` always lists what is wired today.

```sh
cargo run -p echo-proto -- --help
```

#### Recording (Sprint 0 day 5)

List input devices:

```sh
cargo run -p echo-proto -- record-devices
```

Capture 5 seconds from the default microphone to a WAV file:

```sh
cargo run -p echo-proto -- record --duration 5 --output /tmp/sample.wav
```

Pick a specific device by name (use `record-devices` to discover names):

```sh
cargo run -p echo-proto -- record --device "BlackHole 2ch" --duration 3 --output /tmp/sys.wav
```

The capture format follows what CoreAudio negotiates with the device
(typically 44.1 kHz mono `f32`, transcoded to 16-bit PCM in the WAV).
Files are resampled to Whisper-native 16 kHz mono on the fly inside the
`transcribe` subcommand below.

#### Transcription (Sprint 0 day 6)

`transcribe` runs whisper.cpp locally through the `echo-asr` adapter.
On macOS the build uses the Metal backend; on Linux it falls back to a
CPU build (acceleration features land in Phase 1).

Fetch a Whisper model (default: `large-v3-turbo`, multilingual, ~1.6 GiB):

```sh
./scripts/download-models.sh                 # large-v3-turbo (recommended)
./scripts/download-models.sh medium          # ~1.5 GiB, multilingual
./scripts/download-models.sh small           # ~466 MiB, multilingual
./scripts/download-models.sh base.en         # ~142 MiB, English-only (Sprint 0 default)
./scripts/download-models.sh asr-es          # Spanish fine-tune (needs Python; see below)
./scripts/download-models.sh --all           # large-v3-turbo + vad + embed
```

For Spanish-first deployments you can also build the
`whisper-large-v3-turbo-es` fine-tune (5.34 % WER on Common Voice 17 ES vs
6.91 % for upstream turbo). It needs Python ≥ 3.10 the first time:

```sh
./scripts/build-spanish-asr.sh
# Resulting model: ./models/asr/ggml-large-v3-turbo-es.bin
```

Transcribe a WAV (any sample rate, any channel count — it gets
downmixed to mono and resampled to 16 kHz before inference):

```sh
cargo run -p echo-proto -- transcribe /tmp/sample.wav

# pin language, ask for JSON output:
cargo run -p echo-proto -- transcribe /tmp/sample.wav --language en --json

# point at a non-default model:
cargo run -p echo-proto -- transcribe /tmp/sample.wav \
    --model models/asr/ggml-medium.bin --language es

# translate a non-English source to English instead of transcribing:
cargo run -p echo-proto -- transcribe /tmp/sample.wav --translate
```

The plain-text output ends with a footer reporting the detected
language, segment count, audio duration and the **real-time factor**
(`elapsed / audio`). On an Apple M1 Pro with `ggml-large-v3-turbo`
and Metal, expect RTF ≈ 0.08 (≈ 12× realtime). The English-only
`ggml-base.en` is faster (RTF ≈ 0.03) but cannot transcribe Spanish.

#### Streaming pipeline (Sprint 0 day 7)

Live mic → resample → whisper streaming. Same pipeline the desktop UI
uses, headless. Useful as a smoke test or for batch transcribing your
microphone in a terminal session.

```sh
# 30 s capture, 5 s chunks, default mic, model from ECHO_ASR_MODEL:
cargo run -p echo-proto -- stream --duration 30

# Custom chunk window + silence gate (RMS threshold; 0 disables):
cargo run -p echo-proto -- stream --duration 60 --chunk-ms 4000 --silence-threshold 0.01

# Disable the neural VAD and fall back to the energy gate only
# (useful for very soft speakers Silero misclassifies as silence):
cargo run -p echo-proto -- stream --duration 60 --no-neural-vad
```

Each chunk prints with its index, offset, RTF and the decoded text.
Silent chunks are reported as `silence (rms=…)` and skipped. Every
session is **persisted to SQLite** through the same `MeetingRecorder`
the UI uses — inspect afterwards with `meetings show`.

##### Voice Activity Detection (VAD)

EchoNote runs a **two-tier VAD** to keep Whisper from hallucinating
on silent or noisy chunks (the canonical "Gracias por ver el video"
or `[no speech]` outros):

1. **Silero VAD (neural, default)** — when
   `models/vad/silero_vad.onnx` is installed (run
   `./scripts/download-models.sh vad`, ~1.2 MB on disk after
   pre-processing), every resampled chunk is scored by Silero's
   recurrent model. Only chunks classified as `Voiced` reach Whisper;
   the rest are emitted as `Skipped` events and never see the ASR.
   Silero's LSTM keeps temporal context across chunks, so it's much
   sharper than RMS at distinguishing speech from fans, music, keyboard
   noise, room tone, etc.

   > **Note on the ONNX model**: the upstream Silero v5.1.2 ONNX uses
   > the `If` operator and is dispatched at runtime by sample rate
   > (8/16 kHz). Our ONNX backend (`tract-onnx`, chosen in ADR-0007 for
   > a single-binary deploy with no native runtime) does not implement
   > `If` nor ONNX-Runtime's contrib ops (`FusedConv`, …). The download
   > script therefore runs `scripts/simplify-silero-vad.py` once after
   > the download: it inlines the 16 kHz branch, drops the `sr` input,
   > pins shapes, and constant-folds at `ORT_ENABLE_BASIC` to produce a
   > 36-node, contrib-op-free graph that is bitwise-equivalent to the
   > upstream output (Δ = 0). The raw upstream is preserved beside it as
   > `silero_vad.onnx.upstream` for auditing.
2. **Energy gate (fallback)** — when the Silero ONNX is missing, or
   when the user passes `disableNeuralVad: true` (Tauri) /
   `--no-neural-vad` (CLI), the pipeline falls back to a per-chunk
   RMS gate at `0.02` (`StreamingOptions::silence_rms_threshold`,
   tunable per session).

When the Silero VAD is active, the RMS gate is intentionally
**bypassed** — Silero is strictly more discriminating, and its
temporal model needs every chunk in chronological order to stay
coherent. The model is loaded once per process and cloned cheaply per
session via `SileroVad::clone_for_new_session` (Arc share of the
optimized graph + zeroed LSTM state).

Override the model path with `ECHO_VAD_MODEL=/abs/path/silero_vad.onnx`
or `--vad-model /abs/path/silero_vad.onnx` (CLI).

#### Meetings (Sprint 0 day 8)

Inspect the local SQLite database (default path: `./echonote.db`,
override with `ECHO_DB_PATH=…`).

```sh
cargo run -p echo-proto -- meetings list
cargo run -p echo-proto -- meetings list --json
cargo run -p echo-proto -- meetings show <uuid>
cargo run -p echo-proto -- meetings show <uuid> --json
cargo run -p echo-proto -- meetings delete <uuid>
```

#### Benchmarks (Sprint 0 day 9)

Phase-0 ASR benchmark over synthetic fixtures (see
[`fixtures/README.md`](./fixtures/README.md) for the contract). Reports
per-clip + global WER, RTF p50/p95, and fails when global WER exceeds
the gate.

```sh
# Generate the synthetic WAVs locally (macOS `say` + `afconvert`):
./scripts/build-fixtures.sh

# Run the bench. Writes a JSON report and exits non-zero on regression.
cargo run --release -p echo-proto -- bench wer \
    --max-wer 0.25 \
    --report target/bench-reports/wer.json
```

The full baseline + analysis lives in
[`docs/benchmarks/PHASE-0.md`](./docs/benchmarks/PHASE-0.md). To run
the same bench in CI on a clean macOS runner with a downloaded model:

```sh
gh workflow run bench.yml -f whisper_model=base.en -f max_wer=0.25
```

---

## Project documentation

| Document | Purpose |
|---|---|
| [docs/DEVELOPMENT_PLAN.md](./docs/DEVELOPMENT_PLAN.md) | Phased roadmap, scope, milestones |
| [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) | System design, layers, stack justifications |
| [docs/DESIGN.md](./docs/DESIGN.md) | Visual design system, UX flows, screens |
| [docs/SPRINT-0-RETRO.md](./docs/SPRINT-0-RETRO.md) | Sprint 0 outcome, decisions, risks |
| [docs/benchmarks/PHASE-0.md](./docs/benchmarks/PHASE-0.md) | First WER baseline + quality gates |
| `docs/adr/` | Architecture Decision Records (MADR format) |
| `docs/mockup-*.html` | Interactive mockups of key screens |

---

## Contributing

Contribution guidelines will be published in `CONTRIBUTING.md` during Phase 1.
For now, the project is in active bootstrap; external contributions are not yet
being reviewed.

Commits follow [Conventional Commits](https://www.conventionalcommits.org/).
Branching model is described in `DEVELOPMENT_PLAN.md §11.1`.

---

## License

[MIT](./LICENSE) © 2026 Alberto Cruz
