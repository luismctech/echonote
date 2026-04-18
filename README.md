# EchoNote

> Private, local-first meeting transcription and AI summaries.
> Runs 100% on your device. No cloud, no bots, no subscriptions.

EchoNote is a cross-platform desktop application (Windows, macOS, Linux) that
captures, transcribes and summarizes meetings using open-source AI models that
run entirely on the user's machine. It is the privacy-first alternative to
cloud-based tools like Granola.

**Status:** 🚧 Pre-alpha — Sprint 0 (scaffolding & CLI prototype). Not yet usable.

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
| Backend | Rust 1.75+ (Clean Architecture with ports & adapters) |
| Frontend | React 18 + TypeScript + Tailwind + shadcn/ui + Zustand |
| ASR | [whisper.cpp](https://github.com/ggerganov/whisper.cpp) via `whisper-rs` |
| LLM | [llama.cpp](https://github.com/ggerganov/llama.cpp) via `llama-cpp-rs` |
| VAD / Diarization | ONNX Runtime via `ort` (Silero VAD, 3D-Speaker ERes2Net) |
| Storage | SQLite + FTS5 + optional SQLCipher |
| Capture | `cpal` + platform-specific (WASAPI / ScreenCaptureKit / PulseAudio) |

Full rationale is documented in [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md).

---

## Default models (Balanced profile)

| Component | Model | Size on disk |
|---|---|---|
| ASR | Whisper medium q5_0 | ~1.5 GB |
| LLM | Qwen 2.5 7B Instruct Q4_K_M | ~4.4 GB |
| VAD | Silero VAD v5 | ~2 MB |
| Diarization | 3D-Speaker ERes2Net | ~15 MB |

Lite and Quality profiles, plus benchmarks of alternative models, are tracked
in `docs/benchmarks/` (populated during Phase 0).

---

## Repository layout

```
echonote/
├── Cargo.toml                Rust workspace root
├── crates/                   Rust library crates (domain, app, audio, asr, …)
├── src-tauri/                Tauri binary (commands + event wiring)
├── src/                      React frontend
├── tests/                    Fixtures, integration, e2e
├── docs/                     ADRs, benchmarks, user docs
├── scripts/                  Utility scripts (download-models, etc.)
├── docs/
│   ├── ARCHITECTURE.md       Technical architecture
│   ├── DESIGN.md             UI/UX system
│   ├── DEVELOPMENT_PLAN.md   Phased roadmap (28 weeks)
│   ├── adr/                  Architecture Decision Records (MADR)
│   ├── benchmarks/           ASR/LLM benchmark results per phase
│   └── mockup-*.html         Interactive mockups
└── README.md                 You are here
```

---

## Development

**Primary development platform for Phase 0:** macOS (Apple Silicon).
Windows and Linux are added in Phase 1 (weeks 12–15).

### Prerequisites

- macOS 12.3+ (Monterey) on Apple Silicon or Intel
- Rust 1.75+ (`rustup install stable`)
- Node 20+ and pnpm 9+
- CMake, Ninja, Clang (for whisper.cpp / llama.cpp builds)
  ```sh
  brew install cmake ninja
  xcode-select --install
  ```

### First-time setup

```sh
git clone https://github.com/AlbertoMZCruz/echonote.git
cd echonote
git checkout develop
./scripts/download-models.sh    # will exist at the end of Sprint 0
cargo build
```

### Quickstart (will be valid at the end of Sprint 0)

```sh
# 30-second end-to-end prototype: record → transcribe → summarize
cargo run --release -p echo-proto -- record --duration 30
```

---

## Project documentation

| Document | Purpose |
|---|---|
| [docs/DEVELOPMENT_PLAN.md](./docs/DEVELOPMENT_PLAN.md) | Phased roadmap, scope, milestones |
| [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) | System design, layers, stack justifications |
| [docs/DESIGN.md](./docs/DESIGN.md) | Visual design system, UX flows, screens |
| `docs/adr/` | Architecture Decision Records (MADR format) |
| `docs/benchmarks/` | ASR / LLM benchmark results per phase |
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
