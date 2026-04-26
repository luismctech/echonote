# EchoNote

[![CI](https://github.com/luismctech/echonote/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/luismctech/echonote/actions/workflows/ci.yml)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](./LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey)]()

**EchoNote** is a free, open-source desktop app that transcribes and summarizes your meetings using AI — entirely on your device. No cloud services, no bots joining your calls, no subscriptions.

> **Status:** Early alpha. Core features work on macOS; Windows and Linux builds are available but less tested.

---

## Why EchoNote?

Most meeting transcription tools send your audio to the cloud. EchoNote doesn't.

- **Your data stays on your machine.** Audio, transcripts, summaries, and chat history never leave your device.
- **No bot joins your call.** EchoNote captures audio directly from your microphone — no awkward "Otter.ai is joining" moments.
- **No subscription.** It's free and open-source. Download it, run it, own it.
- **Works offline.** No internet connection required after the initial model download.

---

## Features

| Feature | Description |
|---------|-------------|
| **Live transcription** | Real-time speech-to-text as you speak, powered by [Whisper](https://github.com/ggerganov/whisper.cpp) |
| **Speaker identification** | Automatically detects and labels different speakers in the conversation |
| **AI summaries** | Generate meeting summaries with one click using a local LLM (no cloud API) |
| **Meeting search** | Full-text search across all your past meetings |
| **Multiple languages** | Supports 90+ languages via Whisper; optimized for English and Spanish |
| **Cross-platform** | Available for macOS (Apple Silicon & Intel), Windows, and Linux |
| **Auto-updates** | The app checks for new versions on launch |

---

## Download

Get the latest release for your platform from [**GitHub Releases**](https://github.com/luismctech/echonote/releases/latest).

| Platform | File | How to install |
|----------|------|----------------|
| macOS (Apple Silicon) | `EchoNote_x.x.x_aarch64.dmg` | Open `.dmg`, drag to Applications |
| macOS (Intel) | `EchoNote_x.x.x_x64.dmg` | Open `.dmg`, drag to Applications |
| Windows | `EchoNote_x.x.x_x64-setup.exe` | Run the installer |
| Linux (Debian/Ubuntu) | `EchoNote_x.x.x_amd64.deb` | `sudo dpkg -i EchoNote_*.deb` |
| Linux (other) | `EchoNote_x.x.x_amd64.AppImage` | Make executable and run |

### First launch notes

<details>
<summary><strong>macOS — "app is damaged" warning</strong></summary>

The app is not yet code-signed with Apple. Run this once in Terminal:

```sh
xattr -cr /Applications/EchoNote.app
```

Then open the app normally. This is safe — the full source code is available in this repository.
</details>

<details>
<summary><strong>Windows — SmartScreen warning</strong></summary>

Windows SmartScreen may warn about an unrecognized app. Click **"More info"** → **"Run anyway"**. This is normal for new open-source apps without a code signing certificate.
</details>

<details>
<summary><strong>Linux — AppImage permissions</strong></summary>

```sh
chmod +x EchoNote_*.AppImage
./EchoNote_*.AppImage
```
</details>

---

## How it works

1. **Start a meeting** — Click record; EchoNote captures audio from your microphone.
2. **See the transcript live** — Words appear in real-time as speakers talk.
3. **Review later** — All meetings are saved locally. Browse, search, and re-read any past meeting.
4. **Get a summary** — Click "Summarize" to generate an AI-powered summary on your device.

All processing happens locally using these open-source AI models:

| Component | What it does | Size |
|-----------|-------------|------|
| [Whisper](https://github.com/ggerganov/whisper.cpp) | Speech-to-text | ~1.6 GB |
| [Qwen 3](https://huggingface.co/Qwen) | Meeting summaries & chat | ~5–9 GB |
| [Silero VAD](https://github.com/snakers4/silero-vad) | Detects when someone is speaking | ~1.2 MB |
| [ERes2Net](https://github.com/modelscope/3D-Speaker) | Identifies different speakers | ~15 MB |

Models are downloaded automatically on first use.

---

## System requirements

- **macOS** 12.3+ (Monterey or later), Apple Silicon recommended
- **Windows** 10/11 (64-bit)
- **Linux** — Debian/Ubuntu 22.04+ or any distro with AppImage support
- **RAM:** 8 GB minimum, 16 GB recommended
- **Disk:** ~3 GB for the app + base models; ~12 GB with the full LLM

---

## Privacy & security

- **Zero network access** during meetings — all transcription and AI runs locally.
- **No telemetry.** EchoNote does not collect usage data, analytics, or crash reports.
- **Local storage only.** Meetings are stored in a SQLite database on your machine.
- **Open source.** You can audit every line of code in this repository.

---

## Roadmap

- [x] Live transcription with Whisper
- [x] Speaker identification (diarization)
- [x] Meeting persistence and search
- [x] Local LLM summaries
- [ ] System audio capture (transcribe the other side of the call)
- [ ] Conversational chat ("What did Maria say about the deadline?")
- [ ] Encrypted local storage
- [ ] Setup wizard with hardware profile detection
- [ ] More summary templates (1:1, sprint review, interview, sales call)

See [docs/DEVELOPMENT_PLAN.md](./docs/DEVELOPMENT_PLAN.md) for the full roadmap.

---

## Built with

| Layer | Technology |
|-------|------------|
| Desktop app | [Tauri 2](https://tauri.app/) (Rust + native webview) |
| Backend | Rust 1.88+ |
| Frontend | React 18, TypeScript, Tailwind CSS |
| Speech-to-text | [whisper.cpp](https://github.com/ggerganov/whisper.cpp) |
| Summaries | [llama.cpp](https://github.com/ggerganov/llama.cpp) |
| Voice activity | [Silero VAD](https://github.com/snakers4/silero-vad) via ONNX |
| Speaker ID | [3D-Speaker ERes2Net](https://github.com/modelscope/3D-Speaker) via ONNX |
| Storage | SQLite with FTS5 full-text search |

---

## Contributing

EchoNote is in active early development. Contributions are welcome — see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

Commits follow [Conventional Commits](https://www.conventionalcommits.org/).

---

<details>
<summary><h2>Developer guide</h2></summary>

### Prerequisites

- macOS 12.3+, Windows 10+, or Linux (Ubuntu 22.04+)
- Rust 1.88+ (pinned in `rust-toolchain.toml`)
- Node 20+ and pnpm 10+
- CMake, Clang (required by whisper.cpp / llama.cpp)

On macOS:
```sh
brew install cmake ninja
xcode-select --install
```

### Setup

```sh
git clone https://github.com/luismctech/echonote.git
cd echonote
./scripts/bootstrap.sh      # verifies toolchain, sets up git hooks
pnpm install                # frontend dependencies
cargo build --workspace     # backend dependencies
```

### Development

```sh
pnpm tauri:dev              # full app with hot-reload
pnpm dev                    # frontend only (browser, no Tauri IPC)

# Backend checks (same as CI)
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
```

### CLI tool

`echo-proto` is a headless CLI for recording, transcription, streaming, and benchmarks:

```sh
cargo run -p echo-proto -- --help
cargo run -p echo-proto -- record --duration 5 --output /tmp/sample.wav
cargo run -p echo-proto -- transcribe /tmp/sample.wav
cargo run -p echo-proto -- stream --duration 30
cargo run -p echo-proto -- meetings list
```

### Download models

```sh
./scripts/download-models.sh          # Whisper large-v3-turbo (default)
./scripts/download-models.sh --all    # all models (ASR + VAD + embeddings)
```

### Repository structure

```
echonote/
├── src/                  React frontend (TypeScript)
├── src-tauri/            Tauri shell (Rust)
├── crates/               Rust library crates
│   ├── echo-app/         Application services
│   ├── echo-asr/         Whisper speech-to-text
│   ├── echo-audio/       Audio capture & processing
│   ├── echo-diarize/     Speaker identification
│   ├── echo-domain/      Core types & ports
│   ├── echo-llm/         LLM integration
│   ├── echo-proto/       CLI tool
│   ├── echo-storage/     SQLite persistence
│   └── echo-telemetry/   Logging & tracing
├── docs/                 Architecture, design, ADRs
├── models/               AI models (git-ignored)
├── scripts/              Build & setup scripts
└── fixtures/             Test audio & transcripts
```

### Project documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](./docs/ARCHITECTURE.md) | System design and technical decisions |
| [DESIGN.md](./docs/DESIGN.md) | UI/UX design system |
| [DEVELOPMENT_PLAN.md](./docs/DEVELOPMENT_PLAN.md) | Phased roadmap (28 weeks) |
| [docs/adr/](./docs/adr/) | Architecture Decision Records |
| [docs/benchmarks/](./docs/benchmarks/) | ASR quality benchmarks |

</details>

---

## License

[AGPL-3.0](./LICENSE) © 2026 Alberto Cruz
