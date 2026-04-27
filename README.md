<p align="center">
  <img src="src-tauri/icons/128x128.png" width="80" alt="EchoNote logo" />
</p>
<h1 align="center">EchoNote</h1>

<p align="center">
  <a href="https://github.com/luismctech/echonote/actions/workflows/ci.yml"><img src="https://github.com/luismctech/echonote/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI" /></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-AGPL--3.0-blue.svg" alt="License: AGPL-3.0" /></a>
  <img src="https://img.shields.io/badge/platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey" alt="Platform" />
</p>

**EchoNote** is a free, open-source desktop app that transcribes and summarizes your meetings using AI — entirely on your device. No cloud services, no bots joining your calls, no subscriptions.

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
| **Custom summary templates** | Create your own prompt templates for tailored summaries (1:1, sprint review, sales call, or anything you need) |
| **Model selection** | Download multiple ASR or LLM models and switch between them at runtime — no restart required |
| **Meeting search** | Full-text search across all your past meetings |
| **Conversational chat** | Ask follow-up questions about any meeting using the local LLM |
| **Multiple languages** | Supports 90+ languages via Whisper; optimized for English and Spanish |
| **Cross-platform** | Available for macOS (Apple Silicon & Intel), Windows, and Linux |
| **Sleep prevention** | Automatically prevents OS sleep while recording so you never lose a session |
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

#### macOS — "app is damaged" warning

The app is not yet code-signed with Apple. Run this once in Terminal:

```sh
xattr -cr /Applications/EchoNote.app
```

Then open the app normally. This is safe — the full source code is available in this repository.

#### Windows — SmartScreen warning

Windows SmartScreen may warn about an unrecognized app. Click **"More info"** → **"Run anyway"**. This is normal for new open-source apps without a code signing certificate.

#### Linux — AppImage permissions

```sh
chmod +x EchoNote_*.AppImage
./EchoNote_*.AppImage
```

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

Models are **not** downloaded automatically — you choose which ones to install from the built-in model manager in **Settings → Models**.

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
- [x] Conversational chat ("What did Maria say about the deadline?")
- [x] Custom summary templates (create your own prompts)
- [x] Runtime model selection (switch ASR/LLM models without restarting)
- [ ] System audio capture (transcribe the other side of the call)
- [ ] Encrypted local storage
- [ ] Setup wizard with hardware profile detection

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

EchoNote is open-source and contributions are welcome — see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

---

## License

[AGPL-3.0](./LICENSE) © 2026 Luis MC
