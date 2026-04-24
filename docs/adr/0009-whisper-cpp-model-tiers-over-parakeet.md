# ADR-0009: Whisper.cpp model tiers over NVIDIA Parakeet

- **Status:** accepted
- **Date:** 2026-07-19
- **Deciders:** Tech Lead
- **Technical story:** Phase 3B — evaluate NVIDIA Parakeet TDT 0.6B as an
  alternative ASR backend and decide on a model expansion strategy.

## Context and problem statement

EchoNote uses whisper.cpp (via `whisper-rs`) as its sole ASR backend
(see ADR-0003). The competitive landscape has shifted: NVIDIA's Parakeet
TDT 0.6B V2 (English) and V3 (25 European languages) top the HuggingFace
Open ASR Leaderboard with a mean WER of 6.05 % and an RTFx of 3,386.

We evaluated whether to add Parakeet as an alternative ASR backend and, if
not, how to maximise whisper.cpp quality across diverse hardware.

## Decision drivers

- **Local-first, cross-platform.** EchoNote runs on macOS (Metal), Windows
  (DirectX/Vulkan), and Linux without requiring a GPU vendor lock-in or
  Python runtime.
- **Spanish-first audience.** Our primary users operate in Spanish; the ASR
  backend must be multilingual from the default path.
- **Hardware diversity.** Users range from 8 GB laptops (Lite profile) to
  32 GB+ workstations (Quality profile). A single model size cannot serve
  both extremes.
- **Distribution simplicity.** No Python, no Docker, no CUDA toolkit in
  the critical install path.

## Considered options

### Option A: NVIDIA Parakeet TDT 0.6B via ONNX Runtime

Export the NeMo `.nemo` checkpoint to ONNX, run inference via the `ort`
crate (Rust ONNX Runtime bindings).

- **Pros:** State-of-the-art WER (6.05 % mean), word-level timestamps,
  automatic punctuation and capitalisation.
- **Cons:**
  - FastConformer-TDT architecture requires a custom TDT decoder
    (token + duration prediction) not available in any Rust crate today.
  - The NeMo → ONNX export pipeline requires Python + PyTorch + NeMo
    toolkit; no pre-exported ONNX is officially published.
  - 600 M parameters → ~1.2 GB FP16 ONNX; comparable to whisper-large-v3
    but with additional SentencePiece tokenizer dependency.
  - Parakeet V2 is English-only. V3 covers 25 European languages but
    the ONNX export path is even less mature.
  - ONNX Runtime GPU acceleration is CUDA-only for the TDT op set;
    Metal and Vulkan support is unverified.
  - Breaks the current architecture invariant that all ASR flows through
    `whisper-rs` and the `Transcriber` trait's existing state pooling.

### Option B: Whisper.cpp model tier expansion (chosen)

Keep whisper.cpp as the sole ASR backend but expand the downloadable model
catalog to cover the full hardware spectrum, including quantised models and
Distil-Whisper for English-first users who prioritise speed.

- **Pros:**
  - Zero new dependencies; same `whisper-rs` build with Metal, CUDA, and
    Vulkan feature flags already wired (ADR-0003 + Phase 3A).
  - Quantised Turbo Q5_0 (574 MB) delivers ~95 % of FP16 quality at 60 %
    of the size — ideal for the Lite hardware profile.
  - Distil-Whisper Large V3 (756 MB) is 5× faster than large-v3 with only
    ~0.8 % WER degradation on English long-form.
  - Full multilingual support on every tier from Base (142 MB) through
    Large V3 Turbo (1.6 GB).
  - Download infrastructure (`model_catalog`, `download-models.sh`) already
    handles multiple models; adding entries is incremental.
- **Cons:**
  - Whisper large-v3-turbo's mean WER (~8 %) is higher than Parakeet's
    6.05 % on the Open ASR Leaderboard.
  - No automatic punctuation or capitalisation from whisper.cpp (post-
    processing is handled by the LLM summariser downstream).

### Option C: Python sidecar for Parakeet

Shell out to a bundled Python process running NeMo.

- **Rejected.** Violates the "no Python" distribution constraint (ADR-0002)
  and adds ~2 GB of Python + PyTorch + CUDA to the install.

## Decision outcome

**Chosen option: B — Whisper.cpp model tier expansion.**

Parakeet's quality edge does not justify the integration complexity, the
CUDA vendor lock-in, or the deviation from our cross-platform, no-Python
architecture. Instead, we expand the whisper.cpp model catalog to six
tiers that map cleanly to our hardware profiles:

| Model ID | File | Size | Lang | Profile |
|---|---|---|---|---|
| `asr-large-v3-turbo` | `ggml-large-v3-turbo.bin` | 1.6 GB | Multi | Quality / Balanced |
| `asr-large-v3-turbo-q5` | `ggml-large-v3-turbo-q5_0.bin` | 574 MB | Multi | Balanced / Lite |
| `asr-distil-large-v3` | `ggml-distil-large-v3.bin` | 756 MB | EN | English-fast |
| `asr-medium` | `ggml-medium.bin` | 1.5 GB | Multi | Balanced (fallback) |
| `asr-small` | `ggml-small.bin` | 488 MB | Multi | Lite |
| `asr-base` | `ggml-base.bin` | 148 MB | Multi | Ultra-lite / dev |

The auto-resolution order in `preferred_asr_model()` prefers the Spanish
fine-tune first, then full-precision Turbo, then quantised Turbo, then
progressively smaller multilingual models, with English-only (distil) and
legacy English models last.

### Re-evaluation triggers

- An official ONNX export of Parakeet V3 with multilingual support and a
  stable ONNX opset that works on `ort` + CoreML.
- A Rust-native FastConformer/TDT implementation (e.g. in `candle` or
  `burn`).
- Community GGML conversion of Parakeet (if the architecture becomes
  supported by `ggml`/`llama.cpp`).

## Links

- [Parakeet TDT 0.6B V2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) — HuggingFace model card
- [Distil-Whisper Large V3 GGML](https://huggingface.co/distil-whisper/distil-large-v3-ggml) — GGML weights
- [whisper.cpp models](https://huggingface.co/ggerganov/whisper.cpp) — full model catalog
- ADR-0003 — whisper.cpp selection rationale
