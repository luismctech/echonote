#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# EchoNote — download Whisper / VAD / LLM models from their canonical sources.
#
# EchoNote targets Spanish-first meetings, so the defaults below favour
# multilingual ASR and a Spanish-capable LLM. English-only Whisper variants
# remain available for benchmarking and Sprint-0 reproducibility, but they
# are NOT installed by default any more.
#
# Usage:
#   scripts/download-models.sh                # default: ggml-large-v3-turbo (multilingual)
#   scripts/download-models.sh large-v3-turbo # ~1.5 GB, multilingual, recommended
#   scripts/download-models.sh large-v3-turbo-q5_0 # ~574 MB, multilingual, quantized turbo
#   scripts/download-models.sh distil-large-v3 # ~756 MB, English-only, 5x faster
#   scripts/download-models.sh large-v3       # ~3.0 GB, multilingual, top quality
#   scripts/download-models.sh medium         # ~1.5 GB, multilingual
#   scripts/download-models.sh small          # ~466 MB, multilingual
#   scripts/download-models.sh base           # ~142 MB, multilingual
#   scripts/download-models.sh base.en        # ~142 MB, English-only (dev / benchmarks)
#   scripts/download-models.sh small.en       # ~466 MB, English-only
#   scripts/download-models.sh asr-es         # whisper-large-v3-turbo Spanish fine-tune
#                                             # (requires Python; see Pack C below)
#   scripts/download-models.sh vad            # Silero VAD v6.2.3 (~2.7 MB raw, ~1.5 MB inlined)
#   scripts/download-models.sh embed          # 3D-Speaker ERes2Net (~26 MB)
#   scripts/download-models.sh cam-plus-plus  # CAM++ EN-VoxCeleb embedder (~28 MB)
#   scripts/download-models.sh embed-multi    # CAM++ multilingual zh_en embedder (~27 MB, better on Spanish)
#   scripts/download-models.sh segmenter      # pyannote-segmentation-3.0 (~6 MB + tract rewrite)
#   scripts/download-models.sh llm            # Qwen 3 14B Instruct Q4_K_M (~9 GB)
#   scripts/download-models.sh llm-small      # Qwen 3 8B Instruct Q4_K_M (~5 GB)
#   scripts/download-models.sh llm-lite       # Qwen 3 4B Q4_K_M (~2.5 GB, <8 GB RAM)
#   scripts/download-models.sh llm-moe        # Qwen 3 30B-A3B Instruct Q4_K_M (~18 GB)
#   scripts/download-models.sh llm-legacy-7b  # Qwen 2.5 7B (back-compat, ~4.4 GB)
#   scripts/download-models.sh llm-legacy-3b  # Qwen 2.5 3B (back-compat, ~1.9 GB)
#   scripts/download-models.sh --all          # large-v3-turbo + vad + CAM++ + ERes2Net + pyannote-segmenter (no LLM)
#
# ASR models are written to ./models/asr/ggml-<flavor>.bin
# VAD model lives at         ./models/vad/silero_vad.onnx
# Embedder lives at          ./models/embedder/eres2net_en_voxceleb.onnx
# LLM models live at         ./models/llm/<name>.gguf
# Existing files are skipped if their size looks right.
# -----------------------------------------------------------------------------
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ASR_DIR="${REPO_ROOT}/models/asr"
VAD_DIR="${REPO_ROOT}/models/vad"
EMBED_DIR="${REPO_ROOT}/models/embedder"
SEG_DIR="${REPO_ROOT}/models/segmenter"
LLM_DIR="${REPO_ROOT}/models/llm"
mkdir -p "$ASR_DIR" "$VAD_DIR" "$EMBED_DIR" "$SEG_DIR" "$LLM_DIR"
# Back-compat alias for code paths that still reference $MODELS_DIR.
MODELS_DIR="$ASR_DIR"

# Hugging Face repository that mirrors the ggerganov/whisper.cpp models.
HF_BASE="https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
# Silero VAD v6.2.3 (MIT). Pinned to the release commit so downloads are
# reproducible across machines and over time, and verified by SHA-256.
#
# We fetch `silero_vad_op18_ifless.onnx`: upstream already folded its
# nested shape `If`s, but one top-level `If` still dispatches between
# the 16 kHz and 8 kHz sub-networks on the `sr` input. pure-Rust
# `tract-onnx` — the inference backend used by
# `crates/echo-audio/src/preprocess/silero_vad.rs` — does not implement
# `If`, so after download `scripts/inline-silero-vad.py` inlines the
# 16 kHz branch and drops the `sr` input (onnx only, no ORT pass).
#
# The state signature is unchanged from v5 (`state [2,1,128]`); only the
# separate `16k_sequence` export splits h/c. v6 does need 64 samples of
# context prepended to each 512-sample hop (a `[1,576]` input), which the
# adapter supplies.
SILERO_VAD_COMMIT="5cd7945676eb32225748052e2e6a0580e4686a08"
SILERO_VAD_URL="https://github.com/snakers4/silero-vad/raw/${SILERO_VAD_COMMIT}/src/silero_vad/data/silero_vad_op18_ifless.onnx"
SILERO_VAD_SHA256="7671cd04b004e9076da0d4a7b1a5aec36adf161c39230c1cb94a4fd5db6bbd28"
SILERO_VAD_BYTES=2845718
# 3D-Speaker ERes2Net (English VoxCeleb) — the speaker embedder used by
# echo-diarize. Mirrored on Hugging Face by csukuangfj (sherpa-onnx
# maintainer); upstream lives on ModelScope. ~26 MB, opset 13, outputs
# a 192-dim embedding from 80-bin Kaldi fbank features.
ERES2NET_URL="https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main/3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx"
# CAM++ (Context-Aware Masking++) speaker embedder from the 3D-Speaker
# project. ~28 MB, same 192-dim Kaldi-fbank pipeline as ERes2Net.
# Better EER (0.73 % on VoxCeleb-O, 51 % fewer params than ECAPA-TDNN)
# and stronger multilingual generalisation — recommended for Spanish-
# primary meetings where ERes2Net (VoxCeleb-EN only) can over-split.
# Mirrored by csukuangfj on the same HF repo as ERes2Net.
CAMPP_URL="https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"
# CAM++ multilingual (3D-Speaker zh_en 16k-common_advanced). Same
# [1,T,80] Kaldi-fbank contract as the EN model but a 192-dim output
# (the EN-VoxCeleb export is the outlier at 512-dim). Trained on a far
# larger, more varied speaker set (EN+ZH), which generalises better to
# Spanish than the English-only export — recommended A/B for reducing
# over-splitting on Spanish meetings. ~27 MiB, opset 13.
CAMPP_MULTI_URL="https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
# pyannote-segmentation-3.0 ONNX export from the sherpa-onnx project.
# ~17 MB, detects speaker boundaries at 10 ms granularity. Used by the
# PyannoteSegmenter adapter in echo-diarize to split 5-second chunks
# into speaker-homogeneous sub-regions before embedding.
PYANNOTE_SEG_URL="https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/main/model.onnx"
# Two-speaker English fixture from the same HF mirror. Used by the
# echo-diarize integration tests as ground-truth cross-speaker audio.
ERES2NET_FIXTURE_URL="https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main/1-two-speakers-en.wav"
ERES2NET_FIXTURE_DIR="${REPO_ROOT}/crates/echo-diarize/tests/fixtures"

# Default LLM for summaries. Qwen 3 14B Instruct is multilingual (119+
# languages including native-level Spanish vs Qwen 2.5's 29), Apache 2.0,
# and pre-trained on 36 T tokens (vs 18 T in 2.5). Same `<|im_start|>`
# chat template as Qwen 2.5, so our `SummarizeMeeting` prompt and stop
# tokens stay unchanged. Q4_K_M is ~9 GB on disk (~10 GB RAM with KV
# cache at our 4 k context), comfortably inside our < 45 s/30 min target
# from DEVELOPMENT_PLAN.md §3.1 CU-04 on Apple Silicon ≥16 GB.
# Note on naming: the official Qwen team publishes Qwen 3 GGUFs under
# `Qwen/Qwen3-<size>-GGUF` (no `-Instruct-` infix, since every Qwen 3
# checkpoint is instruction-tuned out of the box) and uses capital `Q`
# in the quant suffix (`Qwen3-14B-Q4_K_M.gguf`). We mirror that exact
# casing to avoid 404s on the HF CDN.
QWEN3_14B_URL="https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q4_K_M.gguf"
QWEN3_14B_NAME="Qwen3-14B-Q4_K_M.gguf"
# Smaller dense variant for laptops with 8-16 GB RAM. Drop-in upgrade
# over Qwen 2.5 7B (better multilingual coverage, more recent training)
# at almost the same footprint (~5 GB Q4_K_M).
QWEN3_8B_URL="https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf"
QWEN3_8B_NAME="Qwen3-8B-Q4_K_M.gguf"
# Lite variant for machines with < 8 GB RAM. Qwen 3 4B offers excellent
# multilingual coverage (100+ languages), thinking mode for structured
# summaries, and fits comfortably in ~3.5 GB RAM including KV cache.
# Same chat template as all Qwen 3 models — zero code changes needed.
QWEN3_4B_URL="https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf"
QWEN3_4B_NAME="Qwen3-4B-Q4_K_M.gguf"
# MoE variant for Macs ≥32 GB. 30 B total / 3 B active per token —
# higher quality than the dense 32 B at a fraction of the inference
# cost. Recommended when the user runs the Quality profile.
QWEN3_MOE_URL="https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf"
QWEN3_MOE_NAME="Qwen3-30B-A3B-Q4_K_M.gguf"
# Legacy Qwen 2.5 GGUFs kept for back-compat with Sprint 1 day 9 setups
# and as a fallback when the Qwen 3 mirrors are unavailable.
QWEN_7B_URL="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf"
QWEN_7B_NAME="qwen2.5-7b-instruct-q4_k_m.gguf"
QWEN_3B_URL="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
QWEN_3B_NAME="qwen2.5-3b-instruct-q4_k_m.gguf"

# Spanish fine-tune of whisper-large-v3-turbo (5.34 % WER on Common
# Voice 17 ES vs 6.91 % for the upstream turbo) is built by the
# companion script `scripts/build-spanish-asr.sh` since the upstream
# only ships safetensors and we need ggml. The asr-es flavour below
# delegates to that helper instead of reimplementing the Python flow.

# --- cosmetics ---------------------------------------------------------------
if [[ -t 1 ]]; then
  GRN=$'\033[32m'; YLW=$'\033[33m'; RED=$'\033[31m'; BLD=$'\033[1m'; RST=$'\033[0m'
else
  GRN=""; YLW=""; RED=""; BLD=""; RST=""
fi
info() { printf "%s==>%s %s%s%s\n" "$GRN" "$RST" "$BLD" "$1" "$RST"; }
warn() { printf "%s!%s %s\n" "$YLW" "$RST" "$1"; }
fail() { printf "%s✗%s %s\n" "$RED" "$RST" "$1"; exit 1; }

# Approximate file sizes (in MiB) for sanity-check after download.
# Function instead of assoc array because macOS ships bash 3.2, which
# rejects keys containing dots in `declare -A`.
expected_size_mib() {
  case "$1" in
    tiny|tiny.en)             echo 75 ;;
    base|base.en)             echo 142 ;;
    small|small.en)           echo 466 ;;
    medium|medium.en)         echo 1500 ;;
    large-v3)                 echo 2900 ;;
    large-v3-turbo)           echo 1500 ;;
    large-v3-turbo-q5_0)      echo  550 ;;
    distil-large-v3)          echo  720 ;;
    *)                        echo 0 ;;
  esac
}

download_one() {
  local flavor="$1"
  local fname="ggml-${flavor}.bin"
  local out="${MODELS_DIR}/${fname}"
  local url="${HF_BASE}/${fname}"

  if [[ -f "$out" ]]; then
    local got_mib expected
    got_mib=$(( $(wc -c < "$out") / 1048576 ))
    expected="$(expected_size_mib "$flavor")"
    if (( expected > 0 )) && (( got_mib < expected * 9 / 10 )); then
      warn "${fname} present but truncated (${got_mib} MiB, expected ~${expected} MiB). Re-downloading."
      rm -f "$out"
    else
      info "${fname} already present (${got_mib} MiB) — skipping."
      return
    fi
  fi

  info "Fetching ${fname} → ${out}"
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --progress-bar --output "$out" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget --show-progress --output-document="$out" "$url"
  else
    fail "Neither curl nor wget is installed."
  fi

  local got_mib expected
  got_mib=$(( $(wc -c < "$out") / 1048576 ))
  expected="$(expected_size_mib "$flavor")"
  if (( expected > 0 )) && (( got_mib < expected * 9 / 10 )); then
    fail "${fname} downloaded incompletely (${got_mib} MiB vs ~${expected} MiB)"
  fi
  printf "  ${GRN}✓${RST} %s (%d MiB)\n" "$fname" "$got_mib"
}

sha256_of() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | cut -d' ' -f1
  else
    sha256sum "$1" | cut -d' ' -f1
  fi
}

download_silero_vad() {
  local out="${VAD_DIR}/silero_vad.onnx"
  # The untouched upstream file is cached next to the tract-ready build
  # so re-runs can re-inline without re-downloading. Its SHA-256 is the
  # readiness signal: a cached v5 upstream (or a truncated file) fails
  # the check and triggers a fresh v6.2 download.
  local raw="${out}.upstream"
  local inliner="${REPO_ROOT}/scripts/inline-silero-vad.py"

  if [[ -f "$raw" && "$(sha256_of "$raw")" == "$SILERO_VAD_SHA256" ]]; then
    info "Silero VAD v6.2.3 upstream already cached — skipping download."
  else
    info "Fetching Silero VAD v6.2.3 → ${raw}"
    if command -v curl >/dev/null 2>&1; then
      curl --fail --location --progress-bar --output "$raw" "$SILERO_VAD_URL"
    elif command -v wget >/dev/null 2>&1; then
      wget --show-progress --output-document="$raw" "$SILERO_VAD_URL"
    else
      fail "Neither curl nor wget is installed."
    fi
    local got_bytes
    got_bytes=$(wc -c < "$raw" | tr -d ' ')
    if (( got_bytes != SILERO_VAD_BYTES )); then
      rm -f "$raw"
      fail "Silero VAD downloaded incompletely (${got_bytes} B, expected ${SILERO_VAD_BYTES} B)"
    fi
    if [[ "$(sha256_of "$raw")" != "$SILERO_VAD_SHA256" ]]; then
      rm -f "$raw"
      fail "Silero VAD SHA-256 mismatch (expected ${SILERO_VAD_SHA256})"
    fi
  fi

  info "Inlining the 16 kHz branch of Silero VAD for tract-onnx"
  if ! command -v python3 >/dev/null 2>&1; then
    fail "python3 is required to inline Silero VAD. Install Python 3.9+ and re-run."
  fi
  if ! python3 -c "import onnx" >/dev/null 2>&1; then
    info "Installing inliner deps: pip install --user onnx"
    python3 -m pip install --quiet --user onnx \
      || fail "Could not install onnx. Install it manually and re-run."
  fi
  # Always re-inline from the verified upstream: it takes well under a
  # second and replaces any previous (e.g. v5) build in place.
  python3 "$inliner" --input "$raw" --output "$out" \
    || fail "Silero VAD inlining failed. Delete ${out} and ${raw} and re-run to start clean."

  # The inlined graph drops the 8 kHz weights: ~1,520 KiB.
  local got_kib
  got_kib=$(( $(wc -c < "$out") / 1024 ))
  if (( got_kib < 1400 )); then
    fail "silero_vad.onnx looks wrong after inlining (${got_kib} KiB, expected ~1,520 KiB)"
  fi
  printf "  ${GRN}✓${RST} silero_vad.onnx (%d KiB, tract-ready)\n" "$got_kib"
}

download_eres2net() {
  # ---- model -----------------------------------------------------------------
  local out="${EMBED_DIR}/eres2net_en_voxceleb.onnx"
  local need_model=1
  if [[ -f "$out" ]]; then
    local got_mib
    got_mib=$(( $(wc -c < "$out") / 1048576 ))
    if (( got_mib < 20 )); then
      warn "eres2net_en_voxceleb.onnx present but truncated (${got_mib} MiB, expected ~26 MiB). Re-downloading."
      rm -f "$out"
    else
      info "eres2net_en_voxceleb.onnx already present (${got_mib} MiB) — skipping."
      need_model=0
    fi
  fi

  if (( need_model )); then
    info "Fetching ERes2Net (3D-Speaker, EN VoxCeleb) → ${out}"
    if command -v curl >/dev/null 2>&1; then
      curl --fail --location --progress-bar --output "$out" "$ERES2NET_URL"
    elif command -v wget >/dev/null 2>&1; then
      wget --show-progress --output-document="$out" "$ERES2NET_URL"
    else
      fail "Neither curl nor wget is installed."
    fi
    local got_mib
    got_mib=$(( $(wc -c < "$out") / 1048576 ))
    if (( got_mib < 20 )); then
      fail "eres2net_en_voxceleb.onnx downloaded incompletely (${got_mib} MiB)"
    fi
    printf "  ${GRN}✓${RST} eres2net_en_voxceleb.onnx (%d MiB)\n" "$got_mib"
  fi

  # ---- companion test fixture ------------------------------------------------
  mkdir -p "$ERES2NET_FIXTURE_DIR"
  local fixture_out="${ERES2NET_FIXTURE_DIR}/two_speakers_en.wav"
  if [[ -f "$fixture_out" && $(wc -c < "$fixture_out") -gt 400000 ]]; then
    info "two_speakers_en.wav already present — skipping fixture."
    return
  fi

  info "Fetching two-speaker fixture → ${fixture_out}"
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --progress-bar --output "$fixture_out" "$ERES2NET_FIXTURE_URL"
  elif command -v wget >/dev/null 2>&1; then
    wget --show-progress --output-document="$fixture_out" "$ERES2NET_FIXTURE_URL"
  fi
  local fix_kib
  fix_kib=$(( $(wc -c < "$fixture_out") / 1024 ))
  printf "  ${GRN}✓${RST} two_speakers_en.wav (%d KiB)\n" "$fix_kib"
}

download_camplusplus() {
  local out="${EMBED_DIR}/campplus_en_voxceleb.onnx"
  local need_model=1
  if [[ -f "$out" ]]; then
    local got_mib
    got_mib=$(( $(wc -c < "$out") / 1048576 ))
    if (( got_mib < 20 )); then
      warn "campplus_en_voxceleb.onnx present but truncated (${got_mib} MiB, expected ~28 MiB). Re-downloading."
      rm -f "$out"
    else
      info "campplus_en_voxceleb.onnx already present (${got_mib} MiB) — skipping."
      need_model=0
    fi
  fi

  if (( need_model )); then
    info "Fetching CAM++ (3D-Speaker, EN VoxCeleb) → ${out}"
    if command -v curl >/dev/null 2>&1; then
      curl --fail --location --progress-bar --output "$out" "$CAMPP_URL"
    elif command -v wget >/dev/null 2>&1; then
      wget --show-progress --output-document="$out" "$CAMPP_URL"
    else
      fail "Neither curl nor wget is installed."
    fi
    local got_mib
    got_mib=$(( $(wc -c < "$out") / 1048576 ))
    if (( got_mib < 20 )); then
      fail "campplus_en_voxceleb.onnx downloaded incompletely (${got_mib} MiB)"
    fi
    printf "  ${GRN}✓${RST} campplus_en_voxceleb.onnx (%d MiB)\n" "$got_mib"
  fi

  download_camplusplus_fixture
}

# CAM++ multilingual embedder (zh_en 16k-common_advanced). Same adapter
# as the EN model (echo-diarize detects the 192-dim output at load), so
# it drops straight into the Models panel as an A/B alternative.
download_camplusplus_multi() {
  local out="${EMBED_DIR}/campplus_zh_en_common_advanced.onnx"
  local need_model=1
  if [[ -f "$out" ]]; then
    local got_mib
    got_mib=$(( $(wc -c < "$out") / 1048576 ))
    if (( got_mib < 20 )); then
      warn "campplus_zh_en_common_advanced.onnx present but truncated (${got_mib} MiB, expected ~27 MiB). Re-downloading."
      rm -f "$out"
    else
      info "campplus_zh_en_common_advanced.onnx already present (${got_mib} MiB) — skipping."
      need_model=0
    fi
  fi

  if (( need_model )); then
    info "Fetching CAM++ multilingual (3D-Speaker, zh_en common) → ${out}"
    if command -v curl >/dev/null 2>&1; then
      curl --fail --location --progress-bar --output "$out" "$CAMPP_MULTI_URL"
    elif command -v wget >/dev/null 2>&1; then
      wget --show-progress --output-document="$out" "$CAMPP_MULTI_URL"
    else
      fail "Neither curl nor wget is installed."
    fi
    local got_mib
    got_mib=$(( $(wc -c < "$out") / 1048576 ))
    if (( got_mib < 20 )); then
      fail "campplus_zh_en_common_advanced.onnx downloaded incompletely (${got_mib} MiB)"
    fi
    printf "  ${GRN}✓${RST} campplus_zh_en_common_advanced.onnx (%d MiB)\n" "$got_mib"
  fi
}

# Two-speaker fixture shared by the CAM++ / ERes2Net adapter tests.
download_camplusplus_fixture() {
  mkdir -p "$ERES2NET_FIXTURE_DIR"
  local fixture_out="${ERES2NET_FIXTURE_DIR}/two_speakers_en.wav"
  if [[ -f "$fixture_out" && $(wc -c < "$fixture_out") -gt 400000 ]]; then
    info "two_speakers_en.wav already present — skipping fixture."
    return
  fi
  info "Fetching two-speaker fixture → ${fixture_out}"
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --progress-bar --output "$fixture_out" "$ERES2NET_FIXTURE_URL"
  elif command -v wget >/dev/null 2>&1; then
    wget --show-progress --output-document="$fixture_out" "$ERES2NET_FIXTURE_URL"
  fi
  local fix_kib
  fix_kib=$(( $(wc -c < "$fixture_out") / 1024 ))
  printf "  ${GRN}✓${RST} two_speakers_en.wav (%d KiB)\n" "$fix_kib"
}

download_pyannote_segmenter() {
  local out="${SEG_DIR}/pyannote_segmentation_3.onnx"
  local need=1
  # Upstream sherpa export is ~5.7 MiB; anything under 4 MiB is a truncated
  # download.
  if [[ -f "$out" ]]; then
    local got_mib
    got_mib=$(( $(wc -c < "$out") / 1048576 ))
    if (( got_mib < 4 )); then
      warn "pyannote_segmentation_3.onnx present but truncated (${got_mib} MiB, expected ~5.7 MiB). Re-downloading."
      rm -f "$out"
    else
      info "pyannote_segmentation_3.onnx already present (${got_mib} MiB) — skipping download."
      need=0
    fi
  fi

  if (( need )); then
    info "Fetching pyannote-segmentation-3.0 (sherpa-onnx) → ${out}"
    if command -v curl >/dev/null 2>&1; then
      curl --fail --location --progress-bar --output "$out" "$PYANNOTE_SEG_URL"
    elif command -v wget >/dev/null 2>&1; then
      wget --show-progress --output-document="$out" "$PYANNOTE_SEG_URL"
    else
      fail "Neither curl nor wget is installed."
    fi
    local got_mib
    got_mib=$(( $(wc -c < "$out") / 1048576 ))
    if (( got_mib < 4 )); then
      fail "pyannote_segmentation_3.onnx downloaded incompletely (${got_mib} MiB)"
    fi
    printf "  ${GRN}✓${RST} pyannote_segmentation_3.onnx (%d MiB)\n" "$got_mib"
  fi

  # The upstream export does NOT load in the pure-Rust tract engine
  # (`InstanceNormalization` fails analysis). Produce a tract-compatible,
  # numerically-identical variant that echo-shell prefers at runtime.
  local tract_out="${SEG_DIR}/pyannote_segmentation_3_tract.onnx"
  if [[ -f "$tract_out" && "$tract_out" -nt "$out" ]]; then
    info "pyannote_segmentation_3_tract.onnx already up to date — skipping patch."
    return
  fi
  if python3 -c "import onnx" >/dev/null 2>&1; then
    info "Patching segmenter for tract → ${tract_out}"
    if python3 "${REPO_ROOT}/scripts/patch-pyannote-segmenter.py" "$out" "$tract_out"; then
      printf "  ${GRN}✓${RST} pyannote_segmentation_3_tract.onnx (tract-compatible)\n"
    else
      warn "segmenter patch failed; tract builds will run whole-chunk (no overlap-aware segmentation)."
    fi
  else
    warn "python3 + onnx not available; skipping tract patch. Install with 'pip install onnx onnxruntime' and re-run, or overlap-aware segmentation stays disabled on tract builds."
  fi
}

download_llm() {
  local url="$1"
  local fname="$2"
  local expected_mib="$3"
  local out="${LLM_DIR}/${fname}"

  if [[ -f "$out" ]]; then
    local got_mib
    got_mib=$(( $(wc -c < "$out") / 1048576 ))
    if (( got_mib < expected_mib * 9 / 10 )); then
      warn "${fname} present but truncated (${got_mib} MiB, expected ~${expected_mib} MiB). Re-downloading."
      rm -f "$out"
    else
      info "${fname} already present (${got_mib} MiB) — skipping."
      return
    fi
  fi

  info "Fetching ${fname} → ${out}"
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --progress-bar --output "$out" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget --show-progress --output-document="$out" "$url"
  else
    fail "Neither curl nor wget is installed."
  fi

  local got_mib
  got_mib=$(( $(wc -c < "$out") / 1048576 ))
  if (( got_mib < expected_mib * 9 / 10 )); then
    fail "${fname} downloaded incompletely (${got_mib} MiB vs ~${expected_mib} MiB)"
  fi
  printf "  ${GRN}✓${RST} %s (%d MiB)\n" "$fname" "$got_mib"
}

main() {
  # Default to the multilingual large-v3-turbo since EchoNote is a
  # Spanish-first product; the English-only `base.en` legacy default
  # is still reachable via `scripts/download-models.sh base.en`.
  local choice="${1:-large-v3-turbo}"
  case "$choice" in
    --all)
      download_one large-v3-turbo
      download_silero_vad
      # CAM++ is the recommended embedder for Spanish-first meetings;
      # ERes2Net is fetched too so callers can A/B-compare via the
      # Models panel. `preferred_embed_model()` in echo-shell picks
      # CAM++ when both files are present.
      download_camplusplus
      download_eres2net
      download_pyannote_segmenter
      ;;
    --help|-h)
      sed -n '4,35p' "$0"
      exit 0
      ;;
    vad|silero|silero-vad)
      download_silero_vad
      info "VAD model in ${VAD_DIR}. Set ECHO_VAD_MODEL=${VAD_DIR}/silero_vad.onnx if you move it."
      return
      ;;
    embed|embedder|eres2net)
      download_eres2net
      info "Embedder in ${EMBED_DIR}. Set ECHO_EMBED_MODEL=${EMBED_DIR}/eres2net_en_voxceleb.onnx if you move it."
      return
      ;;
    cam-plus-plus|camplusplus|cam++)
      # Recommended embedder for Spanish-first meetings (lower EER,
      # multilingual training vs ERes2Net VoxCeleb-EN).
      download_camplusplus
      info "CAM++ embedder in ${EMBED_DIR}. Set ECHO_EMBED_MODEL=${EMBED_DIR}/campplus_en_voxceleb.onnx if you move it."
      return
      ;;
    cam-plus-plus-multi|camplusplus-multi|cam++-multi|embed-multi)
      # Multilingual CAM++ (zh_en common). A/B alternative for reducing
      # over-splitting on Spanish audio vs the EN-VoxCeleb export.
      download_camplusplus_multi
      info "CAM++ multilingual embedder in ${EMBED_DIR}. Select it in the Models panel or set ECHO_EMBED_MODEL=${EMBED_DIR}/campplus_zh_en_common_advanced.onnx."
      return
      ;;
    segmenter|pyannote|pyannote-segmentation)
      # pyannote-segmentation-3.0 ONNX for sub-chunk speaker boundary
      # detection. Used by PyannoteSegmenter in echo-diarize (~17 MB).
      download_pyannote_segmenter
      info "Segmenter in ${SEG_DIR}. Set ECHO_SEG_MODEL=${SEG_DIR}/pyannote_segmentation_3.onnx if you move it."
      return
      ;;
    llm|llm-14b|qwen3-14b)
      # Default summary model (Spanish-first, DEVELOPMENT_PLAN.md §3.1
      # CU-04 + ARCHITECTURE.md profile "Balanced"). ~9 GB on disk.
      download_llm "$QWEN3_14B_URL" "$QWEN3_14B_NAME" 8800
      info "LLM in ${LLM_DIR}. Set ECHO_LLM_MODEL=${LLM_DIR}/${QWEN3_14B_NAME} if you move it."
      return
      ;;
    llm-small|llm-8b|qwen3-8b)
      # Lighter alternative for 8-16 GB RAM hosts and dev iteration.
      download_llm "$QWEN3_8B_URL" "$QWEN3_8B_NAME" 5000
      info "LLM in ${LLM_DIR}. Set ECHO_LLM_MODEL=${LLM_DIR}/${QWEN3_8B_NAME} if you move it."
      return
      ;;
    llm-lite|llm-4b|qwen3-4b)
      # Smallest Qwen 3 variant for machines with < 8 GB RAM.
      # ~2.5 GB on disk, ~3.5 GB RAM with KV cache.
      download_llm "$QWEN3_4B_URL" "$QWEN3_4B_NAME" 2500
      info "LLM in ${LLM_DIR}. Set ECHO_LLM_MODEL=${LLM_DIR}/${QWEN3_4B_NAME} if you move it."
      return
      ;;
    llm-moe|llm-30b|qwen3-30b|qwen3-30b-a3b)
      # MoE variant for ≥32 GB RAM hosts (Quality profile). 30 B total
      # / 3 B active per token: higher quality than dense 32 B at
      # comparable inference latency.
      download_llm "$QWEN3_MOE_URL" "$QWEN3_MOE_NAME" 18000
      info "LLM in ${LLM_DIR}. Set ECHO_LLM_MODEL=${LLM_DIR}/${QWEN3_MOE_NAME} if you move it."
      return
      ;;
    llm-legacy-7b|qwen-7b|qwen2.5-7b)
      # Legacy Qwen 2.5 7B for back-compat with Sprint 1 day 9 setups.
      download_llm "$QWEN_7B_URL" "$QWEN_7B_NAME" 4400
      info "LLM in ${LLM_DIR}. Set ECHO_LLM_MODEL=${LLM_DIR}/${QWEN_7B_NAME} if you move it."
      return
      ;;
    llm-legacy-3b|qwen-3b|qwen2.5-3b)
      # Legacy Qwen 2.5 3B for back-compat.
      download_llm "$QWEN_3B_URL" "$QWEN_3B_NAME" 1900
      info "LLM in ${LLM_DIR}. Set ECHO_LLM_MODEL=${LLM_DIR}/${QWEN_3B_NAME} if you move it."
      return
      ;;
    asr-es|spanish|whisper-es)
      # Spanish fine-tune of whisper-large-v3-turbo. The model is only
      # distributed as safetensors, so the actual conversion to ggml
      # (and Q5_0 quantization) lives in the companion script. We
      # delegate without re-implementing the Python toolchain checks.
      local helper="${REPO_ROOT}/scripts/build-spanish-asr.sh"
      if [[ ! -x "$helper" ]]; then
        fail "Helper script not found or not executable: ${helper}"
      fi
      "$helper"
      return
      ;;
    distil-large-v3)
      # Distil-Whisper uses a different HF repo — handle separately.
      local fname="ggml-distil-large-v3.bin"
      local out="${MODELS_DIR}/${fname}"
      if [[ -f "$out" ]]; then
        local got_mib
        got_mib=$(( $(wc -c < "$out") / 1048576 ))
        if (( got_mib < 650 )); then
          warn "${fname} present but truncated (${got_mib} MiB). Re-downloading."
          rm -f "$out"
        else
          info "${fname} already present (${got_mib} MiB) — skipping."
          return
        fi
      fi
      info "Fetching ${fname} → ${out}"
      if command -v curl >/dev/null 2>&1; then
        curl --fail --location --progress-bar --output "$out" "$DISTIL_V3_URL"
      elif command -v wget >/dev/null 2>&1; then
        wget --show-progress --output-document="$out" "$DISTIL_V3_URL"
      else
        fail "Neither curl nor wget is installed."
      fi
      local got_mib
      got_mib=$(( $(wc -c < "$out") / 1048576 ))
      printf "  ${GRN}✓${RST} %s (%d MiB)\n" "$fname" "$got_mib"
      ;;
    tiny|tiny.en|base|base.en|small|small.en|medium|medium.en|large-v3|large-v3-turbo|large-v3-turbo-q5_0)
      download_one "$choice"
      ;;
    *)
      fail "Unknown flavor: ${choice}. Run with --help."
      ;;
  esac

  info "All requested models are in ${REPO_ROOT}/models/"
  info "Try: cargo run -p echo-proto -- transcribe /tmp/sample.wav --language es --model ${ASR_DIR}/ggml-${choice}.bin"
}

main "$@"
