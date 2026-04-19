#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# EchoNote — download Whisper / VAD / LLM models from their canonical sources.
#
# Usage:
#   scripts/download-models.sh                # default: ggml-base.en
#   scripts/download-models.sh small.en       # ~466 MB
#   scripts/download-models.sh medium         # ~1.5 GB, multilingual
#   scripts/download-models.sh large-v3       # ~3.0 GB
#   scripts/download-models.sh vad            # Silero VAD v5 (~2 MB)
#   scripts/download-models.sh embed          # 3D-Speaker ERes2Net (~26 MB)
#   scripts/download-models.sh llm            # Qwen 2.5 7B Instruct Q4_K_M (~4.4 GB)
#   scripts/download-models.sh llm-small      # Qwen 2.5 3B Instruct Q4_K_M (~1.9 GB)
#   scripts/download-models.sh --all          # base.en + small.en + vad + embed (no LLM)
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
LLM_DIR="${REPO_ROOT}/models/llm"
mkdir -p "$ASR_DIR" "$VAD_DIR" "$EMBED_DIR" "$LLM_DIR"
# Back-compat alias for code paths that still reference $MODELS_DIR.
MODELS_DIR="$ASR_DIR"

# Hugging Face repository that mirrors the ggerganov/whisper.cpp models.
HF_BASE="https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
# Silero VAD v5 lives in the upstream GitHub repo. Pinned to a commit
# so we get reproducible downloads across machines and over time.
SILERO_VAD_URL="https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx"
# 3D-Speaker ERes2Net (English VoxCeleb) — the speaker embedder used by
# echo-diarize. Mirrored on Hugging Face by csukuangfj (sherpa-onnx
# maintainer); upstream lives on ModelScope. ~26 MB, opset 13, outputs
# a 192-dim embedding from 80-bin Kaldi fbank features.
ERES2NET_URL="https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main/3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx"
# Two-speaker English fixture from the same HF mirror. Used by the
# echo-diarize integration tests as ground-truth cross-speaker audio.
ERES2NET_FIXTURE_URL="https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main/1-two-speakers-en.wav"
ERES2NET_FIXTURE_DIR="${REPO_ROOT}/crates/echo-diarize/tests/fixtures"

# Default LLM for summaries. Qwen 2.5 7B Instruct is multilingual (good
# Spanish quality), permissively licensed (Apache 2.0), and the Q4_K_M
# quantization fits in ~5 GB of RAM with ~6-8 t/s on Apple Silicon —
# right inside the < 45 s/30 min target from DEVELOPMENT_PLAN.md §3.1
# CU-04. The mirror is the official Qwen team's HF repo.
QWEN_7B_URL="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf"
QWEN_7B_NAME="qwen2.5-7b-instruct-q4_k_m.gguf"
# Smaller variant for laptops with limited RAM and faster iteration in
# development. Same family + license, just fewer parameters.
QWEN_3B_URL="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
QWEN_3B_NAME="qwen2.5-3b-instruct-q4_k_m.gguf"

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

download_silero_vad() {
  local out="${VAD_DIR}/silero_vad.onnx"
  if [[ -f "$out" ]]; then
    local got_kib
    got_kib=$(( $(wc -c < "$out") / 1024 ))
    if (( got_kib < 1500 )); then
      warn "silero_vad.onnx present but truncated (${got_kib} KiB, expected ~2200 KiB). Re-downloading."
      rm -f "$out"
    else
      info "silero_vad.onnx already present (${got_kib} KiB) — skipping."
      return
    fi
  fi

  info "Fetching Silero VAD v5 → ${out}"
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --progress-bar --output "$out" "$SILERO_VAD_URL"
  elif command -v wget >/dev/null 2>&1; then
    wget --show-progress --output-document="$out" "$SILERO_VAD_URL"
  else
    fail "Neither curl nor wget is installed."
  fi

  local got_kib
  got_kib=$(( $(wc -c < "$out") / 1024 ))
  if (( got_kib < 1500 )); then
    fail "silero_vad.onnx downloaded incompletely (${got_kib} KiB)"
  fi
  printf "  ${GRN}✓${RST} silero_vad.onnx (%d KiB)\n" "$got_kib"
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
  local choice="${1:-base.en}"
  case "$choice" in
    --all)
      download_one base.en
      download_one small.en
      download_silero_vad
      download_eres2net
      ;;
    --help|-h)
      sed -n '4,20p' "$0"
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
    llm|llm-7b|qwen-7b|qwen2.5-7b)
      # Default summary model (DEVELOPMENT_PLAN.md §3.1 CU-04 +
      # ARCHITECTURE.md profile "Balanced"). ~4.4 GB on disk.
      download_llm "$QWEN_7B_URL" "$QWEN_7B_NAME" 4400
      info "LLM in ${LLM_DIR}. Set ECHO_LLM_MODEL=${LLM_DIR}/${QWEN_7B_NAME} if you move it."
      return
      ;;
    llm-small|llm-3b|qwen-3b|qwen2.5-3b)
      # Lighter alternative for low-RAM hosts and dev iteration.
      download_llm "$QWEN_3B_URL" "$QWEN_3B_NAME" 1900
      info "LLM in ${LLM_DIR}. Set ECHO_LLM_MODEL=${LLM_DIR}/${QWEN_3B_NAME} if you move it."
      return
      ;;
    tiny|tiny.en|base|base.en|small|small.en|medium|medium.en|large-v3|large-v3-turbo)
      download_one "$choice"
      ;;
    *)
      fail "Unknown flavor: ${choice}. Run with --help."
      ;;
  esac

  info "All requested models are in ${REPO_ROOT}/models/"
  info "Try: cargo run -p echo-proto -- transcribe /tmp/sample.wav --model ${ASR_DIR}/ggml-${choice%.en}.en.bin"
}

main "$@"
