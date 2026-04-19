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
#   scripts/download-models.sh --all          # base.en + small.en + vad
#
# ASR models are written to ./models/asr/ggml-<flavor>.bin
# VAD model lives at  ./models/vad/silero_vad.onnx
# Existing files are skipped if their size looks right.
# -----------------------------------------------------------------------------
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ASR_DIR="${REPO_ROOT}/models/asr"
VAD_DIR="${REPO_ROOT}/models/vad"
mkdir -p "$ASR_DIR" "$VAD_DIR"
# Back-compat alias for code paths that still reference $MODELS_DIR.
MODELS_DIR="$ASR_DIR"

# Hugging Face repository that mirrors the ggerganov/whisper.cpp models.
HF_BASE="https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
# Silero VAD v5 lives in the upstream GitHub repo. Pinned to a commit
# so we get reproducible downloads across machines and over time.
SILERO_VAD_URL="https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx"

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

main() {
  local choice="${1:-base.en}"
  case "$choice" in
    --all)
      download_one base.en
      download_one small.en
      download_silero_vad
      ;;
    --help|-h)
      sed -n '4,15p' "$0"
      exit 0
      ;;
    vad|silero|silero-vad)
      download_silero_vad
      info "VAD model in ${VAD_DIR}. Set ECHO_VAD_MODEL=${VAD_DIR}/silero_vad.onnx if you move it."
      return
      ;;
    tiny|tiny.en|base|base.en|small|small.en|medium|medium.en|large-v3|large-v3-turbo)
      download_one "$choice"
      ;;
    *)
      fail "Unknown flavor: ${choice}. Run with --help."
      ;;
  esac

  info "All requested models are in ${ASR_DIR}"
  info "Try: cargo run -p echo-proto -- transcribe /tmp/sample.wav --model ${ASR_DIR}/ggml-${choice%.en}.en.bin"
}

main "$@"
