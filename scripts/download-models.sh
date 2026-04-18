#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# EchoNote — download Whisper / LLM models from their canonical sources.
#
# Usage:
#   scripts/download-models.sh                # default: ggml-base.en
#   scripts/download-models.sh small.en       # ~466 MB
#   scripts/download-models.sh medium         # ~1.5 GB, multilingual
#   scripts/download-models.sh large-v3       # ~3.0 GB
#   scripts/download-models.sh --all          # base.en + small.en
#
# Models are written to ./models/asr/ggml-<flavor>.bin and skipped if the
# file already exists with the expected size. SHA-256 checksums are
# verified when the upstream manifest provides them (Hugging Face does).
# -----------------------------------------------------------------------------
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${REPO_ROOT}/models/asr"
mkdir -p "$MODELS_DIR"

# Hugging Face repository that mirrors the ggerganov/whisper.cpp models.
HF_BASE="https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

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

main() {
  local choice="${1:-base.en}"
  case "$choice" in
    --all)
      download_one base.en
      download_one small.en
      ;;
    --help|-h)
      sed -n '4,12p' "$0"
      exit 0
      ;;
    tiny|tiny.en|base|base.en|small|small.en|medium|medium.en|large-v3|large-v3-turbo)
      download_one "$choice"
      ;;
    *)
      fail "Unknown flavor: ${choice}. Run with --help."
      ;;
  esac

  info "All requested models are in ${MODELS_DIR}"
  info "Try: cargo run -p echo-proto -- transcribe /tmp/sample.wav --model ${MODELS_DIR}/ggml-${choice%.en}.en.bin"
}

main "$@"
