#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# EchoNote — build the Spanish-fine-tuned Whisper ggml model.
#
# Pulls `adriszmar/whisper-large-v3-turbo-es` (MIT, 5.34 % WER on Common
# Voice 17 ES vs 6.91 % for the upstream turbo) from Hugging Face and
# converts it to the ggml format whisper.cpp expects, dropping the
# result at `./models/asr/ggml-large-v3-turbo-es.bin` so
# `preferred_asr_model()` picks it up automatically.
#
# Why this is its own script instead of inlined into `download-models.sh`:
# the conversion needs Python ≥ 3.10, ~3 GB of transient downloads
# (HF model + openai/whisper repo) and ~5 minutes of CPU work — heavy
# enough that the main downloader stays a pure curl-based script.
#
# Usage:
#   ./scripts/build-spanish-asr.sh                 # default flow
#   ASR_ES_REPO=other/repo ./scripts/build-spanish-asr.sh  # override source
#   ASR_ES_KEEP_TMP=1 ./scripts/build-spanish-asr.sh       # keep cache
#
# The script is idempotent: re-running with the output already present
# is a no-op. Pass `--force` to rebuild from scratch.
# -----------------------------------------------------------------------------
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ASR_DIR="${REPO_ROOT}/models/asr"
TMP_DIR="${REPO_ROOT}/models/.tmp/spanish-asr"
VENV_DIR="${TMP_DIR}/venv"
HF_DIR="${TMP_DIR}/hf-model"
WHISPER_REPO_DIR="${TMP_DIR}/openai-whisper"
WHISPER_CPP_DIR="${TMP_DIR}/whisper.cpp"

ASR_ES_REPO="${ASR_ES_REPO:-adriszmar/whisper-large-v3-turbo-es}"
OUTPUT_NAME="ggml-large-v3-turbo-es.bin"
OUTPUT_PATH="${ASR_DIR}/${OUTPUT_NAME}"

# --- cosmetics ---------------------------------------------------------------
if [[ -t 1 ]]; then
  GRN=$'\033[32m'; YLW=$'\033[33m'; RED=$'\033[31m'; BLD=$'\033[1m'; RST=$'\033[0m'
else
  GRN=""; YLW=""; RED=""; BLD=""; RST=""
fi
info() { printf "%s==>%s %s%s%s\n" "$GRN" "$RST" "$BLD" "$1" "$RST"; }
warn() { printf "%s!%s %s\n" "$YLW" "$RST" "$1"; }
fail() { printf "%s✗%s %s\n" "$RED" "$RST" "$1"; exit 1; }

# --- arg parsing -------------------------------------------------------------
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force|-f) FORCE=1 ;;
    --help|-h)  sed -n '4,22p' "$0"; exit 0 ;;
    *) fail "Unknown argument: ${arg}" ;;
  esac
done

# --- early-exit: model already built ----------------------------------------
if [[ -f "$OUTPUT_PATH" && $FORCE -eq 0 ]]; then
  size_mib=$(( $(wc -c < "$OUTPUT_PATH") / 1048576 ))
  info "${OUTPUT_NAME} already present (${size_mib} MiB) — skipping build."
  info "Re-run with --force to rebuild from scratch."
  exit 0
fi

# --- preflight ---------------------------------------------------------------
command -v python3 >/dev/null 2>&1 || fail "python3 is required (>= 3.10). Install via Homebrew: brew install python@3.11"
command -v git >/dev/null 2>&1 || fail "git is required."

PY_VERSION="$(python3 -c 'import sys; print("{}.{}".format(*sys.version_info))')"
PY_MAJOR="${PY_VERSION%%.*}"
PY_MINOR="${PY_VERSION#*.}"
if (( PY_MAJOR < 3 )) || (( PY_MAJOR == 3 && PY_MINOR < 10 )); then
  fail "python3 ${PY_VERSION} is too old. Need >= 3.10 (transformers + torch wheels)."
fi
info "python3 ${PY_VERSION} detected"

mkdir -p "$ASR_DIR" "$TMP_DIR"

# --- step 1: venv ------------------------------------------------------------
if [[ ! -d "$VENV_DIR" ]]; then
  info "Creating Python venv at ${VENV_DIR}"
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
trap 'deactivate >/dev/null 2>&1 || true' EXIT

info "Installing/upgrading conversion dependencies (transformers, torch, hf_hub)"
pip install --quiet --upgrade pip
# `tiktoken` is needed by openai/whisper's tokenizer; `torch` powers
# the safetensors → fp32 read path inside convert-h5-to-ggml.py.
pip install --quiet \
  "transformers>=4.40" \
  "torch>=2.2" \
  "tiktoken>=0.7" \
  "huggingface_hub>=0.24" \
  "safetensors>=0.4" \
  "numpy>=1.26"

# --- step 2: openai/whisper checkout (needed by convert script) --------------
if [[ ! -d "$WHISPER_REPO_DIR/.git" ]]; then
  info "Cloning openai/whisper (sparse, for tokenizer assets)"
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/openai/whisper "$WHISPER_REPO_DIR"
  (
    cd "$WHISPER_REPO_DIR"
    git sparse-checkout set whisper
  )
fi

# --- step 3: whisper.cpp convert script -------------------------------------
if [[ ! -d "$WHISPER_CPP_DIR/.git" ]]; then
  info "Cloning whisper.cpp (sparse, for the conversion script)"
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/ggml-org/whisper.cpp "$WHISPER_CPP_DIR"
  (
    cd "$WHISPER_CPP_DIR"
    git sparse-checkout set models
  )
fi

CONVERT_SCRIPT="${WHISPER_CPP_DIR}/models/convert-h5-to-ggml.py"
[[ -f "$CONVERT_SCRIPT" ]] || fail "convert script not found at ${CONVERT_SCRIPT} after clone."

# --- step 4: download fine-tune from HF -------------------------------------
if [[ ! -f "${HF_DIR}/config.json" ]]; then
  info "Downloading ${ASR_ES_REPO} from Hugging Face (~3 GB safetensors)"
  python3 - "$ASR_ES_REPO" "$HF_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id, local_dir = sys.argv[1], sys.argv[2]
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    # We only need the architecture + weights + tokenizer files; the
    # README, .gitattributes, and example audios are dead weight.
    allow_patterns=[
        "*.json",
        "*.safetensors",
        "*.bin",                 # in case the repo ships pytorch_model.bin
        "tokenizer*",
        "vocab*",
        "added_tokens*",
        "merges*",
        "normalizer*",
        "special_tokens_map*",
        "preprocessor_config*",
        "generation_config*",
    ],
)
print(f"downloaded {repo_id} to {local_dir}")
PY
else
  info "Hugging Face model already cached at ${HF_DIR} — skipping download."
fi

# --- step 5: convert to ggml -------------------------------------------------
info "Converting to ggml (fp16) — this can take 2–5 minutes on Apple Silicon"
CONVERT_OUT_DIR="${TMP_DIR}/ggml-out"
mkdir -p "$CONVERT_OUT_DIR"

# The script always names its output `ggml-model.bin` in the target dir,
# regardless of the input model. We rename afterwards.
(
  cd "$WHISPER_CPP_DIR"
  python3 "$CONVERT_SCRIPT" \
    "$HF_DIR" \
    "$WHISPER_REPO_DIR" \
    "$CONVERT_OUT_DIR"
)

GENERATED="${CONVERT_OUT_DIR}/ggml-model.bin"
[[ -f "$GENERATED" ]] || fail "conversion did not produce ${GENERATED}"

# --- step 6: install -------------------------------------------------------
mv "$GENERATED" "$OUTPUT_PATH"
size_mib=$(( $(wc -c < "$OUTPUT_PATH") / 1048576 ))
info "Wrote ${OUTPUT_NAME} (${size_mib} MiB) to ${ASR_DIR}"

# --- cleanup ---------------------------------------------------------------
if [[ "${ASR_ES_KEEP_TMP:-0}" != "1" ]]; then
  info "Cleaning up scratch dir ${TMP_DIR} (set ASR_ES_KEEP_TMP=1 to keep it)"
  # Keep the venv so re-runs don't reinstall torch (~700 MB on disk).
  rm -rf "$HF_DIR" "$CONVERT_OUT_DIR"
fi

cat <<EOF

${GRN}✓${RST} Spanish ASR model ready: ${OUTPUT_PATH}

Next steps:
  - Re-launch the desktop shell; ${OUTPUT_NAME} now wins
    \`preferred_asr_model()\` over the upstream turbo.
  - Or pin it explicitly: export ECHO_ASR_MODEL="${OUTPUT_PATH}"
  - To shrink it (~1.6 GB → ~550 MB) build whisper.cpp's quantize tool
    and run: ./quantize ${OUTPUT_PATH} ${OUTPUT_PATH%.bin}-q5_0.bin q5_0
EOF
