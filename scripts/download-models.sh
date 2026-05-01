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
#   scripts/download-models.sh vad            # Silero VAD v5.1.2 (~2 MB)
#   scripts/download-models.sh embed          # 3D-Speaker ERes2Net (~26 MB)
#   scripts/download-models.sh segmenter      # pyannote-segmentation-3.0 (~17 MB)
#   scripts/download-models.sh llm            # Qwen 3 14B Instruct Q4_K_M (~9 GB)
#   scripts/download-models.sh llm-small      # Qwen 3 8B Instruct Q4_K_M (~5 GB)
#   scripts/download-models.sh llm-lite       # Qwen 3 4B Q4_K_M (~2.5 GB, <8 GB RAM)
#   scripts/download-models.sh llm-moe        # Qwen 3 30B-A3B Instruct Q4_K_M (~18 GB)
#   scripts/download-models.sh llm-legacy-7b  # Qwen 2.5 7B (back-compat, ~4.4 GB)
#   scripts/download-models.sh llm-legacy-3b  # Qwen 2.5 3B (back-compat, ~1.9 GB)
#   scripts/download-models.sh --all          # large-v3-turbo + vad + embed (no LLM)
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
# Silero VAD v5.1.2 (last v5 release). Pinned to a tagged release so we
# get reproducible downloads across machines and over time.
#
# The upstream v5 ONNX *does* contain a control-flow `If` operator
# (dispatching between the 16 kHz and 8 kHz sub-networks on the `sr`
# input). pure-Rust `tract-onnx` — the inference backend used by
# `crates/echo-audio/src/preprocess/silero_vad.rs` — does not
# implement `If`, so loading the raw file errors with
# `optimize: Failed analyse for node #5 "If_0"`.
#
# We work around this by post-processing the ONNX after download with
# `scripts/simplify-silero-vad.py`: inline the 16 kHz branch, drop the
# now-orphan `sr` input and let ORT's constant-folding eliminate the
# nested shape-dependent `If`s. The resulting graph is ~31 nodes of
# pure feed-forward + LSTM, bitwise-equivalent to the upstream model
# at 16 kHz, and loads cleanly in tract.
#
# v6 was evaluated and rejected: it changed the state signature
# (`[2,1,128]` → two separate LSTM states) so the Rust adapter would
# need a full rewrite on top of the simplification; the ~16% noisy-
# audio WER improvement it promises is not worth that churn for MVP.
SILERO_VAD_URL="https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx"
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

download_silero_vad() {
  local out="${VAD_DIR}/silero_vad.onnx"
  local simplifier="${REPO_ROOT}/scripts/simplify-silero-vad.py"

  # Caching strategy: file size alone is NOT a reliable readiness signal.
  # A previous version of the simplifier ran ORT at ENABLE_ALL and emitted
  # files of the same ~1220 KiB as the current BASIC pipeline but laced
  # with FusedConv (an ORT contrib op tract rejects). To recover those
  # users transparently, we always run the simplifier — its
  # `already_simplified` check is O(few-ms) and correctly inspects op
  # types, not just byte counts.
  local need_download=1
  if [[ -f "$out" ]]; then
    local got_kib
    got_kib=$(( $(wc -c < "$out") / 1024 ))
    if (( got_kib < 1000 )); then
      warn "silero_vad.onnx present but truncated (${got_kib} KiB). Re-downloading."
    else
      info "silero_vad.onnx already on disk (${got_kib} KiB) — will verify with simplifier."
      need_download=0
    fi
  fi

  if (( need_download )); then
    info "Fetching Silero VAD v5.1.2 → ${out}"
    if command -v curl >/dev/null 2>&1; then
      curl --fail --location --progress-bar --output "$out" "$SILERO_VAD_URL"
    elif command -v wget >/dev/null 2>&1; then
      wget --show-progress --output-document="$out" "$SILERO_VAD_URL"
    else
      fail "Neither curl nor wget is installed."
    fi
    local got_kib
    got_kib=$(( $(wc -c < "$out") / 1024 ))
    if (( got_kib < 1000 )); then
      fail "silero_vad.onnx downloaded incompletely (${got_kib} KiB)"
    fi
  fi

  info "Verifying / simplifying Silero VAD for tract-onnx (BASIC ORT level, no contrib ops)"
  if ! command -v python3 >/dev/null 2>&1; then
    fail "python3 is required to simplify Silero VAD. Install Python 3.9+ and re-run."
  fi
  if ! python3 -c "import onnx, onnxruntime" >/dev/null 2>&1; then
    info "Installing simplifier deps: pip install --user onnx onnxruntime"
    python3 -m pip install --quiet --user onnx onnxruntime \
      || fail "Could not install onnx/onnxruntime. Install them manually and re-run."
  fi
  # The simplifier is idempotent: it short-circuits on a clean output
  # and re-emits a fresh BASIC build on a stale FusedConv-tainted file.
  python3 "$simplifier" --input "$out" --output "$out" \
    || fail "Silero VAD simplification failed. Delete ${out} and ${out}.upstream and re-run to start clean."

  local got_kib
  got_kib=$(( $(wc -c < "$out") / 1024 ))
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

  # Share the two-speaker fixture with ERes2Net — download it if missing.
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
  if [[ -f "$out" ]]; then
    local got_mib
    got_mib=$(( $(wc -c < "$out") / 1048576 ))
    if (( got_mib < 10 )); then
      warn "pyannote_segmentation_3.onnx present but truncated (${got_mib} MiB, expected ~17 MiB). Re-downloading."
      rm -f "$out"
    else
      info "pyannote_segmentation_3.onnx already present (${got_mib} MiB) — skipping."
      return
    fi
  fi

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
  if (( got_mib < 10 )); then
    fail "pyannote_segmentation_3.onnx downloaded incompletely (${got_mib} MiB)"
  fi
  printf "  ${GRN}✓${RST} pyannote_segmentation_3.onnx (%d MiB)\n" "$got_mib"
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
      download_eres2net
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
