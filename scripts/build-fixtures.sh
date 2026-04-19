#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# scripts/build-fixtures.sh
#
# Generate synthetic ASR fixtures from the gold transcripts under
# `fixtures/transcripts/*.txt`. Uses the macOS `say` command + `afconvert`
# to produce 16 kHz mono PCM WAV files alongside each transcript.
#
# WAV files are intentionally NOT committed to the repo (they're large and
# the synthetic voice is recoverable from this script). The CI bench job
# runs this script first.
#
# Requirements: macOS (say, afconvert). Voice defaults to Samantha; override
# with VOICE=<name>.
# -----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRANSCRIPTS_DIR="$ROOT/fixtures/transcripts"
AUDIO_DIR="$ROOT/fixtures/audio"
VOICE="${VOICE:-Samantha}"

if [[ "$OSTYPE" != darwin* ]]; then
  echo "error: build-fixtures.sh requires macOS (uses 'say' and 'afconvert')." >&2
  exit 1
fi

if ! command -v say >/dev/null; then
  echo "error: 'say' not found on PATH." >&2
  exit 1
fi
if ! command -v afconvert >/dev/null; then
  echo "error: 'afconvert' not found on PATH." >&2
  exit 1
fi

mkdir -p "$AUDIO_DIR"

if ! ls "$TRANSCRIPTS_DIR"/*.txt >/dev/null 2>&1; then
  echo "error: no transcripts found in $TRANSCRIPTS_DIR" >&2
  exit 1
fi

count=0
for txt in "$TRANSCRIPTS_DIR"/*.txt; do
  name="$(basename "$txt" .txt)"
  wav="$AUDIO_DIR/$name.wav"
  aiff="$AUDIO_DIR/$name.aiff"

  if [[ -f "$wav" ]] && [[ "$wav" -nt "$txt" ]]; then
    echo "  · $name.wav up to date, skipping"
    continue
  fi

  echo "  · synthesizing $name (voice=$VOICE)"
  # `say -o` writes AIFF; afconvert downsamples to 16 kHz mono PCM s16le WAV.
  say -v "$VOICE" -o "$aiff" -f "$txt"
  afconvert \
    -f WAVE \
    -d LEI16@16000 \
    -c 1 \
    "$aiff" "$wav" \
    >/dev/null
  rm -f "$aiff"
  count=$((count + 1))
done

echo "done — generated/updated $count fixture(s) in $AUDIO_DIR"
