#!/usr/bin/env bash
# Build echo-mcp and place it where Tauri expects external binaries.
# Tauri sidecar convention: src-tauri/binaries/<name>-<target_triple>
#
# The target triple is resolved in this order:
#   1. TAURI_ENV_TARGET_TRIPLE env var (set by Tauri build scripts)
#   2. --target <triple> flag passed to this script
#   3. Host triple from `rustc --print host-tuple`
set -euo pipefail

TARGET=""
PROFILE="release"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="$2"; shift 2 ;;
    *) PROFILE="$1"; shift ;;
  esac
done

# Resolve triple: env var > flag > host
TRIPLE="${TAURI_ENV_TARGET_TRIPLE:-${TARGET:-$(rustc --print host-tuple)}}"

CARGO_TARGET_ARGS=""
if [ "$TRIPLE" != "$(rustc --print host-tuple)" ]; then
  CARGO_TARGET_ARGS="--target $TRIPLE"
fi

cargo build --profile "$PROFILE" -p echo-mcp $CARGO_TARGET_ARGS

# Cargo outputs dev profile to target/debug/, not target/dev/
if [ "$PROFILE" = "dev" ]; then
  OUT_DIR="debug"
else
  OUT_DIR="$PROFILE"
fi

# When cross-compiling, cargo puts output under target/<triple>/<profile>/
if [ -n "$CARGO_TARGET_ARGS" ]; then
  SRC="target/${TRIPLE}/${OUT_DIR}/echo-mcp"
else
  SRC="target/${OUT_DIR}/echo-mcp"
fi

DEST="src-tauri/binaries/echo-mcp-${TRIPLE}"

mkdir -p src-tauri/binaries
cp "$SRC" "$DEST"
echo "Copied echo-mcp → $DEST (triple=$TRIPLE)"
