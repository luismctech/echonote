#!/usr/bin/env bash
# Build echo-mcp and place it where Tauri expects external binaries.
# Tauri sidecar convention: src-tauri/binaries/<name>-<target_triple>
set -euo pipefail

TRIPLE=$(rustc --print host-tuple)
PROFILE="${1:-release}"

cargo build --profile "$PROFILE" -p echo-mcp

SRC="target/${PROFILE}/echo-mcp"
DEST="src-tauri/binaries/echo-mcp-${TRIPLE}"

mkdir -p src-tauri/binaries
cp "$SRC" "$DEST"
echo "Copied echo-mcp → $DEST"
