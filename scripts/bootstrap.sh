#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# EchoNote — developer bootstrap.
#
# Verifies required toolchains and wires up versioned git hooks. Does not
# install anything automatically; prints actionable commands when something
# is missing. Safe to re-run on every checkout.
#
# Supported hosts: macOS (primary), Linux. Windows devs should use WSL2.
# -----------------------------------------------------------------------------
set -Eeuo pipefail

# --- cosmetics ---------------------------------------------------------------
if [[ -t 1 ]]; then
  RED=$'\033[31m'; GRN=$'\033[32m'; YLW=$'\033[33m'; BLU=$'\033[34m'
  BLD=$'\033[1m'; DIM=$'\033[2m'; RST=$'\033[0m'
else
  RED=""; GRN=""; YLW=""; BLU=""; BLD=""; DIM=""; RST=""
fi

section() { printf "\n%s==>%s %s%s%s\n" "$BLU" "$RST" "$BLD" "$1" "$RST"; }
ok()      { printf "  %s✓%s %s\n" "$GRN" "$RST" "$1"; }
warn()    { printf "  %s!%s %s\n" "$YLW" "$RST" "$1"; }
fail()    { printf "  %s✗%s %s\n" "$RED" "$RST" "$1"; }

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ERRORS=0
record_error() { ERRORS=$((ERRORS + 1)); }

# --- 1. host check -----------------------------------------------------------
section "Host detection"
OS="$(uname -s)"
case "$OS" in
  Darwin) ok "macOS ($(uname -m))" ;;
  Linux)  ok "Linux ($(uname -m))" ;;
  MINGW*|MSYS*|CYGWIN*)
    fail "Native Windows is not supported. Use WSL2 (Ubuntu 22.04+)."
    exit 1
    ;;
  *)
    fail "Unsupported host: $OS"
    exit 1
    ;;
esac

# --- 2. required tools -------------------------------------------------------
section "Required tooling"

require() {
  # require <bin> <humanName> <installHint>
  local bin="$1" name="$2" hint="$3"
  if command -v "$bin" >/dev/null 2>&1; then
    ok "$name: $($bin --version 2>/dev/null | head -n1)"
  else
    fail "$name not found. Install with: ${DIM}$hint${RST}"
    record_error
  fi
}

require git "git"  "xcode-select --install  # or: brew install git"
require cargo "cargo" "brew install rust  # or: https://rustup.rs"
require rustc "rustc" "brew install rust  # or: https://rustup.rs"

# rust-toolchain.toml pin
EXPECTED_RUST="$(grep -E '^channel' rust-toolchain.toml | sed -E 's/.*"([^"]+)".*/\1/')"
if command -v rustc >/dev/null 2>&1; then
  ACTUAL_RUST="$(rustc --version | awk '{print $2}')"
  if [[ "$ACTUAL_RUST" == "$EXPECTED_RUST" ]]; then
    ok "Rust pin matches ($EXPECTED_RUST)"
  else
    warn "Rust pin mismatch: toolchain=$EXPECTED_RUST, local=$ACTUAL_RUST"
    warn "  CI uses $EXPECTED_RUST via rust-toolchain.toml; local drift is OK short-term."
  fi
fi

# --- 3. rustup components ----------------------------------------------------
section "Rust components"
check_component() {
  local c="$1"
  if cargo "$c" --version >/dev/null 2>&1; then
    ok "cargo $c available"
  else
    fail "cargo $c missing. Install with: rustup component add $c"
    record_error
  fi
}
check_component fmt
check_component clippy

# --- 4. frontend tooling -----------------------------------------------------
section "Frontend tooling"
FRONTEND_ERRORS=0
if command -v node >/dev/null 2>&1; then
  ok "node: $(node --version)"
else
  fail "node not found. Install with: brew install node@20  # or use nvm"
  FRONTEND_ERRORS=$((FRONTEND_ERRORS + 1))
fi

if command -v pnpm >/dev/null 2>&1; then
  ok "pnpm: $(pnpm --version)"
elif command -v corepack >/dev/null 2>&1; then
  warn "pnpm not activated. Run: corepack enable && corepack prepare pnpm@10 --activate"
  FRONTEND_ERRORS=$((FRONTEND_ERRORS + 1))
else
  fail "pnpm not found. Enable via corepack once node is installed."
  FRONTEND_ERRORS=$((FRONTEND_ERRORS + 1))
fi

if [[ $FRONTEND_ERRORS -eq 0 && -f "package.json" ]]; then
  if [[ -d "node_modules" ]]; then
    ok "node_modules present (run 'pnpm install' after every pnpm-lock.yaml change)"
  else
    warn "node_modules missing. Run: pnpm install"
  fi
fi
record_error_count=$FRONTEND_ERRORS

# --- 5. git hooks ------------------------------------------------------------
section "Git hooks"
HOOKS_DIR=".githooks"
if [[ -d "$HOOKS_DIR" ]]; then
  git config core.hooksPath "$HOOKS_DIR"
  chmod +x "$HOOKS_DIR"/* 2>/dev/null || true
  ok "core.hooksPath -> $HOOKS_DIR"
  for h in pre-commit commit-msg; do
    if [[ -x "$HOOKS_DIR/$h" ]]; then
      ok "hook executable: $h"
    else
      warn "hook missing or not executable: $HOOKS_DIR/$h"
    fi
  done
else
  fail "$HOOKS_DIR directory not found"
  record_error
fi

# --- 6. workspace sanity -----------------------------------------------------
section "Workspace sanity"
if cargo metadata --no-deps --format-version=1 >/dev/null 2>&1; then
  MEMBERS=$(cargo metadata --no-deps --format-version=1 \
            | python3 -c 'import sys,json;print(len(json.load(sys.stdin)["packages"]))')
  ok "cargo workspace resolves ($MEMBERS member crates)"
else
  fail "cargo metadata failed. Run 'cargo check --workspace' manually to diagnose."
  record_error
fi

# --- summary -----------------------------------------------------------------
section "Summary"
if [[ $ERRORS -eq 0 ]]; then
  printf "%s✓ Environment ready.%s Next: %scargo build --workspace%s\n" \
    "$GRN" "$RST" "$BLD" "$RST"
  exit 0
else
  printf "%s✗ %d issue(s) above must be fixed before you can build.%s\n" \
    "$RED" "$ERRORS" "$RST"
  exit 1
fi
