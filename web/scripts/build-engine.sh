#!/usr/bin/env bash
# Builds kepler-wasm and writes the browser package into src/lib/engine/pkg (committed).
# Vercel has no Rust toolchain, so this runs on a developer machine, not in the site build.
set -euo pipefail
export PATH="$HOME/.cargo/bin:$PATH"

WEB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$WEB/.." && pwd)"
OUT="$WEB/src/lib/engine/pkg"

cd "$REPO/engine"
# Keep machine-specific paths out of the committed binary (panic locations embed them).
# RUSTFLAGS replaces config rustflags, and the repo sets none (no engine/.cargo/config.toml).
export RUSTFLAGS="--remap-path-prefix=$HOME/.cargo/registry/src=registry --remap-path-prefix=$REPO=repo --remap-path-prefix=$HOME=home"
cargo build -p kepler-wasm --release --target wasm32-unknown-unknown -j6

TARGET_DIR="$(cargo metadata --format-version 1 --no-deps | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')"
WASM="$TARGET_DIR/wasm32-unknown-unknown/release/kepler_wasm.wasm"

rm -rf "$OUT"
mkdir -p "$OUT"
wasm-bindgen --target web --out-dir "$OUT" "$WASM"

# Last commit that touched the engine sources, plus a marker if they have uncommitted changes.
SRC=(crates Cargo.toml Cargo.lock rust-toolchain.toml)
STAMP="$(git log -1 --format=%H -- "${SRC[@]}")"
if [ -n "$(git status --porcelain -- "${SRC[@]}")" ]; then STAMP="$STAMP-dirty"; fi
echo "$STAMP" > "$OUT/ENGINE_COMMIT"

if grep -aq "/ho""me/" "$OUT/kepler_wasm_bg.wasm"; then
  echo "warning: a machine path is still embedded in the wasm" >&2
fi
echo "engine package written to $OUT (commit $STAMP)"
