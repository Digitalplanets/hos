#!/bin/sh
# flwr installer. Installs the `hos` engine and the `flwr` app.
#
#   curl -fsSL https://www.flwr.systems/install.sh | sh
#
# Fast path: downloads a prebuilt binary for your platform.
# Fallback:  installs the Rust toolchain if needed, then builds from source.
# Override the install dir with FLWR_BIN_DIR (default: ~/.local/bin).
set -e

REPO="Digitalplanets/hos"
BINDIR="${FLWR_BIN_DIR:-$HOME/.local/bin}"
OS="$(uname -s)"
ARCH="$(uname -m)"
installed=""

say() { printf '\033[38;2;206;142;168mflwr\033[0m  %s\n' "$1"; }

mkdir -p "$BINDIR"

asset=""
case "$OS-$ARCH" in
  Darwin-arm64)  asset="flwr-macos-arm64.tar.gz" ;;
  Darwin-x86_64) asset="flwr-macos-x86_64.tar.gz" ;;
esac

if [ -n "$asset" ]; then
  url="https://github.com/$REPO/releases/latest/download/$asset"
  say "downloading prebuilt binaries for $OS $ARCH ..."
  if curl -fsSL "$url" 2>/dev/null | tar xz -C "$BINDIR" 2>/dev/null; then
    chmod +x "$BINDIR/hos" "$BINDIR/flwr" 2>/dev/null || true
    installed=1
    say "installed hos + flwr to $BINDIR"
  fi
fi

if [ -z "$installed" ]; then
  say "building from source for $OS-$ARCH ..."
  if ! command -v cargo >/dev/null 2>&1; then
    say "installing the Rust toolchain (rustup) ..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    . "$HOME/.cargo/env"
  fi
  cargo install --git "https://github.com/$REPO" --bins
  BINDIR="$HOME/.cargo/bin"
  say "installed hos + flwr to $BINDIR"
fi

case ":$PATH:" in
  *":$BINDIR:"*) : ;;
  *) say "add this to your shell profile:  export PATH=\"$BINDIR:\$PATH\"" ;;
esac

say "done. try it:"
say "  flwr pull flwr-bloom   &&   flwr run flwr-bloom"
