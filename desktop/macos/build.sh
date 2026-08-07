#!/bin/bash
# Build Flwr.app — a native macOS desktop wrapper for flwr, using only Apple's
# system frameworks (compiled with the built-in swiftc). No packages, no crates.
#
#   ./build.sh [output.app]      # default: ~/Applications/Flwr.app
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
app="${1:-$HOME/Applications/Flwr.app}"

command -v swiftc >/dev/null || { echo "error: swiftc not found (install Xcode Command Line Tools: xcode-select --install)"; exit 1; }

echo "building $app ..."
rm -rf "$app"
mkdir -p "$app/Contents/MacOS"

swiftc -O \
  -o "$app/Contents/MacOS/flwr-desktop" \
  "$here/flwr_desktop.swift" \
  -framework AppKit -framework WebKit -framework Foundation

cat > "$app/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Flwr</string>
  <key>CFBundleDisplayName</key><string>flwr</string>
  <key>CFBundleIdentifier</key><string>dev.hos.flwr</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>flwr-desktop</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSPrincipalClass</key><string>NSApplication</string>
</dict>
</plist>
PLIST

echo "✓ built $app"
echo "  launch:  open \"$app\""
echo "  config:  FLWR_MODEL, FLWR_PORT, FLWR_BIN environment variables"
