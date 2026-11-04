#!/bin/bash
# Builds "Edit Pairs.app" in this folder (needs Xcode or the Swift command line tools).
#   ./make_app.sh            then double-click "Edit Pairs.app" (or move it to /Applications)
set -euo pipefail
cd "$(dirname "$0")"
swift build -c release
APP="Edit Pairs.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"
cp "$(swift build -c release --show-bin-path)/EditPairs" "$APP/Contents/MacOS/EditPairs"
cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>Edit Pairs</string>
    <key>CFBundleDisplayName</key><string>Edit Pairs</string>
    <key>CFBundleIdentifier</key><string>local.mosstrainer.editpairs</string>
    <key>CFBundleExecutable</key><string>EditPairs</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>1.0</string>
    <key>CFBundleVersion</key><string>1</string>
    <key>LSMinimumSystemVersion</key><string>13.0</string>
    <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST
codesign --force --sign - "$APP" >/dev/null 2>&1 || true  # ad-hoc, so it opens without a "damaged" warning
echo "Built $(pwd)/$APP"
