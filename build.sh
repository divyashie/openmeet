#!/bin/bash
set -e
set -o pipefail

echo "=== OpenMeet Build Script ==="
echo ""

# Ensure we're in the project root
cd "$(dirname "$0")"

# Step 1: Activate venv
echo "[1/5] Activating virtual environment..."
source venv/bin/activate

# Step 2: Check PyInstaller
echo "[2/5] Checking PyInstaller..."
pip install pyinstaller --quiet

# Step 3: Generate .icns icon from .png
echo "[3/5] Generating app icon..."
mkdir -p build/icon.iconset
sips -z 16 16     src/assets/openmeet.png --out build/icon.iconset/icon_16x16.png      2>/dev/null
sips -z 32 32     src/assets/openmeet.png --out build/icon.iconset/icon_16x16@2x.png   2>/dev/null
sips -z 32 32     src/assets/openmeet.png --out build/icon.iconset/icon_32x32.png      2>/dev/null
sips -z 64 64     src/assets/openmeet.png --out build/icon.iconset/icon_32x32@2x.png   2>/dev/null
sips -z 128 128   src/assets/openmeet.png --out build/icon.iconset/icon_128x128.png    2>/dev/null
sips -z 256 256   src/assets/openmeet.png --out build/icon.iconset/icon_128x128@2x.png 2>/dev/null
sips -z 256 256   src/assets/openmeet.png --out build/icon.iconset/icon_256x256.png    2>/dev/null
sips -z 512 512   src/assets/openmeet.png --out build/icon.iconset/icon_256x256@2x.png 2>/dev/null
sips -z 512 512   src/assets/openmeet.png --out build/icon.iconset/icon_512x512.png    2>/dev/null
sips -z 1024 1024 src/assets/openmeet.png --out build/icon.iconset/icon_512x512@2x.png 2>/dev/null
iconutil -c icns build/icon.iconset -o src/assets/openmeet.icns
echo "  Icon generated: src/assets/openmeet.icns"

# Step 4: Clean previous builds
echo "[4/5] Cleaning previous builds..."
rm -rf dist/OpenMeet dist/OpenMeet.app

# Step 5: Build the .app
echo "[5/5] Building OpenMeet.app (this may take several minutes)..."
pyinstaller OpenMeet.spec --noconfirm 2>&1 | tail -10

echo ""
echo "=== Build Verification ==="
echo "App bundle: dist/OpenMeet.app"
du -sh dist/OpenMeet.app
echo ""

# Check critical bundled files (PyInstaller puts them in Contents/Frameworks/)
echo "Checking critical files..."
FW_DIR="dist/OpenMeet.app/Contents/Frameworks"
FAIL_COUNT=0

if test -f "$FW_DIR/whisper.cpp/build/bin/whisper-cli"; then
    echo "  [OK] whisper-cli binary"
else
    echo "  [FAIL] whisper-cli binary"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if test -f "$FW_DIR/whisper.cpp/models/ggml-base.bin"; then
    echo "  [OK] whisper base model"
else
    echo "  [FAIL] whisper base model"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if test -f "$FW_DIR/assets/openmeet_menu.png"; then
    echo "  [OK] menu bar icon"
else
    echo "  [FAIL] menu bar icon"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if test -f "$FW_DIR/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"; then
    echo "  [OK] LLM model"
else
    echo "  [FAIL] LLM model"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if [ $FAIL_COUNT -gt 0 ]; then
    echo "ERROR: $FAIL_COUNT critical file(s) missing. Build verification failed."
    exit 1
fi
echo ""
echo "=== Build Complete ==="
echo "To run:     open dist/OpenMeet.app"
echo "To install: cp -r dist/OpenMeet.app /Applications/"
