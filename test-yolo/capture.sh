#!/usr/bin/env bash
# Capture a still from the RPi 5 camera over SSH and save it next to this script.
# Usage: ./capture.sh [camera_index]   (default 0)
set -euo pipefail

HOST="${RPI_HOST:-rpi}"
CAM="${1:-0}"
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$DIR/capture_cam${CAM}_$(date +%Y%m%d_%H%M%S).jpg"
REMOTE_TMP="/tmp/capture_$$.jpg"

echo "Connecting to $HOST..."
ssh -o ConnectTimeout=10 "$HOST" true || { echo "Cannot reach $HOST" >&2; exit 1; }

# rpicam-* on Bookworm, libcamera-* on older images
TOOL=$(ssh "$HOST" 'command -v rpicam-still || command -v libcamera-still' || true)
[ -n "$TOOL" ] || { echo "No rpicam-still/libcamera-still on $HOST" >&2; exit 1; }
LIST="${TOOL%-still}-hello"

echo "Checking cameras..."
CAMS=$(ssh "$HOST" "$LIST --list-cameras 2>&1")
echo "$CAMS"
if ! grep -qE "^\s*$CAM : " <<<"$CAMS"; then
    echo "Camera $CAM not detected" >&2; exit 1
fi

echo "Capturing from camera $CAM..."
ssh "$HOST" "$TOOL --camera $CAM -n -t 1000 -o $REMOTE_TMP"
scp -q "$HOST:$REMOTE_TMP" "$OUT"
ssh "$HOST" "rm -f $REMOTE_TMP"

echo "Saved $OUT"
