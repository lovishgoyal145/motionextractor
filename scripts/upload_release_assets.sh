#!/usr/bin/env bash
set -e

PROJECT_DIR="/home/lovish/.gemini/antigravity/scratch/motion_extraction"
cd "$PROJECT_DIR"

PAT=$(grep -E "^GITHUB_PAT=" .env | cut -d '=' -f2- | tr -d '\r\n"')
RELEASE_ID="401725767"

upload_asset() {
    FILE="$1"
    FILENAME=$(basename "$FILE")
    FILESIZE=$(stat -c%s "$FILE")
    echo "=================================================="
    echo "Uploading $FILENAME ($(( FILESIZE / 1048576 )) MB)..."
    echo "=================================================="
    
    # Delete if existing incomplete upload
    EXISTING_ID=$(curl -s -H "Authorization: Bearer $PAT" "https://api.github.com/repos/lovishgoyal145/motionextractor/releases/$RELEASE_ID" | grep -B 2 "$FILENAME" | grep "\"id\":" | head -n 1 | tr -dc '0-9' || true)
    if [ -n "$EXISTING_ID" ]; then
        echo "Cleaning up prior asset ID $EXISTING_ID..."
        curl -s -X DELETE -H "Authorization: Bearer $PAT" "https://api.github.com/repos/lovishgoyal145/motionextractor/releases/assets/$EXISTING_ID"
    fi
    
    curl -# -f -X POST \
        -H "Authorization: Bearer $PAT" \
        -H "Content-Type: application/octet-stream" \
        --data-binary @"$FILE" \
        "https://uploads.github.com/repos/lovishgoyal145/motionextractor/releases/$RELEASE_ID/assets?name=$FILENAME"
    echo ""
    echo "Successfully uploaded $FILENAME!"
}

upload_asset "models/dwpose/dw-ll_ucoco_384.onnx"
upload_asset "models/dwpose/yolox_l.onnx"

echo "All model weights successfully published to GitHub Releases!"
