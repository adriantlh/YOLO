#!/usr/bin/env bash
set -euo pipefail

# Request annotated image and save to serve/out.png

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
cd "$REPO_ROOT"

URL="${URL:-http://localhost:8000}"
IMG="${IMG:-demo/images/inference/image.png}"
OUT="${OUT:-serve/out.png}"

if [[ ! -f "$IMG" ]]; then
  echo "Image not found at $IMG" >&2
  exit 1
fi

echo "POST $URL/predict?visualize=true with $IMG -> $OUT"
curl -s -o "$OUT" -F "file=@${IMG}" "$URL/predict?visualize=true"
echo "Saved: $OUT"

