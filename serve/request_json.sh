#!/usr/bin/env bash
set -euo pipefail

# Send a sample image to the /predict endpoint and print JSON.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
cd "$REPO_ROOT"

URL="${URL:-http://localhost:8000}"
IMG="${IMG:-demo/images/inference/image.png}"

if [[ ! -f "$IMG" ]]; then
  echo "Image not found at $IMG" >&2
  exit 1
fi

echo "POST $URL/predict with $IMG"
curl -s -F "file=@${IMG}" "$URL/predict" || true
echo

