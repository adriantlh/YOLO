#!/usr/bin/env bash
set -euo pipefail

# Start FastAPI (uvicorn) for YOLO server.
# Uses env vars if provided: YOLO_MODEL_CFG, YOLO_WEIGHT_PATH, YOLO_DATA_YAML

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
cd "$REPO_ROOT"

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

# Default configs in serve/config if not explicitly set
export YOLO_MODEL_CFG="${YOLO_MODEL_CFG:-serve/config/model.yaml}"
if [[ -f "serve/config/data.yaml" && -z "${YOLO_DATA_YAML:-}" ]]; then
  export YOLO_DATA_YAML="serve/config/data.yaml"
fi
if [[ -f "serve/config/Defect Types.txt" && -z "${YOLO_DEFECT_TYPES:-}" ]]; then
  export YOLO_DEFECT_TYPES="serve/config/Defect Types.txt"
fi

# Auto-pick first weights file in serve/weights if available
if [[ -z "${YOLO_WEIGHT_PATH:-}" && -d "serve/weights" ]]; then
  first_weight=$(ls serve/weights/*.pt 2>/dev/null | head -n1 || true)
  if [[ -n "$first_weight" ]]; then
    export YOLO_WEIGHT_PATH="$first_weight"
  fi
fi

echo "Starting server at http://${HOST}:${PORT} (Ctrl+C to stop)"
echo "Using MODEL_CFG=$YOLO_MODEL_CFG"
[[ -n "${YOLO_WEIGHT_PATH:-}" ]] && echo "Using WEIGHTS=$YOLO_WEIGHT_PATH" || echo "No weights configured"
[[ -n "${YOLO_DATA_YAML:-}" ]] && echo "Using DATA=$YOLO_DATA_YAML"
[[ -n "${YOLO_DEFECT_TYPES:-}" ]] && echo "Using DEFECT_TYPES=$YOLO_DEFECT_TYPES"

exec uvicorn api.main:app --host "$HOST" --port "$PORT" --reload
