#!/usr/bin/env bash
# Pre-download buffalo_l model pack if not already present.
# Non-blocking — if the download fails (behind a firewall, slow network),
# the app starts in stub mode (see app/services/face_embedder.py).
set -eo pipefail

MODEL_ROOT="${INSIGHTFACE_HOME:-/root/.insightface}"
PACK_NAME="buffalo_l"
URL="${INSIGHTFACE_MODEL_URL:-https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip}"

DEST="${MODEL_ROOT}/models/${PACK_NAME}"

if [ -f "${DEST}/w600k_r50.onnx" ] || [ -f "${DEST}/det_10g.onnx" ]; then
  echo "[model] ${PACK_NAME} already present, skip download"
else
  echo "[model] trying download (best-effort, never blocks boot)"
  mkdir -p "${MODEL_ROOT}/models"
  # Fast attempt — 30s connect, 180s max, or we give up.
  if curl -fsSL --connect-timeout 15 --max-time 180 \
    -o "${MODEL_ROOT}/models/${PACK_NAME}.zip" "${URL}" 2>/dev/null; then
    if unzip -o "${MODEL_ROOT}/models/${PACK_NAME}.zip" \
      -d "${MODEL_ROOT}/models" >/dev/null 2>&1; then
      rm -f "${MODEL_ROOT}/models/${PACK_NAME}.zip"
      echo "[model] downloaded and extracted"
    else
      echo "[model] unzip failed, will use stub"
    fi
  else
    echo "[model] download skipped (network restricted), stub mode active"
    rm -f "${MODEL_ROOT}/models/${PACK_NAME}.zip" 2>/dev/null
  fi
fi

exec "$@"
