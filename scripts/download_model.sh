#!/usr/bin/env bash
# Pre-download buffalo_l model pack if not already present, then exec CMD.
# We bundle the download here so ``docker run`` can use a clean image without a
# separate model layer (good for the interview demo; prod would mount a volume).
set -eo pipefail

MODEL_ROOT="${INSIGHTFACE_HOME:-/root/.insightface}"
PACK_NAME="buffalo_l"
ZIP_NAME="insightface_model_pack_buffalo_l.zip"
URL="${INSIGHTFACE_MODEL_URL:-https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip}"

DEST="${MODEL_ROOT}/models/${PACK_NAME}"
ZIP_DEST="${MODEL_ROOT}/models/${ZIP_NAME}"

if [ -f "${DEST}/w600k_r50.onnx" ] || [ -f "${DEST}/det_10g.onnx" ]; then
  echo "[model] ${PACK_NAME} already present, skip download"
else
  echo "[model] downloading ${PACK_NAME} from ${URL}"
  mkdir -p "${MODEL_ROOT}/models"
  curl -fsSL -o "${ZIP_DEST}" "${URL}" || {
    echo "[model] download failed — service will start in degraded mode"; exec "$@"
  }
  # InsightFace stores the zip with a top-level folder named "buffalo_l".
  # The python code expects an folder layout under models/<name>.
  if unzip -o "${ZIP_DEST}" -d "${MODEL_ROOT}/models" >/dev/null 2>&1; then
    rm -f "${ZIP_DEST}"
  else
    echo "[model] unzip failed — keeping zip at ${ZIP_DEST}"; exec "$@"
  fi
fi

echo "[model] pack ready"
exec "$@"
