#!/usr/bin/env bash
set -euo pipefail

# Full end-to-end demo: generate synthetic images, send to API, show clustering
SERVICE_URL="${1:-http://localhost:8765}"

echo "=== Face Cluster Service Demo ==="
echo "Service: $SERVICE_URL"
echo ""

# 1. Health check
echo "--- Health ---"
curl -s "$SERVICE_URL/health" | python -m json.tool
echo ""

# 2. Generate synthetic test images (3 identities x 3 shots)
echo "--- Generating 9 synthetic images (3 identities) ---"
OUTDIR=$(mktemp -d)
python -c "
import numpy as np, cv2, os
out = '$OUTDIR'
for ident in range(3):
    for shot in range(3):
        img = np.ones((200,200,3), dtype=np.uint8) * 220
        cv2.putText(img, f'ID{ident}S{shot}', (30,110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,50,50), 2)
        cv2.imwrite(os.path.join(out, f'ident{ident}_shot{shot}.jpg'), img)
print('Generated 9 images in', out)
"
echo ""

# 3. Clustering call with default settings
echo "--- POST /cluster (default threshold=0.6) ---"
FILES=""
for f in "$OUTDIR"/*.jpg; do
    FILES="$FILES -F files=@$f"
done
# shellcheck disable=SC2086
curl -s -X POST "$SERVICE_URL/cluster" $FILES \
  -F "demo_mode=true" \
  | python -m json.tool
echo ""

# 4. Clustering with higher threshold (should split more)
echo "--- POST /cluster (threshold=0.8, backend=dbscan) ---"
# shellcheck disable=SC2086
curl -s -X POST "$SERVICE_URL/cluster" $FILES \
  -F "threshold=0.8" \
  -F "backend=dbscan" \
  -F "demo_mode=true" \
  | python -m json.tool
echo ""

# Cleanup
rm -rf "$OUTDIR"

echo "=== Demo Complete ==="
