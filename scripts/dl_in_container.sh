#!/bin/bash
set -e
mkdir -p /root/.insightface/models
cd /root/.insightface/models

URLS=(
  "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
  "https://ghfast.top/https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
)

for url in "${URLS[@]}"; do
  echo "trying: $url"
  if curl -fsSL --connect-timeout 10 --max-time 300 -o bl.zip "$url" 2>/dev/null; then
    sz=$(wc -c < bl.zip)
    echo "SUCCESS: $sz bytes"
    unzip -qo bl.zip
    rm -f bl.zip
    exit 0
  fi
  rm -f bl.zip
  echo "FAILED -> next"
done

echo "ALL_DOWNLOAD_SOURCES_FAILED"
exit 1
