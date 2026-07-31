#!/usr/bin/env bash
# Start the API server, download model in background
set -eu
nohup /usr/local/bin/download_model.sh >/tmp/model_dl.log 2>&1 &
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
