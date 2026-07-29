# Testing

## Test pyramid

```
                ┌──────────────┐
                │  Integration │   ← test_integration_live.py (live HTTP)
                └──────┬───────┘
                       │
                ┌──────┴───────┐
                │  Contract API│   ← tests/test_api.py (TestClient)
                └──────┬───────┘
                       │
                ┌──────┴───────┐
                │  Unit (math) │   ← tests/test_clusterer.py
                └──────────────┘
```

## 1. Unit — `tests/test_clusterer.py`

Pure-Numpy, no model. Asserts:
- two clearly-separated identity groups produce 2 clusters
- DBSCAN flags lone points as `-1`
- empty embedding matrix raises `ServiceError`
- invalid backend raises `ServiceError`
- label compression is deterministic across runs

Run:

```bash
pytest tests/test_clusterer.py -v
```

## 2. Contract API — `tests/test_api.py`

Uses `TestClient` against the real FastAPI app. The ArcFace model is
**stubbed** with a deterministic RNG so the suite stays fast (sub-second)
and offline. Covered cases:

| Case | Expected |
|---|---|
| `GET /health` | 200 with status key |
| `POST /cluster` no files | 400, `error.code=4001` |
| `POST /cluster` 3 files over budget | 400, `error.code=4002` |
| `POST /cluster` NaN threshold (`9.9`) | 400, `error.code=4005` |
| `POST /cluster` text/plain content-type | 400, `error.code=4004` |
| `POST /cluster` corrupt PNG bytes | 400, `error.code=4006` |
| `POST /cluster` valid 3 PNGs | 200, `n_clusters ≥ 1` |
| `POST /cluster/async` submit | 200 with `task_id` |
| `GET  /cluster/async/{task_id}` | 200 with `state` field |
| `GET  /cluster/async/unknown_id` | 404, `error.code=5005` |
| `GET /metrics` | 200 text, contains `fc_requests_total` |

Run:

```bash
pytest tests/test_api.py -v
```

## 3. Container integration — `tests/test_integration_live.py`

Black-box HTTP test against the container stack you started with
`docker compose up`.

The default sample dataset is **synthetic placeholder images** (no real
faces) so the algorithm will respond with **HTTP 422 NO_FACE** — that's
the success criteria for the smoke test, because it proves:

- image bytes flow through full HTTP stack
- model loaded successfully inside the container
-OpenCV decoded the PNG and produced a numpy array
- InsightFace detector ran without crash
- error-code middleware surfaced the typed 422 response

For a clustering accuracy demo swap in real-face images and assert
`status_code == 200` and the expected `n_clusters` count. See
`docs/INTERVIEW_NOTES.md` for the recommended LFW-style dataset.

Run:

```bash
BASE=http://localhost:8000 python tests/test_integration_live.py
```

## 4. JMeter performance — `jmeter/`

Methodology and expected metrics ranges are documented in
**docs/PERFORMANCE.md**.

## 5. Manual verification with curl & Postman

### Health
```bash
curl -s http://localhost:8000/health | jq
```

### Happy path
```bash
curl -s -X POST http://localhost:8000/cluster \
  -F "files=@tests/data/images/ident0_shot0.png" \
  -F "files=@tests/data/images/ident0_shot1.png" \
  -F "files=@tests/data/images/ident1_shot0.png" \
  -F "threshold=0.5" | jq
```

### Bad input — empty files list
```bash
curl -s -X POST http://localhost:8000/cluster | jq
# {"detail":{"error":{"code":4001,"name":"NO_IMAGES","message":"no files supplied"}}}
```

### Bad input — corrupted PNG
```bash
echo "not a png" > bad.png
curl -s -X POST http://localhost:8000/cluster \
  -F "files=@bad.png;type=image/png" | jq
# {"detail":{"error":{"code":4006,"name":"BAD_FILE_PAYLOAD", ...}}}
```

### Bad input — unsupported content-type
```bash
curl -s -X POST http://localhost:8000/cluster \
  -F "files=@README.md;type=text/markdown" | jq
# {"detail":{"error":{"code":4004,...}}}
```

### NaN threshold
```bash
curl -s -X POST http://localhost:8000/cluster \
  -F "files=@tests/data/images/ident0_shot0.png" \
  -F "threshold=9.9" | jq
# {"detail":{"error":{"code":4005,...}}}
```

### Async job
```bash
TID=$(curl -s -X POST http://localhost:8000/cluster/async \
  -F "files=@tests/data/images/ident0_shot0.png" \
  -F "files=@tests/data/images/ident0_shot1.png" \
  | jq -r .task_id)
echo "task_id=$TID"

# poll manually:
curl -s http://localhost:8000/cluster/async/$TID | jq
```

## 6. Postman collection

Import `postman/face-cluster.postman_collection.json` (placeholder file
described in INTERVIEW_NOTES — see `scripts/postman_collection.py`).
Suggested collections:

1. **Smoke** — health + 1 happy-path request
2. **Edge cases** — the five error-code curls above, with response-time
   assertions ≤2s for `400`s
3. **Stress** — JMeter scenario as Postman runner (10 iterations, 5s delay)

## 7. Error code reference

| Code | Name | HTTP | Cause |
|------|------|------| ------------------------------------------------------|
| 4001 | NO_IMAGES | 400 | Empty `files` array |
| 4002 | TOO_MANY_IMAGES | 400 | `len(files) > MAX_IMAGES_PER_REQUEST` |
| 4003 | IMAGE_TOO_LARGE | 400 | `len(bytes) > MAX_IMAGE_BYTES` |
| 4004 | UNSUPPORTED_CONTENT_TYPE | 400 | Not in `allowed_content_types` |
| 4005 | INVALID_THRESHOLD | 400 | threshold outside `[0,2]` |
| 4006 | BAD_FILE_PAYLOAD | 400 | `cv2.imdecode` returned None |
| 5001 | MODEL_LOAD_FAILED | 503 | insightface import / onnx load error |
| 5002 | NO_FACE_IN_IMAGE | 422 | detector returned zero faces |
| 5003 | INFERENCE_FAILED | 500 | Runtime exception during embedding |
| 5004 | CLUSTERING_FAILED | 500 | Clusterer raised unexpected exception |
| 5005 | TASK_NOT_FOUND | 404 | Async task-id unknown / expired |
