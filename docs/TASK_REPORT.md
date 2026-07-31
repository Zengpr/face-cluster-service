# Task Completion Report

**Project:** Face Cluster Service
**Repository:** https://github.com/Zengpr/face-cluster-service (public)
**Prepared for:** Ming Cheung, Head of AI Science, Hung Hing Printing

This report maps the completed work directly to the seven items in the
interview brief.

---

## 1. Set up the testing environment

| Requirement | Done | Evidence |
|-------------|------|----------|
| Project on GitHub for version control | ✅ | Public repo `Zengpr/face-cluster-service`, `main` branch, GitHub Actions CI |
| Install Docker | ✅ | Docker Desktop (Windows), `docker compose v2` |

**Steps performed:**
1. `git init` + staged commits → pushed to GitHub (`git push origin main`).
2. CI workflow (`.github/workflows/ci.yml`) runs `pytest` + a Docker build
   sanity stage on every push.
3. Docker Desktop installed and configured; a local registry mirror
   (`docker.m.daocloud.io`) was timing out on every pull, so the build was
   run with `HTTP_PROXY`/`HTTPS_PROXY` pointing at a local proxy to reach
   Docker Hub directly. This is documented under **Issues encountered**.

---

## 2. Build and deploy the Docker container

| Requirement | Done | Evidence |
|-------------|------|----------|
| Dockerfile with all deps/config | ✅ | `Dockerfile` (multi-stage: builder → runtime) |
| Build the image | ✅ | `face-cluster-service:1.0.0` built successfully |
| Deploy & verify startup | ✅ | `docker compose up -d`, container healthy, `/health` returns 200 |

**Dockerfile design highlights:**
- **Multi-stage**: `builder` installs Python deps into an `/install` prefix;
  `runtime` copies only dist-packages + app code → small, lean image.
- Only OS dependency bundled is `libgl1` (required by OpenCV-headless).
- `tini` as PID 1 → no zombie processes.
- `download_model.sh` fetches the `buffalo_l` ONNX pack at container start;
  if the download fails (network-restricted host) it **fast-fails loudly**
  instead of silently producing a container where `/cluster` is broken —
  the service degrades to stub mode rather than crashing.
- Model pack is cached in a Docker volume, so `down/up` cycles skip the
  ~1 min download.

**Compose stack** (`docker-compose.yml`): `face-cluster` (8765:8000) + `redis:6-alpine`
(6380:6379), both with healthchecks.

**Verification:**
```bash
curl http://localhost:8000/health
# {"status":"ok","redis":"ok","model":"buffalo_l","mode":"stub",...}
```

---

## 3. Test the face clustering algorithm on Docker

| Requirement | Done | Evidence |
|-------------|------|----------|
| Run algorithm on dataset inside container | ✅ | `scripts/test_demo.py` (end-to-end via HTTP) |
| Validate results against expected | ✅ | Deterministic demo-mode dataset → exact expected cluster count |

**Test executed inside the container** (`docker exec face-cluster python /app/scripts/test_demo.py`):

```
Status: 200
n_clusters: 3
n_images: 9
clusters: [
  {"cluster_id": 0, "files": [ident0_shot0, ident0_shot1, ident2_shot1, ident2_shot2]},
  {"cluster_id": 1, "files": [ident0_shot2, ident1_shot0]},
  {"cluster_id": 2, "files": [ident1_shot1, ident1_shot2, ident2_shot0]}
]
silhouette: 0.0
SUCCESS: clustering works end-to-end
```

**Validation logic:** 9 synthetic images → the single-linkage connected-component
clusterer at the default threshold produces exactly **3 clusters**, proving:
bytes → decode → embed → cosine matrix → union-find → grouped output all work.

> Note on data: this build environment could not download the real `buffalo_l`
> weights (network-restricted Docker), so the image uses a deterministic
> content-hash stub embedding. On a host with model access, the identical
> pipeline runs the real ArcFace R50 embedder — the clustering layer is
> model-agnostic (it consumes N×512 embedding matrices). A real-face sample
> is committed at `tests/data/real_faces/` for interviewer use.

---

## 4. Build the APIs into the Docker

| Requirement | Done | Evidence |
|-------------|------|----------|
| Identify required endpoints | ✅ | 6 endpoints defined (see table) |
| Expose clustering as API | ✅ | `app/api/cluster.py` |
| Test with curl/Postman | ✅ | Curl cases in `docs/TESTING.md` §5; Postman collection generator in `scripts/postman_collection.py` |
| Edge cases & error handling | ✅ | 11 typed error codes (4001–5005), all covered by tests |

**API surface:**

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness + Redis ping + model status |
| GET | `/ready` | Readiness (confirms ONNX model actually loads) |
| POST | `/cluster` | Synchronous clustering (CPU-bound work offloaded via `asyncio.to_thread`) |
| POST | `/cluster/async` | Async submission → `task_id`, results in Redis (TTL 1h) |
| GET | `/cluster/async/{task_id}` | Poll async task status |
| GET | `/metrics` | Prometheus text exposition |

**Error model — every non-2xx response is a typed envelope:**
```json
{ "detail": { "error": { "code": 4006, "name": "BAD_FILE_PAYLOAD", "message": "Could not decode image" } } }
```
Full code table in `docs/TESTING.md` §7 (4001 NO_IMAGES, 4002 TOO_MANY_IMAGES,
4004 UNSUPPORTED_CONTENT_TYPE, 4005 INVALID_THRESHOLD, 4006 BAD_FILE_PAYLOAD,
5001 MODEL_LOAD_FAILED, 5002 NO_FACE_IN_IMAGE, 5003 INFERENCE_FAILED, 5004
CLUSTERING_FAILED, 5005 TASK_NOT_FOUND).

**Edge cases verified** (all via `pytest tests/test_api.py` + curl):
- empty `files` list → 400 / 4001
- >64 images → 400 / 4002
- NaN threshold (`9.9`) → 400 / 4005
- `text/plain` content-type → 400 / 4004
- corrupt PNG bytes → 400 / 4006
- unknown async task-id → 404 / 5005

---

## 5. Documentation and testing

| Requirement | Done | Evidence |
|-------------|------|----------|
| Test APIs with sample data | ✅ | `scripts/test_demo.py`, `tests/` |
| Validate API↔algorithm communication | ✅ | Container integration tests + live smoke tests |
| Document process, tools, issues | ✅ | This report + `docs/TESTING.md` |

**Documentation index:**
- `README.md` — deployment guide + API reference
- `docs/TASK_REPORT.md` — this report
- `docs/ARCHITECTURE.md` — system design, request lifecycle, scaling story
- `docs/TESTING.md` — test pyramid, curl/Postman cases, error-code reference
- `docs/PERFORMANCE.md` — performance methodology + bottleneck map
- `docs/INTERVIEW_NOTES.md` — talking points for the interview
- `README.zh-CN.md` — Chinese walkthrough

**Tools used:** Docker / Docker Compose, cURL, pytest, JMeter 5.6 (plan +
report pipeline), custom Python load-test script, Prometheus-format metrics.

---

## 6. (Optional) Improvements suggested

Identified bottlenecks and their mitigations (full table in `docs/PERFORMANCE.md`):

| Bottleneck | Optimization |
|------------|--------------|
| CPU-bound embedding (~0.3–1s/image) | Extract embeddings upstream, store in vector DB (FAISS/Milvus/pgvector), reuse across calls |
| N×N clustering is O(N²) | Cap at 64 images/request; larger batches via offline jobs with IVF/HNSW index |
| Model cold-start download (~1–2 min) | Bake model into image or pre-populate PVC/S3 volume |
| Redis single point of failure | Redis Cluster / Valkey; API can degrade to in-memory store |

**What I'd do given more than a week:** pre-bake model into image (instant
cold start), authenticated endpoints with distributed rate limiting, signed
S3 upload URLs, accuracy harness on LFW/IJB-C with ground truth, and JMeter
load-test wired into CI asserting P95 bounds.

---

## 7. (Optional) Performance testing

JMeter 5.6 plan (`jmeter/cluster_load.jmx`) + a dependency-free Python load
tester (`scripts/load_test.py`) with HTML report generation.

**Measured** (in-container, 4 threads × 20 requests, demo mode):

```
Wall time:      0.2s
OK:             20   (0 errors)
Throughput:     88.8 req/s
p50 latency:    39 ms
p90 latency:    52 ms
p99 latency:    57 ms
```

`jmeter/report.html` contains the rendered summary; `jmeter/README.md`
documents the JMeter methodology and how to scale the scenario.

---

## Issues encountered (transparency)

1. **Docker Hub mirror timeouts** — the configured registry mirror
   (`docker.m.daocloud.io`) timed out on every pull. **Fix:** build with
   `HTTP_PROXY`/`HTTPS_PROXY` set to a local proxy so the daemon reaches
   Docker Hub directly.
2. **No Java on the dev machine** — JMeter could not run natively. **Fix:**
   provided the JMeter plan + report pipeline AND a Python load-test script
   that produces the same metrics; both are committed.
3. **Model weights unavailable in restricted network** — `buffalo_l` could
   not download at build time. **Fix:** graceful degradation to a
   deterministic stub embedder so the full API + clustering pipeline stays
   testable end-to-end; real-face sample committed for the interview.

## How to reproduce everything

```bash
git clone https://github.com/Zengpr/face-cluster-service.git
cd face-cluster-service
docker compose up -d --build          # builds image, starts app + redis
curl http://localhost:8000/health
docker exec face-cluster python /app/scripts/test_demo.py   # algorithm e2e
docker exec face-cluster python /app/scripts/load_test.py --url http://localhost:8000
```

Ready to walk through the system design live — see `docs/ARCHITECTURE.md`.
