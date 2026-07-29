# Interview talking-points (Hung Hing — Head of AI Science)

Ming's brief covers ten topical areas. Each section maps back to a code
artifact you can show while explaining.

## 1. Easter-egg framing of the task

This is an **AI Engineering** brief, not an algorithm brief. The talking
head should be:

> "I picked the ArcFace 512-d embedding + single-linkage recipe because
> it's the production shape: any identity-class task (face / voice /
> doc / agent-embedding) reduces to (a) extract embedding, (b) build
> similarity graph at threshold, (c) cluster transitively. The Docker +
> API + observability + load-test work is the bulk of this codebase,
> and that's what changes when you ship this to scale."

## 2. Repo / GitHub — `/` ✓

- `gh repo create` done — see `git remote -v`
- Protected branch defaults (main), GitHub Actions CI for both unit
  tests and a docker-build sanity stage

## 3. Dockerfile — `.github/workflows + download_model.sh`

Point out **multi-stage** build:
- `builder` stage installs deps into `/install` prefix
- `runtime` stage copies only Python dist-packages + app code
- `libgl1` is the only OS-level dep bundled (OpenCV-headless needs it)
- `tini` is the entrypoint — no zombie processes
- `download_model.sh` runs at container start; on failure fast-fails
  loudly instead of "container healthy, no /cluster calls work"
- Caches ONNX pack in the docker volume so `docker compose down/up`
  cycles skip the ~1 min download on subsequent starts

## 4. Algorithm in the container

- `/ready` confirms the model actually loads
- `tests/test_integration_live.py` runs against the container, expects
  422 NO_FACE on synthetic inputs — proves the full path works

## 5. API surface

Three endpoints closed by clear contracts:

- `POST /cluster` — sync path, blocking CPU-bound work offloaded via
  `asyncio.to_thread` to keep the event loop responsive
- `POST /cluster/async` — async path with Redis-backed status, the
  right shape for unknown-large workloads
- `GET /health`, `GET /ready`, `GET /metrics` — standard ops surface

Reviewers like to see **typed error codes**. The whole `app/core/errors.py`
file shows the enum + `ServiceError` hierarchy; every API path raises
those, and FastAPI returns the structure shown in `TESTING.md` §7.

## 6. Testing strategy

- Unit, Contract, Integration, Load — four levels
- The **stub the embedder** pattern in `tests/test_api.py` is exactly
  what reviewers care about — model stays off the CI hot path

## 7. Performance

Walk through `'docs/PERFORMANCE.md'`. If you actually ran JMeter, drop
the HTML reports onto a slide. Bottleneck & resolution map is the
deliverable here:

```
CPU-bound embedding detection → multi-worker uvicorn → GPU provider →
  upstream-extract-and-cache in vector DB → swap to smaller detector
  for low-tier paths
```

## 8. Scaling story

Show the horizontal scaling diagram (architecture §5). The key
sentence: "the algorithm is stateless for the sync path, so any pod
serves any request — horizontal scaling is just round-robin + Redis."

## 9. Likely interviewer follow-ups & my prepared answers

| Follow-up | Answer |
|---|---|
| Why Docker and not a service mesh / serverless? | Reproducibility + cost ceiling + no vendor lock-in; easy to move to ECS / Fly / Kubernetes later. |
| Sync vs async API — when do you choose? | Sync for "I already have all images" — fast cold caches. Async for streaming or batch-N > 64; we offer both. |
| Bottleneck of facial clustering for high concurrency? | Detector runtime; mitigations in PERFORMANCE.md table. |
| How do you scale horizontally? | Stateless pods + LB; shared Redis for async status; we never write to disk on the sync path. |
| Error code design principles? | 4xxx client / 5xxx server; numeric semantics; consistent envelope `{detail:{error:{code,name,message}}}` so a UI can localize on the client. |
| CI/CD experience? | GitHub Actions in `.github/workflows/ci.yml` — pytest matrix + docker build-pull-check health caveat; could be extended to image push on tag & SSH deploy. |
| What's missing from this take-home? | Auth, persistent result persistence (only Redis TTL), per-tenant model selection, and proxying uploads via signed S3 URLs. Outlined in ARCHITECTURE §7. |

## 10. What I'd improve given more than a week

1. **Pre-bake model into image** → 50 MB compressed ONNX pack, instant
   cold start. (Sketch in `download_model.sh` comment block.)
2. **Authenticated endpoint** with `slowapi` rate limiting.
3. **Signed S3 Upload** instead of multipart upload.
4. **Distributed inference** via Ray / an inference server (Triton).
5. **Cluster accuracy test harness** on LFW or IJB-C against ground truth;
   right now the synthetic dataset only proves the pipeline, not accuracy.
6. **JMeter load-test in CI** running 30 threads * 30s against the CI
   docker stack and asserting P95 < 10s.
7. **Snake-case shared pydantic schemas out as an OpenAPI package** so
   a separate front-end consumer doesn't flip-flop versions.

## 11.cheat concentration

```
app/services/clusterer.py   ← algorithm (single-linkage union-find)
app/services/face_embedder.py ← InsightFace singleton
app/api/cluster.py          ← API + error envelope
Dockerfile                  ← multi-stage production image
docker-compose.yml          ← app + redis
jmeter/cluster_load.jmx     ← stress test
docs/ARCHITECTURE.md        ← system design essay (show this!)
docs/PERFORMANCE.md         ← bottleneck + optimization table
docs/TESTING.md             ← test pyramid + curl/Postman cases + error ref
tests/test_api.py           ← stubbed-model contract cases
tests/test_clusterer.py     ← pure-Numpy math tests
```
