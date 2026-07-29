# Architecture

## 1. High-level diagram

```
                ┌────────────────────────────────────────────┐
                │                Client (curl / Postman / UI)  │
                └──────────────┬─────────────────────┬─────────┘
                               │                     │
                       POST /cluster         POST /cluster/async
                               │                     │
                               ▼                     ▼
                ┌──────────────────────────┐  ┌─────────────────────────┐
                │  uvicorn (worker * 1..N)  │  │  uvicorn                 │
                │  └─ FastAPI app           │  │  └─ asyncio.create_task  │
                │     ├─ POST /cluster      │  │     └─ to_thread=pipeline│
                │     │    → to_thread       │  │         (blocking)       │
                │     │    → run pipeline    │  │                          │
                │     └─ error codes         │  │  results stored in       │
                └──────────────┬──────────┘  │     redis (TTL 1h)        │
                               │             └──────────────┬──────────┘
                               │                            │
                               │            ┌───────────────┘
                               │            │
                               ▼            ▼
                ┌─────────────────────────────────────────┐
                │  Service Layer (pure python)            │
                │  ├─ preprocess.decode_image             │
                │  ├─ FaceEmbedder.get().embed_image      │
                │  │    (InsightFace buffalo_l — ArcFace 512-d)
                │  └─ cluster_embeddings                  │
                │       ├─ cosine sim matrix              │
                │       ├─ agglomerative single-linkage    │
                │       └─ sklearn silhouette (optional)  │
                └─────────────────────────────────────────┘
                               │
                               ▼
                          Media: N images in
                          Output: K clusters out
```

## 2. Why these choices

| Decision | Option | Rationale |
|---|---|---|
| Web framework | **FastAPI** over Flask | Native async, pydantic v2, OpenAPI docs auto-generated, modular DI via `Depends()`. |
| Face embedding | **InsightFace buffalo_l** | Industry-standard ArcFace 512-d, free permissive ONNX, GPU/CPU portability. Smaller alternatives (`buffalo_s`, `buffalo_m`) drop in via `DETECTOR_NAME`. |
| Clustering    | **Single-linkage threshold** | Reference recipe for face clustering (transitive grouping — same identity across many shots). DBSCAN offered as alternative for noise filtering. |
| Async tasks   | **Redis + asyncio.to_thread** | No need for Celery for a demo. Embedding is CPU-bound, so we must release the event loop. Redis gives cross-worker state needed when scaling uvicorn workers > 1. |
| Container init | **tini** + entrypoint-as-model-loader | Prevents zombie processes, ensures ONNX weights present before uvicorn binds, gracefully degrades on download failure. |
| Observability | **Prometheus exposition** | Industry default, pairs with grafana out-of-the-box. Counter + histogram naming aligns with USE+RED conventions. |

## 3. Request lifecycle (POST /cluster)

1. **HTTP/FastAPI** validates `files` body & content-type permission list.
2. **`enforce_image_budget`** rejects empty requests and >64-image requests.
3. **`embedder_loaded`** forces lazy model load once.
4. **`asyncio.to_thread(run_cluster, ...)`** offloads CPU-bound work —
   uvicorn async loop stays responsive even while ArcFace runs.
5. **`preprocess.decode_image`** uses `cv2.imdecode` directly on bytes
   to avoid disk I/O. We convert BGR→RGB once for InsightFace.
6. **`FaceEmbedder.embed_image`** returns a L2-normalized 512-d vector
   of the largest face. Faces that fail detection are dropped and
   reported in `dropped_files`.
7. **`cluster_embeddings`** computes the NxN cosine matrix on a float32
   set, then runs **single-linkage union-find** at the supplied threshold.
   A silhouette score is computed when ≥2 clusters form — useful for
   tuning threshold offline.
8. FastAPI serializes the result via pydantic v2 → JSON.

## 4. Lifecycle (POST /cluster/async)

Same path before step 4; instead of `to_thread` we submit a task,
return a `task_id`, and write results to Redis under
`task:result:{task_id}` with an hour TTL.

Production move: replace `asyncio.create_task(runner)` with enqueueing
an [Arq](https://github.com/python-arq/arq) task on a Redis queue; the
worker pool becomes horizontal.

## 5. Horizontal scaling

```
                ┌──────────── Load Balancer (Nginx / Traefik / AWS ALB) ──────────┐
                │                                                                  │
   ┌────────────┴────┐ ┌────────────┴───┐ ┌────────────┴───┐ ┌────────────┴───┐
   │ pod-1: 4-8 uv   │ │ pod-2: 4-8 uv  │ │ pod-...         │ │ pod-N: 4-8 uv  │
   │  workers, CPU   │ │  workers        │ │                 │ │                │
   └─────────────────┘ └────────────────┘ └────────────────┘ └────────────────┘
                                    │
                          shared: Redis (HA), shared NFS for model cache
```

Same image implies deterministic embedding & clustering, so any pod can
serve any request — fully stateless for sync path.

## 6. Friction points / limits

| Limit | Mitigation |
|---|---|
| CPU-bound embedding (~0.3-1s per image) | Extract embeddings upstream via push-based upload pipeline, store in vector DB (Milvus / pgvector / FAISS) and reuse across calls. |
| NxN clustering is O(N²) | Cap requests to 64; beyond that land in analytic SQL/Python jobs (e.g. faiss IVF/HNSW index over precomputed embeddings). |
| Model download cold start (~1-2 min) | Bake model into image (anti-pattern for prod size); better — mount the `~/.insightface` cache as a PVC / S3-backed pre-populated volume. |
| Redis single-point | Replace with Cluster or Valkey; the API can also gracefully degrade to in-memory store if you want for demo. |

## 7. Security considerations

- `/cluster` accepts only `image/jpeg|png|webp|bmp` content types.
- Hard limits: 64 images & 15 MiB per file by default.
- No filesystem writes — everything streamed via `numpy.frombuffer`.
- For prod: add auth (API key header), rate limit via `slowapi`, and signed
  S3 upload URLs instead of multipart.
