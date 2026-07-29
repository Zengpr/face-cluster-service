# Face Cluster Service

> Containerized face clustering API — InsightFace ArcFace embeddings + single-linkage
> threshold clustering, deployed via FastAPI + Docker. Built for the Hung Hing Printing
> AI-Science interview take-home.

[![CI](https://github.com/Zengpr/face-cluster-service/actions/workflows/ci.yml/badge.svg)](https://github.com/Zengpr/face-cluster-service/actions)

## TL;DR

```bash
git clone https://github.com/Zengpr/face-cluster-service.git
cd face-cluster-service
docker compose up -d --build
curl http://localhost:8000/health
curl -X POST http://localhost:8000/cluster \
  -F "files=@tests/data/images/ident0_shot0.png" \
  -F "files=@tests/data/images/ident1_shot0.png" \
  -F "threshold=0.6"
```

First run takes ~1-2 minutes while the `buffalo_l` model pack downloads
(~300 MB) from the InsightFace release. Subsequent starts use the
docker volume cache.

##-stack overview|feature|
|---|---|
| Face embedding   | `buffalo_l` pack (ArcFace R50, 512-d) via ONNX Runtime |
| Detector         | RetinaFace-R50 (`det_10g.onnx`) inside `buffalo_l` |
| Clustering       | Single-linkage connected components on cosine sim (default) |
|                  | Optional DBSCAN with noise flagging (`backend=dbscan`) |
| Web framework    | FastAPI 0.110 + uvicorn |
| Async tasks      | Redis (status & result store), in-process `asyncio` worker |
| Image decode     | OpenCV-headless (no GUI deps) |
| Observability    | `/metrics` Prometheus exposition + structured JSON logs |
| Containerization| Multi-stage Dockerfile, tini init, non-root-able runtime |
| CI               | GitHub Actions (unit+API tests + Docker build sanity) |
| Load testing     | JMeter 5.6 scenario + report generation |

## Project layout

```
face-cluster-service/
├── app/
│   ├── api/                # FastAPI routers (cluster, meta)
│   ├── core/                # config, errors, logging, metrics
│   ├── models/             # pydantic schemas
│   └── services/            # face_embedder, clusterer, pipeline, tasks
├── tests/                   # unit + api + integration
├── scripts/                  # model downloader, synthetic dataset generator
├── jmeter/                   # load test plan + README
├── .github/workflows/ci.yml  # CI: pytest + docker build
├── Dockerfile                # multi-stage: builder -> runtime
├── docker-compose.yml        # app + redis
└── requirements.txt
```

## API surface

| Method | Path                       | Description |
|--------|----------------------------|-------------|
| GET    | `/health`                  | Liveness + redis ping + model loaded |
| GET    | `/ready`                   | Readiness — confirms embedder actually loads onnx |
| POST   | `/cluster`                 | Synchronous clustering: upload N images, get groups |
| POST   | `/cluster/async`           | Async submission; returns `task_id` |
| GET    | `/cluster/async/{task_id}` | Poll task status |
| GET    | `/metrics`                 | Prometheus text exposition |

### Sample `POST /cluster`

```bash
curl -X POST http://localhost:8000/cluster \
  -F "files=@a.png" -F "files=@b.png" -F "files=@c.png" \
  -F "threshold=0.6" \
  -F "backend=agglomerative"
```

Response:

```json
{
  "ok": true,
  "n_images": 9,
  "n_clusters": 3,
  "n_noise": 0,
  "threshold": 0.6,
  "backend": "agglomerative",
  "silhouette": 0.41,
  "cluster_sizes": {"0": 3, "1": 3, "2": 3},
  "clusters": [
    {"cluster_id": 0, "files": ["alice_1.png", "alice_2.png", "alice_3.png"]},
    {"cluster_id": 1, "files": ["bob_1.png",   "bob_2.png",   "bob_3.png"]},
    {"cluster_id": 2, "files": ["eve_1.png",   "eve_2.png",   "eve_3.png"]}
  ],
  "label_by_file": {"alice_1.png": 0, "alice_2.png": 0, "alice_3.png": 0, /* ... */},
  "dropped_files": []
}
```

### Error model

All non-2xx responses are shaped as:

```json
{ "detail": { "error": { "code": 4006, "name": "BAD_FILE_PAYLOAD",
                          "message": "Could not decode image" } } }
```

Full code table in **docs/TESTING.md**.

## Configuration

All runtime knobs live in `app/core/config.py` and can be overridden
through env vars (see `.env.example`). Common ones:

| Var                     | Default | Effect |
|-------------------------|---------|--------|
| `DETECTOR_NAME`         | `buffalo_l` | InsightFace pack name |
| `DEFAULT_THRESHOLD`     | `0.6`   | Cosine similarity threshold |
| `CLUSTERING_BACKEND`    | `agglomerative` | or `dbscan` |
| `MIN_SAMPLES_FOR_CLUSTER`| `2`     | Used by dbscan only |
| `MAX_IMAGES_PER_REQUEST`| `64`    | Per-request hard limit |
| `MAX_IMAGE_BYTES`        | `15 MiB`| Per-image size cap |
| `REDIS_URL`              | `redis://redis:6379/0` | Used by async status store |

## Documentation

- **docs/ARCHITECTURE.md** — system design, request lifecycle, scaling story
- **docs/TESTING.md** — comprehensive test strategy & results
- **docs/PERFORMANCE.md** — JMeter methodology + interpretation guide
- **docs/INTERVIEW_NOTES.md** — talking points for the Ming rubric

## License

MIT — see `LICENSE` (or add one). The bundled model
`buffalo_l` is © InsightFace contributors and follows their respective
license; redistributing the ONNX weights is not done here, `download_model.sh`
pulls them at container start.
