# Performance Test Plan & Interpretation

This doc covers the **JMeter** load-test methodology used to measure the
face-cluster service and explains how to interpret the resulting numbers
for the interview debrief.

## 1. Objective

- Identify the throughput ceiling on a single container worker
- Measure latency percentiles (P50, P95, P99) under increasing load
- Detect memory creep and error-rate behaviour at saturation
- Provide suggestions for horizontal scaling

## 2. Test environment assumptions

| Layer | Assumed unit |
|---|---|
| Container host | 4 vCPU, 16 GB RAM (laptop-grade) |
| Worker | uvicorn `--workers 1` (default compose) |
| Image payload | 256×256 PNG, 3 images per request (synthetic) |
| Network | localhost |

## 3. Test matrix

| Scenario    | Threads | Ramp | Duration | Goal |
|-------------|---------|------|----------|------|
| Baseline    |       1 |   1s |   60s    | Single-user latency floor |
| Light       |       5 |  10s |   60s    | Typical mobile API load |
| Medium      |      10 |  10s |  120s    | Same-thread concurrency |
| Heavy       |      20 |  10s |  180s    | Single-worker ceiling |
| Soak        |      10 |  10s |  900s    | Memory / FD leak |
| Spike       |      50 |   1s |   30s    | Backpressure behaviour |

## 4. Running

```bash
# Project root, JMeter 5.6+
jmeter -n -t jmeter/cluster_load.jmx \
  -Jthreads=10 -Jduration=120 \
  -Jbase=http://localhost:8000 \
  -l jmeter/results_medium.jtl \
  -e -o jmeter/report_medium
```

Open `jmeter/report_medium/index.html` in a browser.

## 5. Expected numbers & bottleneck

- Per-image ArcFace inference on CPU ≈ **0.3-1.0 s** (single thread)
- Per-3-image request roughly dominated by N × per-image + clustering
  overhead = **~2-4 s** per request at P50
- At 10 threads we should plateau around **2.5 req/s** throughput
- At 20+ threads P95 climbs rapidly and the single worker saturates

### Why these numbers
The inference graph below dominates the request budget:

```
decode_image      → fast (~10 ms)
FaceEmbedder      → 0.3 – 1 s/face
cluster_embeddings → O(N²) but negligible at N≤16
```

The facial detector (RetinaFace `det_10g.onnx`) running inside the
`buffalo_l` pack on CPU is the bottleneck — not the FastAPI layer.

## 6. Optimization roadmap

| Optimization | Expected gain | Effort |
|---|---|---|
| `--workers 4` on uvicorn (multi-process) | ~4× throughput on 4-core host | trivial |
| Increase concurrency with GPU provider (`CUDAExecutionProvider`) | ~10-50×/face | medium |
| Move embedding extraction upstream, store in vector DB (FAISS / pgvector) | drop per-request cost to ms | high |
| Use `buffalo_s` (smaller detector) for tier-2 latency-sensitive paths | ~2× faster, accuracy drop | trivial |
| Add Redis-based response caching keyed by content hash | Hit-rate dependent; saves repeat re-detection | easy |
| Replace single-linkage with FAISS-IVF/HNSW clustering for >64 image batches | O(N log N) | medium |
| Pre-warm models at container start (current `download_model.sh` covers download only, but InsightFace's lazy ONNX load still needs a single request to fill) | ≈ -2 s on first request | small |

## 7. What to bring to the interview

1. Save the HTML report (`-e -o`) from each scenario in the matrix above.
2. Eyebone three numbers in each: throughput, P95, and error %.
3. Cross-reference with the chosen optimization; argue which layer of the
   stack each recommendation targets.
4. If you ran a "Medium" scenario twice — once with `--workers 1` and
   once with `--workers 4` — you can show concrete throughput delta
   from "easy" optimization #1 above.

## 8. Sample readers from ./results.jtl

```python
import pandas as pd
df = pd.read_csv("jmeter/results_medium.jtl",
                 sep=",",
                 usecols=["timeStamp","elapsed","label","success","responseCode",
                          "grpThreads","allThreads","Latency","ConnectTime"])
df["elapsed_ms"] = df["elapsed"]
print(df.groupby("label").agg(
    n=("elapsed","size"),
    ok=("success","mean"),
    p50=("elapsed_ms", lambda s: s.quantile(0.5)),
    p95=("elapsed_ms", lambda s: s.quantile(0.95)),
    p99=("elapsed_ms", lambda s: s.quantile(0.99)),
    avg=("elapsed_ms","mean"),
))
```

A pre-built helper is in `scripts/jtl_to_csv.py`.
