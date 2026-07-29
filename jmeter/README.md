# JMeter load test plan

This directory contains the load-test assets for the Face Cluster Service.

## Prerequisites
- JMeter 5.6+  (https://jmeter.apache.org/)
- The face-cluster docker stack running locally (``docker compose up -d``)
- Sample images reachable relative to the JMeter CWD (project root by default)

## Running

```bash
# Quick smoke (10 users, 60s, single 256x256 image per request)
cd ..   # project root
jmeter -n -t jmeter/cluster_load.jmx \
      -Jthreads=10 -Jduration=60 \
      -Jbase=http://localhost:8000 \
      -l jmeter/results_$(date +%Y%m%d_%H%M).jtl

# Heavy load (50 users, 5 min, 3 images per request)
# substitute file paths via -Jfile1, -Jfile2, -Jfile3 if you want a batch
jmeter -n -t jmeter/cluster_load.jmx \
      -Jthreads=50 -Jduration=300 \
      -l jmeter/results_heavy.jtl \
      -e -o jmeter/report_heavy
```

## Generating the HTML report
```bash
jmeter -g jmeter/results_heavy.jtl -o jmeter/report_heavy
```

## Recommended test matrix

| Scenario | Threads | Duration | Notes |
|----------|---------|----------|-------|
| Baseline         |  1 |  60s | Single-user latency floor |
| Light            |  5 |  60s | Realistic mobile-API usage |
| Stress           | 20 | 120s | uvicorn single-worker ceiling |
| Soak             | 10 | 900s | Look for memory creep        |
| Spike (ramp=1s)  | 50 |  30s | Backpressure behaviour       |

## Key metrics to capture
- **Throughput** (req/s) — limited by CPU + ArcFace inference cost
- **P50 / P95 / P99** latency — primary user-experience numbers
- **Error rate** at high concurrency — tells us when the worker saturates
- **Per-request bytes** — sanity-check upload bandwidth

## Expected bottleneck
On a single CPU worker, ~0.3–1 s per image for buffalo_l embedding.

At 20 concurrent threads we expect throughput to plateau around
**1/thread × workers / mean_image_time ≈ 3-7 req/s** with P95 ~5-10s.
Scaling horizontally is achieved by running N uvicorn workers per pod
and N pods behind a load balancer — see ``docs/ARCHITECTURE.md``.
