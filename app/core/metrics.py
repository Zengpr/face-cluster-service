"""Prometheus metrics — request latency / counts / cluster cardinality."""
from __future__ import annotations

from prometheus_client import Counter, Histogram, generate_latest

REQUESTS = Counter(
    "fc_requests_total", "Total count of clustering requests", ("endpoint", "status")
)
LATENCY = Histogram(
    "fc_request_seconds",
    "Per-request wall time for clustering",
    ("endpoint",),
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60),
)
CLUSTERS = Histogram(
    "fc_clusters_per_call",
    "Number of clusters produced per successful call",
    buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256),
)
IMAGES_PER_CALL = Histogram(
    "fc_images_per_call",
    "Number of images per call",
    buckets=(1, 2, 4, 8, 16, 32, 64, 128),
)


def exposition() -> bytes:
    return generate_latest()
