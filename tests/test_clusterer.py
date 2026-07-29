"""Pure-Numpy unit tests of the clustering math (no model required)."""
from __future__ import annotations

import numpy as np

from app.services.clusterer import cluster_embeddings


def _two_clusters_far_apart() -> np.ndarray:
    rng = np.random.default_rng(42)
    a = rng.standard_normal((3, 512)).astype(np.float32)
    b = rng.standard_normal((3, 512)).astype(np.float32) + 10.0
    return np.vstack([a, b])


def test_two_clusters_with_perfect_separation():
    emb = _two_clusters_far_apart()
    # normalize so cosine *behaves like euclidean here
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    res = cluster_embeddings(emb, threshold=0.8, backend="agglomerative")
    assert res.n_clusters == 2
    assert res.labels == [0, 0, 0, 1, 1, 1]


def test_dbscan_noise_flagged():
    rng = np.random.default_rng(7)
    base = np.zeros(512, dtype=np.float32)
    base[0] = 1.0
    sims = rng.standard_normal((5, 512)).astype(np.float32)
    sims[0] = base
    sims[1] = base * 0.98
    sims[2] = base * 1.02
    # Two lonely points should be flagged as noise (-1)
    sims[3] = rng.standard_normal(512).astype(np.float32) * 1000
    sims[4] = rng.standard_normal(512).astype(np.float32) * 1000
    sims /= np.linalg.norm(sims, axis=1, keepdims=True) + 1e-9
    res = cluster_embeddings(sims, threshold=0.7, backend="dbscan", min_samples=2)
    assert -1 in res.labels
    assert res.n_clusters >= 1


def test_empty_embeddings_raises():
    import pytest
    from app.core.errors import ServiceError
    with pytest.raises(ServiceError):
        cluster_embeddings(np.zeros((0, 512)))


def test_invalid_backend_raises():
    import pytest
    from app.core.errors import ServiceError
    with pytest.raises(ServiceError):
        cluster_embeddings(np.zeros((3, 512)), backend="kmeans")  # type: ignore


def test_label_compression_is_stable():
    # When union-find output isn't already 0..K-1 contiguous, the
    # compact mapping must produce deterministic labels regardless
    # of internal ordering.
    emb = _two_clusters_far_apart()
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    res1 = cluster_embeddings(emb, threshold=0.8)
    res2 = cluster_embeddings(emb, threshold=0.8)
    assert res1.labels == res2.labels
