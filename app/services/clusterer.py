"""Face clustering engine.

Two interchangeable back-ends:

  1. ``agglomerative`` — connected-components on the similarity graph
     using cosine threshold. Deterministic, no hyperparameter sensitivity
     to cluster shape. This is the default and is the standard recipe
     used in production face clustering pipelines.

  2. ``dbscan`` — density-based clustering with cosine metric. Better
     when there are non-identity outliers to drop.

Both return the same shape so the API can swap back-ends transparently.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.core.config import settings
from app.core.errors import ErrCode, ServiceError
from app.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class ClusterResult:
    labels: list[int]
    n_clusters: int
    cluster_sizes: dict[int, int]
    silhouette: float
    sim_matrix: np.ndarray | None = None


def _cosine_sim_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    normalized = embeddings / norms
    return normalized @ normalized.T


def _agglomerative_threshold(sim: np.ndarray, threshold: float) -> np.ndarray:
    """Single-linkage connected components via union-find.

    Two embeddings share a cluster when cosine similarity >= threshold.
    Single-linkage is the rule used by InsightFace's reference clusterer;
    it allows transitive grouping which is what we want for one identity
    spread across many photos.
    """
    n = sim.shape[0]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    iu, ju = np.triu_indices(n, k=1)
    mask = sim[iu, ju] >= threshold
    for i, j in zip(iu[mask], ju[mask]):
        union(int(i), int(j))

    return np.array([find(i) for i in range(n)], dtype=np.int64)


def _dbscan_threshold(sim: np.ndarray, threshold: float, min_samples: int) -> np.ndarray:
    """DBSCAN on a precomputed similarity graph.

    Threshold is interpreted as: any pair with sim >= threshold is a
    neighbour. Points with fewer than ``min_samples`` neighbours are
    labelled -1 (noise).
    """
    n = sim.shape[0]
    neighbours: list[list[int]] = [[] for _ in range(n)]
    iu, ju = np.triu_indices(n, k=1)
    mask = sim[iu, ju] >= threshold
    for i, j in zip(iu[mask], ju[mask]):
        neighbours[int(i)].append(int(j))
        neighbours[int(j)].append(int(i))

    visited = [False] * n
    labels = [-2] * n
    cluster_id = 0

    def expand(seed: int, cid: int) -> None:
        stack = [seed]
        labels[seed] = cid
        while stack:
            x = stack.pop()
            if not visited[x]:
                visited[x] = True
                for y in neighbours[x]:
                    if labels[y] == -2:
                        labels[y] = cid
                        if len(neighbours[y]) >= min_samples:
                            stack.append(y)

    for p in range(n):
        if visited[p]:
            continue
        if len(neighbours[p]) < min_samples:
            labels[p] = -1
            visited[p] = True
            continue
        expand(p, cluster_id)
        cluster_id += 1

    return np.array(labels, dtype=np.int64)


def _silhouette_safe(sim: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels[labels >= 0])
    if unique.size < 2:
        return 0.0
    from sklearn.metrics import silhouette_score

    distance = 1.0 - sim
    try:
        return float(silhouette_score(distance, labels, metric="precomputed"))
    except Exception:
        return 0.0


def cluster_embeddings(
    embeddings: np.ndarray,
    threshold: float | None = None,
    backend: Literal["agglomerative", "dbscan"] | None = None,
    min_samples: int | None = None,
) -> ClusterResult:
    if embeddings is None or embeddings.size == 0:
        raise ServiceError(ErrCode.NO_IMAGES, "empty embedding matrix")

    if backend is None:
        backend = settings.clustering_backend
    if threshold is None:
        threshold = settings.default_threshold
    if min_samples is None:
        min_samples = settings.min_samples_for_cluster

    if embeddings.ndim != 2:
        raise ServiceError(ErrCode.NO_IMAGES, "embeddings must be 2D (N,512)")

    sim = _cosine_sim_matrix(embeddings.astype(np.float32))
    if backend == "agglomerative":
        labels = _agglomerative_threshold(sim, threshold)
    elif backend == "dbscan":
        labels = _dbscan_threshold(sim, threshold, min_samples)
    else:
        raise ServiceError(ErrCode.NO_IMAGES, f"unknown backend {backend!r}")

    # Compress dense labels to 0..K-1 in encounter order; preserve noise=-1.
    compact = np.full_like(labels, -1)
    next_id = 0
    flatten = labels.tolist()
    seen: dict[int, int] = {}
    for idx, lbl in enumerate(flatten):
        if lbl < 0:
            continue
        if lbl not in seen:
            seen[int(lbl)] = next_id
            next_id += 1
        compact[idx] = seen[int(lbl)]

    out_labels = compact.tolist()
    sizes: dict[int, int] = {}
    for lbl in out_labels:
        sizes[int(lbl)] = sizes.get(int(lbl), 0) + 1

    n_clusters = next_id
    sil = _silhouette_safe(sim, compact)

    log.info(
        "cluster.done",
        backend=backend,
        threshold=threshold,
        n_images=int(compact.size),
        n_clusters=n_clusters,
        silhouette=sil,
    )
    return ClusterResult(
        labels=out_labels,
        n_clusters=n_clusters,
        cluster_sizes={int(k): int(v) for k, v in sizes.items()},
        silhouette=sil,
        sim_matrix=sim,
    )
