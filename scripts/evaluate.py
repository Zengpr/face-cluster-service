"""Evaluate clustering accuracy against ground-truth labels.

Usage:
  python scripts/evaluate.py

Generates synthetic structured embeddings with known ground-truth,
runs the full clustering pipeline, and reports:
  - Adjusted Rand Index (ARI)
  - Normalised Mutual Information (NMI)
  - Homogeneity / Completeness / V-Measure
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.clusterer import cluster_embeddings


def _make_benchmark(n_ids: int = 5, shots: int = 4,
                    dim: int = 512, noise: float = 0.15) -> tuple[np.ndarray, list[int]]:
    """Generate structured embeddings with known ground-truth."""
    rng = np.random.default_rng(42)
    centroids = rng.standard_normal((n_ids, dim)).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9

    emb_list: list[np.ndarray] = []
    labels: list[int] = []
    for cid in range(n_ids):
        for _ in range(shots):
            v = centroids[cid] + rng.standard_normal(dim).astype(np.float32) * noise
            v /= np.linalg.norm(v) + 1e-9
            emb_list.append(v)
            labels.append(cid)

    return np.stack(emb_list, axis=0), labels


def evaluate() -> None:
    embeddings, ground_truth = _make_benchmark()
    n = len(embeddings)

    for backend in ("agglomerative", "dbscan"):
        result = cluster_embeddings(embeddings, threshold=0.6, backend=backend)
        pred = list(result.labels)

        from sklearn.metrics import (adjusted_rand_score, homogeneity_completeness_v_measure,
                                     normalized_mutual_info_score)

        ari = adjusted_rand_score(ground_truth, pred)
        nmi = normalized_mutual_info_score(ground_truth, pred)
        h, c, v = homogeneity_completeness_v_measure(ground_truth, pred)

        print(f"\n{'='*50}")
        print(f"Backend: {backend}")
        print(f"{'='*50}")
        print(f"  Images:           {n}")
        print(f"  Ground-truth ids: {len(set(ground_truth))}")
        print(f"  Found clusters:   {result.n_clusters}")
        print(f"  Noise points:     {result.cluster_sizes.get(-1, 0)}")
        print(f"  Silhouette:       {result.silhouette:.4f}")
        print(f"  ARI:              {ari:.4f}")
        print(f"  NMI:              {nmi:.4f}")
        print(f"  Homogeneity:      {h:.4f}")
        print(f"  Completeness:     {c:.4f}")
        print(f"  V-Measure:        {v:.4f}")

        if ari > 0.9:
            print("  >>> PASS (ARI > 0.9)")
        else:
            print("  >>> WARNING: ARI below expected threshold")


if __name__ == "__main__":
    evaluate()
