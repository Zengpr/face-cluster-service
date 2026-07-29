"""High-level orchestration: bytes -> embeddings -> clusters -> DTO."""
from __future__ import annotations

from typing import Literal

import numpy as np

from app.core.errors import ErrCode, NoFaceError, ServiceError
from app.core.logging import get_logger
from app.services.clusterer import cluster_embeddings
from app.services.face_embedder import FaceEmbedder
from app.services.preprocess import decode_image

log = get_logger(__name__)


def run_cluster(
    files: list[tuple[str, bytes, str | None]],
    threshold: float | None,
    backend: Literal["agglomerative", "dbscan"] | None,
) -> dict:
    if not files:
        raise ServiceError(ErrCode.NO_IMAGES, "No images supplied")

    images_rgb: list[tuple[str, np.ndarray]] = []
    for fname, data, ct in files:
        try:
            rgb = decode_image(data, ct)
        except ServiceError:
            log.warning("decode.failed", file=fname)
            raise
        images_rgb.append((fname, rgb))

    if not images_rgb:
        raise ServiceError(ErrCode.NO_IMAGES, "All supplied files failed to decode")

    embedder = FaceEmbedder.get()
    embeddings, kept_files = [], []
    for fname, rgb in images_rgb:
        try:
            emb, n_faces = embedder.embed_image(rgb)
        except NoFaceError:
            log.warning("no.face", file=fname)
            continue
        embeddings.append(emb)
        kept_files.append(fname)

    if not embeddings:
        raise NoFaceError(
            "No face detected in any of the supplied images"
        )

    matrix = np.stack(embeddings, axis=0)
    result = cluster_embeddings(matrix, threshold=threshold, backend=backend)

    groups: dict[int, list[str]] = {}
    for fname, label in zip(kept_files, result.labels):
        groups.setdefault(int(label), []).append(fname)
    # Stable sort each group by filename for deterministic output.
    groups = {k: sorted(v) for k, v in sorted(groups.items(), key=lambda kv: kv[0])}

    return {
        "n_images": len(kept_files),
        "n_clusters": result.n_clusters,
        "n_noise": result.cluster_sizes.get(-1, 0),
        "threshold": float(threshold or 0.0),
        "backend": backend or "agglomerative",
        "silhouette": float(result.silhouette),
        "cluster_sizes": result.cluster_sizes,
        "clusters": [
            {"cluster_id": cid, "files": files_list}
            for cid, files_list in groups.items()
        ],
        "label_by_file": {f: int(l) for f, l in zip(kept_files, result.labels)},
        "dropped_files": [f for f, _ in images_rgb if f not in kept_files],
    }
