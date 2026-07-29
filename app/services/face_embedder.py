"""Singleton InsightFace model wrapper.

The buffalo_l pack produces 512-d normalized ArcFace embeddings.
We keep the model loaded for the lifetime of the worker to avoid
~0.5s overhead per request on cold inference paths.
"""
from __future__ import annotations

import threading
from typing import Iterable

import numpy as np

from app.core.config import settings
from app.core.errors import InferenceError, ModelLoadError, NoFaceError
from app.core.logging import get_logger

log = get_logger(__name__)


class FaceEmbedder:
    _instance_lock = threading.Lock()
    _instance: "FaceEmbedder | None" = None

    def __init__(self) -> None:
        try:
            import insightface as _if  # type: ignore
            from insightface.utils import face_align  # noqa: F401
        except Exception as exc:  # pragma: no cover - defensive
            raise ModelLoadError(f"insightface import failed: {exc}") from exc

        self._app = _if.app.FaceAnalysis(
            name=settings.detector_name,
            root=str(settings.models_root.parent),
            providers=["CPUExecutionProvider"]
            if settings.ctx_id < 0
            else ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=settings.ctx_id, det_size=(settings.det_size, settings.det_size))
        log.info(
            "insightface.loaded",
            detector=settings.detector_name,
            ctx=settings.ctx_id,
            det_size=settings.det_size,
        )

    @classmethod
    def get(cls) -> "FaceEmbedder":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def embed_image(self, image_rgb: np.ndarray) -> tuple[np.ndarray, int]:
        """Return (512-d embedding, num_faces_detected).

        Raises ``NoFaceError`` when no face is found and ``InferenceError``
        when embedding extraction fails for a detected face.
        """
        if image_rgb is None or image_rgb.size == 0:
            raise InferenceError("Empty image array passed to embedder")

        try:
            faces = self._app.get(image_rgb)
        except Exception as exc:
            raise InferenceError(f"detector.run failed: {exc}") from exc

        if not faces:
            raise NoFaceError()

        # Keep the largest face when several are present — that gives
        # the most reliable embedding for identity clustering.
        primary = max(faces, key=lambda f: (int(f.bbox[2]) - int(f.bbox[0])) *
                      (int(f.bbox[3]) - int(f.bbox[1])))
        emb = primary.embedding
        if emb is None or emb.size == 0:
            raise InferenceError("Detector returned face without embedding")
        emb = emb.astype(np.float32)
        norm = np.linalg.norm(emb) + 1e-9
        return emb / norm, len(faces)

    def embed_batch(self, images_rgb: Iterable[np.ndarray]) -> np.ndarray:
        rows = []
        for img in images_rgb:
            emb, _ = self.embed_image(img)
            rows.append(emb)
        return np.stack(rows, axis=0).astype(np.float32)
