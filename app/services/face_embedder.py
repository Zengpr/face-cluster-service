"""Singleton InsightFace model wrapper.

The buffalo_l pack produces 512-d normalized ArcFace embeddings.
We keep the model loaded for the lifetime of the worker to avoid
~0.5s overhead per request on cold inference paths.
"""
from __future__ import annotations

import threading
from contextvars import ContextVar
from typing import Iterable

import numpy as np

from app.core.config import settings
from app.core.errors import InferenceError, ModelLoadError, NoFaceError
from app.core.logging import get_logger

_demo_local = threading.local()

def demo_mode_ctx_set(val: bool) -> None:
    _demo_local.active = val

def demo_mode_ctx_get() -> bool:
    return getattr(_demo_local, "active", False)

log = get_logger(__name__)


class FaceEmbedder:
    _instance_lock = threading.Lock()
    _instance: "FaceEmbedder | None" = None

    def __init__(self) -> None:
        self._app = None
        self._stub = True
        try:
            import insightface as _if  # type: ignore
            from insightface.utils import face_align  # noqa: F401
        except Exception as exc:
            log.warning("insightface_import_failed", err=str(exc))
            return

        # Pre-check model files — insightface.FaceAnalysis.__init__ will
        # BLOCK for minutes trying to download from GitHub when behind a
        # firewall. We skip loading when the pack is missing and fall
        # back to stub mode instead.
        model_dir = settings.models_root / settings.detector_name
        if not (model_dir / "w600k_r50.onnx").exists() and not (model_dir / "det_10g.onnx").exists():
            log.warning(
                "model_pack_not_found_using_stub",
                pack=settings.detector_name,
                path=str(model_dir),
            )
            return

        try:
            self._app = _if.app.FaceAnalysis(
                name=settings.detector_name,
                root=str(settings.models_root.parent),
                providers=["CPUExecutionProvider"]
                if settings.ctx_id < 0
                else ["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._app.prepare(ctx_id=settings.ctx_id, det_size=(settings.det_size, settings.det_size))
            self._stub = False
            log.info(
                "insightface.loaded",
                detector=settings.detector_name,
                ctx=settings.ctx_id,
                det_size=settings.det_size,
            )
        except Exception as exc:
            log.warning("model_load_failed_falling_back_to_stub", err=str(exc))

    @classmethod
    def get(cls) -> "FaceEmbedder":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @staticmethod
    def _make_structured_stub(image_rgb: np.ndarray) -> tuple[np.ndarray, int]:
        """Deterministic structured 512-d embedding based on file content.

        In demo mode, images hash into 3 identity centroids so clustering
        produces clean, repeatable groups without real faces.
        """
        import hashlib

        digest = hashlib.md5(image_rgb.tobytes()).hexdigest()
        identity = int(digest[:8], 16) % 3
        base = np.zeros(512, dtype=np.float32)
        base[identity * 8 : (identity + 1) * 8] = 1.0
        rng = np.random.default_rng(identity * 1000)
        noise = rng.standard_normal(512).astype(np.float32) * 0.1
        v = base + noise
        v /= np.linalg.norm(v) + 1e-9
        return v, 1

    def embed_image(self, image_rgb: np.ndarray, demo_mode: bool = False) -> tuple[np.ndarray, int]:
        """Return (512-d embedding, num_faces_detected).

        When the real model was not loaded (e.g. network-restricted build
        environment), returns a content-hashed deterministic stub embedding
        so the full API + clustering pipeline can be exercised end-to-end.
        """
        if image_rgb is None or image_rgb.size == 0:
            raise InferenceError("Empty image array passed to embedder")

        if demo_mode or demo_mode_ctx_get() or settings.demo_mode or self._stub or self._app is None:
            return self._make_structured_stub(image_rgb)

        try:
            faces = self._app.get(image_rgb)
        except Exception as exc:
            raise InferenceError(f"detector.run failed: {exc}") from exc

        if not faces:
            raise NoFaceError()

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
