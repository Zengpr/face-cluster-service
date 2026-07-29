"""FastAPI dependencies: model load guard and image budget."""
from __future__ import annotations

from fastapi import Depends, HTTPException, UploadFile

from app.core.config import settings
from app.core.errors import ErrCode, ServiceError
from app.services.face_embedder import FaceEmbedder


def embedder_loaded() -> None:
    try:
        FaceEmbedder.get()
    except Exception:
        # We swallow the in-route load and let the handler surface a
        # clean 503 via ServiceError after first request.
        pass


def enforce_image_budget(files: list[UploadFile]) -> None:
    if not files:
        raise HTTPException(
            status_code=400,
            detail=ServiceError(ErrCode.NO_IMAGES, "no files supplied").to_payload(),
        )
    if len(files) > settings.max_images_per_request:
        raise HTTPException(
            status_code=400,
            detail=ServiceError(
                ErrCode.TOO_MANY_IMAGES,
                f"too many images ({len(files)} > {settings.max_images_per_request})",
            ).to_payload(),
        )
