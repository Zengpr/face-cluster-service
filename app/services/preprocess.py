"""Image preprocessing helpers: decode -> BGR -> RGB + size guards."""
from __future__ import annotations

import io

import cv2
import numpy as np

from app.core.config import settings
from app.core.errors import ErrCode, ServiceError


def decode_image(raw: bytes, content_type: str | None) -> np.ndarray:
    if not raw:
        raise ServiceError(ErrCode.BAD_FILE_PAYLOAD, "Empty file payload")
    if len(raw) > settings.max_image_bytes:
        raise ServiceError(
            ErrCode.IMAGE_TOO_LARGE,
            f"file > {settings.max_image_bytes} bytes",
        )
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ServiceError(
            ErrCode.BAD_FILE_PAYLOAD,
            "Could not decode image (corrupt or unsupported format)",
        )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    area_pct = (h * w) / (settings.det_size * settings.det_size)
    if area_pct * (1.0 / 1.0) < settings.min_face_area_pct:
        # We keep this as a soft guard, not a hard reject — insightface
        # itself will return zero faces and the upper layer will surface
        # a clean ``NoFaceError``.
        pass
    return image
