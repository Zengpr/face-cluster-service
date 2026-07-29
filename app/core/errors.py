"""Centralized error codes & typed service exceptions.

Error code convention:
  - 4xxx: client error (bad request, unsupported file, quota exceeded)
  - 5xxx: server error (model load failed, inference crash)
"""
from __future__ import annotations

from enum import IntEnum


class ErrCode(IntEnum):
    # ---- 4xxx client ----
    NO_IMAGES = 4001
    TOO_MANY_IMAGES = 4002
    IMAGE_TOO_LARGE = 4003
    UNSUPPORTED_CONTENT_TYPE = 4004
    INVALID_THRESHOLD = 4005
    BAD_FILE_PAYLOAD = 4006

    # ---- 5xxx server ----
    MODEL_LOAD_FAILED = 5001
    NO_FACE_IN_IMAGE = 5002
    INFERENCE_FAILED = 5003
    CLUSTERING_FAILED = 5004
    TASK_NOT_FOUND = 5005


class ServiceError(Exception):
    def __init__(self, code: ErrCode, message: str, http_status: int = 400):
        super().__init__(message)
        self.code = int(code)
        self.err_name = code.name
        self.message = message
        self.http_status = http_status

    def to_payload(self) -> dict:
        return {
            "error": {
                "code": self.code,
                "name": self.err_name,
                "message": self.message,
            }
        }


class ModelLoadError(ServiceError):
    def __init__(self, message: str):
        super().__init__(ErrCode.MODEL_LOAD_FAILED, message, http_status=503)


class NoFaceError(ServiceError):
    def __init__(self, message: str = "No face detected in one or more images"):
        super().__init__(ErrCode.NO_FACE_IN_IMAGE, message, http_status=422)


class InferenceError(ServiceError):
    def __init__(self, message: str):
        super().__init__(ErrCode.INFERENCE_FAILED, message, http_status=500)
