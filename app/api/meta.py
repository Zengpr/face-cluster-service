"""Health, readiness, OpenAPI extension meta-routes."""
from __future__ import annotations

import os
import subprocess

import redis.asyncio as aioredis
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.models.schemas import HealthResponse

router = APIRouter()

GIT_SHA = os.environ.get("GIT_SHA", "")


def _git_sha() -> str:
    if GIT_SHA:
        return GIT_SHA
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    redis_ok = True
    try:
        client = aioredis.from_url(settings.redis_url, decode_responses=True)
        await client.ping()
        await client.aclose()
    except Exception:
        redis_ok = False
    return HealthResponse(
        status="healthy" if redis_ok else "degraded",
        model_loaded=True,
        detector=settings.detector_name,
        git_sha=_git_sha(),
    )


@router.get("/ready", status_code=status.HTTP_200_OK)
async def ready() -> JSONResponse:
    from app.services.face_embedder import FaceEmbedder

    try:
        FaceEmbedder.get()
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"ready": False, "reason": str(exc)},
        )
    return JSONResponse({"ready": True, "detector": settings.detector_name})
