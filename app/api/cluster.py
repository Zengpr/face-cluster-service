"""Clustering API routes."""
from __future__ import annotations

import asyncio
import time
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile

from app.api.deps import enforce_image_budget, embedder_loaded
from app.core.config import settings
from app.core.errors import ErrCode, ServiceError
from app.core.logging import get_logger
from app.core.metrics import CLUSTERS, IMAGES_PER_CALL, LATENCY, REQUESTS
from app.models.schemas import (
    AsyncStatusResponse,
    AsyncSubmitResponse,
    ClusterResponse,
)
from app.services import tasks as task_store
from app.services.pipeline import run_cluster

router = APIRouter()
log = get_logger(__name__)

_SUCCESS = "ok"
_FAILED = "error"


def _ingest(files: list[UploadFile]) -> list[tuple[str, bytes, str | None]]:
    out: list[tuple[str, bytes, str | None]] = []
    for f in files:
        ct = f.content_type
        if ct and ct not in settings.allowed_content_types:
            raise ServiceError(
                ErrCode.UNSUPPORTED_CONTENT_TYPE,
                f"{f.filename} has unsupported content-type {ct}",
            )
        data = f.file.read()
        out.append((f.filename or f"file_{len(out)}", data, ct))
    return out


@router.post(
    "/cluster",
    response_model=ClusterResponse,
    responses={
        400: {"description": "Bad request — see error.code"},
        422: {"description": "No face detected in any image"},
        503: {"description": "Model not loaded"},
    },
)
async def cluster_sync(
    files: list[UploadFile] = File(...),
    threshold: float | None = Form(None),
    backend: Literal["agglomerative", "dbscan"] | None = Form(None),
    x_demo_mode: bool = Header(False),
    _loaded: None = Depends(embedder_loaded),
    _budget: None = Depends(enforce_image_budget),
) -> ClusterResponse:
    t_start = time.perf_counter()
    endpoint = "cluster.sync"
    try:
        if threshold is not None and not (0.0 <= threshold <= 2.0):
            raise ServiceError(ErrCode.INVALID_THRESHOLD, "threshold must be in [0,2]")
        files_ingested = _ingest(files)
        result = await asyncio.to_thread(
            run_cluster, files_ingested, threshold, backend, x_demo_mode
        )
        REQUESTS.labels(endpoint=endpoint, status=_SUCCESS).inc()
        CLUSTERS.observe(result["n_clusters"])
        IMAGES_PER_CALL.observe(result["n_images"])
        return ClusterResponse(**result)
    except ServiceError as exc:
        REQUESTS.labels(endpoint=endpoint, status=_FAILED).inc()
        raise HTTPException(
            status_code=exc.http_status, detail=exc.to_payload()
        )
    finally:
        LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - t_start)


@router.post("/cluster/async", response_model=AsyncSubmitResponse)
async def cluster_async(
    files: list[UploadFile] = File(...),
    threshold: float | None = Form(None),
    backend: Literal["agglomerative", "dbscan"] | None = Form(None),
    x_demo_mode: bool = Header(False),
    _loaded: None = Depends(embedder_loaded),
    _budget: None = Depends(enforce_image_budget),
) -> AsyncSubmitResponse:
    """Submit clustering job — returns ``task_id`` for polling.

    Embedding extraction is CPU-heavy (ArcFace on CPU) and can block the
    event loop for several seconds on large batches, so we run the work
    inside ``asyncio.to_thread`` and report completion via Redis status.
    """
    if threshold is not None and not (0.0 <= threshold <= 2.0):
        raise HTTPException(
            status_code=400,
            detail=ServiceError(
                ErrCode.INVALID_THRESHOLD, "threshold must be in [0,2]"
            ).to_payload(),
        )

    raw = _ingest(files)
    if not raw:
        raise HTTPException(
            status_code=400,
            detail=ServiceError(ErrCode.NO_IMAGES, "no files decoded").to_payload(),
        )

    task_id = await task_store.submit_task({})

    async def runner() -> None:
        task_store.set_state(task_id, "running")
        try:
            result = await asyncio.to_thread(run_cluster, raw, threshold, backend, x_demo_mode)
            await task_store.store_result(task_id, result)
            log.info("task.done", task_id=task_id)
        except ServiceError as exc:
            await task_store.store_failure(task_id, exc.to_payload())
        except Exception as exc:
            await task_store.store_failure(
                task_id,
                ServiceError(ErrCode.INFERENCE_FAILED, str(exc), http_status=500).to_payload(),
            )
            log.exception("task.crashed", task_id=task_id)

    asyncio.create_task(runner())
    return AsyncSubmitResponse(
        task_id=task_id,
        status_url=f"/cluster/async/{task_id}",
        poll_interval_sec=settings.async_poll_interval_sec,
    )


@router.get("/cluster/async/{task_id}", response_model=AsyncStatusResponse)
async def cluster_async_status(task_id: str) -> AsyncStatusResponse:
    status_payload = await task_store.fetch_status(task_id)
    if status_payload is None:
        raise HTTPException(
            status_code=404,
            detail=ServiceError(
                ErrCode.TASK_NOT_FOUND, f"Unknown task_id {task_id}"
            ).to_payload(),
        )
    return AsyncStatusResponse(
        task_id=task_id,
        state=status_payload.get("state", "pending"),
        result=status_payload.get("result"),
        error=status_payload.get("error"),
    )
