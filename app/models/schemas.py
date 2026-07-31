"""Pydantic response/request schemas exposed by the API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"]
    model_loaded: bool
    detector: str
    git_sha: str = ""


class ClusterResponse(BaseModel):
    ok: bool = True
    n_images: int
    n_clusters: int
    n_noise: int
    threshold: float
    backend: str
    silhouette: float
    cluster_sizes: dict[int, int]
    clusters: list[dict]
    label_by_file: dict[str, int]
    dropped_files: list[str]


class AsyncSubmitResponse(BaseModel):
    task_id: str
    status_url: str
    poll_interval_sec: float


class AsyncStatusResponse(BaseModel):
    task_id: str
    state: Literal["pending", "running", "succeeded", "failed"]
    result: dict | None = None
    error: dict | None = None
