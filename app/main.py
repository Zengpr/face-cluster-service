"""FastAPI application factory."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from app.api import cluster as cluster_api
from app.api import meta as meta_api
from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.core.metrics import exposition


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    log = get_logger("app")
    log.info(
        "app.boot",
        env=settings.env,
        detector=settings.detector_name,
        backend=settings.clustering_backend,
    )
    yield
    log.info("app.shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Face Cluster Service",
        description=(
            "Containerized face-clustering API. Upload N images, "
            "receive an identity-preserving grouping based on "
            "InsightFace ArcFace embeddings and single-linkage "
            "approximate-agglomerative clustering."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_allow_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(cluster_api.router, tags=["Cluster"])
    app.include_router(meta_api.router, tags=["Meta"])
    app.add_route("/metrics", _metrics, methods=["GET"])
    return app


async def _metrics(request=None) -> PlainTextResponse:
    return PlainTextResponse(
        exposition(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


app = create_app()
