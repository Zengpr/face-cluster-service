"""FastAPI application factory."""
from __future__ import annotations

import signal
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from app.api import cluster as cluster_api
from app.api import meta as meta_api
from app.core.config import settings
from app.core.errors import ServiceError
from app.core.logging import get_logger, setup_logging
from app.core.metrics import exposition
from app.core.middleware import RequestIDMiddleware
from app.core.rate_limit import rate_limiter


_shutting_down = False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    log = get_logger("app")
    log.info(
        "app.boot",
        env=settings.env,
        detector=settings.detector_name,
        backend=settings.clustering_backend,
        demo_mode=settings.demo_mode,
    )
    yield
    log.info("app.shutdown")


def _handle_sigterm(*_) -> None:
    global _shutting_down
    _shutting_down = True
    get_logger("app").info("sigterm_received")


signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigterm)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Face Cluster Service",
        description=(
            "Containerized face-clustering API. Upload N images, "
            "receive an identity-preserving grouping based on "
            "InsightFace ArcFace embeddings and single-linkage "
            "approximate-agglomerative clustering."
        ),
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(RequestIDMiddleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_allow_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _shutdown_guard(request: Request, call_next):
        if _shutting_down:
            return JSONResponse(
                status_code=503,
                content={"detail": {"error": {"code": 5030, "name": "SHUTTING_DOWN", "message": "Server is shutting down"}}},
            )
        return await call_next(request)

    @app.middleware("http")
    async def _rate_limit(request: Request, call_next):
        if request.url.path in ("/health", "/ready", "/metrics"):
            return await call_next(request)
        client_ip = request.client.host if request.client else "unknown"
        if not rate_limiter.is_allowed(client_ip):
            return JSONResponse(
                status_code=429,
                content={"detail": {"error": {"code": 4290, "name": "RATE_LIMITED", "message": "Too many requests"}}},
                headers={"Retry-After": "60"},
            )
        return await call_next(request)

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
