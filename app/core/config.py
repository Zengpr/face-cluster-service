"""Application configuration loaded from env / .env file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "face-cluster-service"
    env: Literal["dev", "staging", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    detector_name: str = "buffalo_l"
    ctx_id: int = -1
    det_size: int = 640
    min_face_area_pct: float = Field(0.02, ge=0.0, le=1.0)

    clustering_backend: Literal["agglomerative", "dbscan"] = "agglomerative"
    default_threshold: float = Field(0.6, ge=0.0, le=2.0)
    min_samples_for_cluster: int = Field(2, ge=1)
    min_cluster_size: int = Field(1, ge=1)

    max_images_per_request: int = Field(64, ge=1, le=512)
    max_image_bytes: int = Field(15 * 1024 * 1024, ge=1024)
    allowed_content_types: tuple[str, ...] = (
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp",
    )

    redis_url: str = "redis://localhost:6379/0"
    async_result_ttl_sec: int = 3600
    async_poll_interval_sec: float = 0.5

    prometheus_enabled: bool = True
    cors_allow_origins: tuple[str, ...] = ("*",)

    demo_mode: bool = False
    rate_limit_per_minute: int = Field(60, ge=0)

    models_root: Path = Path("/root/.insightface/models")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
