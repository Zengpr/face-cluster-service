# syntax=docker/dockerfile:1.6
# ---------------------------------------------------------------------------
# Multi-stage build: wheel builder -> runtime image
# ---------------------------------------------------------------------------
ARG PYTHON_VERSION=3.11-slim
ARG ONNXRUNTIME_WHEEL=onnxruntime-1.16.3-cp311-cp311-manylinux_2_28_x84_64.whl

FROM python:${PYTHON_VERSION} AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        wget \
        unzip \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-deps -r requirements.txt \
    && for dep in $(grep -v '^#' requirements.txt | grep -v '^$'); do \
         echo "Resolving transitive deps for $dep"; \
         pip install --prefix=/install "$dep" 2>&1 | tail -1 ; \
       done

# ---------------------------------------------------------------------------
# Runtime layer
FROM python:${PYTHON_VERSION} AS runtime

LABEL org.opencontainers.image.title="face-cluster-service" \
      org.opencontainers.image.description="InsightFace ArcFace + single-linkage clustering API" \
      org.opencontainers.image.source="https://github.com/Zengpr/face-cluster-service"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    HF_HOME=/tmp/.hf \
    INSIGHTFACE_HOME=/root/.insightface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY app ./app
COPY scripts ./scripts

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

# Pre-download buffalo_l on container start so first request stays hot.
COPY scripts/download_model.sh /usr/local/bin/download_model.sh
RUN chmod +x /usr/local/bin/download_model.sh

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/download_model.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
