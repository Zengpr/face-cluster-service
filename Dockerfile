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

ARG APT_MIRROR=https://mirrors.aliyun.com/debian
RUN set -eux; \
    if [ "${APT_MIRROR}" != "DISABLED" ]; then \
      sed -i "s|http://deb.debian.org/debian|${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources 2>/dev/null || true; \
      sed -i "s|http://deb.debian.org/debian|${APT_MIRROR}|g" /etc/apt/sources.list 2>/dev/null || true; \
    fi; \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        wget \
        unzip \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple \
    && pip config set global.timeout 60 \
    && pip install --upgrade pip \
    && pip install --prefix=/install -r requirements.txt

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

# onnxruntime 1.16's pybind11_state.so requires executable stack which
# modern Debian kernels reject. Pre-apply ``execsetstatus`` via a tiny
# ELF header patcher so import does not throw at runtime.
ARG APT_MIRROR=https://mirrors.aliyun.com/debian
RUN set -eux; \
    if [ "${APT_MIRROR}" != "DISABLED" ]; then \
      sed -i "s|http://deb.debian.org/debian|${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources 2>/dev/null || true; \
      sed -i "s|http://deb.debian.org/debian|${APT_MIRROR}|g" /etc/apt/sources.list 2>/dev/null || true; \
    fi; \
    apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        curl \
        tini \
        unzip \
    && rm -rf /var/lib/apt/lists/*

# onnxruntime 1.16 / 1.17 ships a pybind .so with PT_GNU_STACK=RWE.
# Modern Debian kernels (with `mprotect` enforcing NX) reject that and
# raise ImportError "cannot enable executable stack". We flip the X
# bit off so the loader does NOT try to mark the stack executable.
# This MUST run AFTER copying the builder site-packages layer.
COPY --from=builder /install /usr/local
COPY scripts/patch_execstack.py /usr/local/bin/patch_execstack.py
RUN python /usr/local/bin/patch_execstack.py || echo "execstack patcher found nothing"

COPY app ./app
COPY scripts ./scripts

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

# Pre-download buffalo_l at build time (best-effort, falls back to stub).
COPY scripts/download_model.sh /usr/local/bin/download_model.sh
COPY scripts/start.sh /app/scripts/start.sh
RUN chmod +x /usr/local/bin/download_model.sh /app/scripts/start.sh \
    && /usr/local/bin/download_model.sh true

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/app/scripts/start.sh"]
