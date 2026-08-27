# ══════════════════════════════════════════════════════════════════════════
# Football Analysis API — CPU or GPU image
#
#   docker build --build-arg GPU=0 -t football-api-cpu .
#   docker build --build-arg GPU=1 -t football-api-gpu .
#   (docker-compose.yml passes GPU=1 by default)
#
# GPU=1 requires: NVIDIA driver + nvidia-container-toolkit on the host, and
# `gpus: all` on the service (set in docker-compose.yml).
#
# Wheel pairing (same as dev machine, proven working):
#   onnxruntime-gpu 1.26.0  +  onnxruntime-genai-cuda 0.14.1  +  CUDA 12.6
# ══════════════════════════════════════════════════════════════════════════

ARG GPU=1

# ── Stage 1: dependency resolution (always CPU base — fast, cached) ──────────
FROM python:3.12-slim AS deps

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app
COPY pyproject.toml ./ 
COPY uv.lock* ./
RUN uv sync --frozen --no-dev --no-install-project \
    || uv sync --no-dev --no-install-project

# ── Stage 2: runtime (stage NAMES must match the GPU arg values: 1 | 0) ──────
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04 AS runtime-1
FROM python:3.12-slim AS runtime-0
FROM runtime-${GPU} AS runtime

# re-declare so $GPU is visible to RUN instructions inside this stage
ARG GPU=1

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf_cache \
    VIRTUAL_ENV=/app/.venv \
    UV_HTTP_TIMEOUT=600 \
    UV_HTTP_RETRIES=10 \
    LD_LIBRARY_PATH=/app/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/app/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:/app/.venv/lib/python3.12/site-packages/nvidia/cufft/lib:/app/.venv/lib/python3.12/site-packages/nvidia/curand/lib:/app/.venv/lib/python3.12/site-packages/nvidia/nvjitlink/lib:/usr/local/cuda/lib64

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl tini ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# GPU image has no system python → provision 3.12 via uv
RUN if command -v python3.12 >/dev/null 2>&1 || command -v python3 >/dev/null 2>&1; then \
        echo "system python present"; \
    else \
        uv python install 3.12; \
    fi

COPY pyproject.toml ./
COPY uv.lock* ./

# ── CPU deps (cached as its own layer + uv cache mount: big downloads live ×1) ─
COPY --from=deps /app/.venv /app/.venv
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$GPU" = "1" ]; then \
        rm -rf /app/.venv && \
        (uv venv --python 3.12 /app/.venv || uv venv --python "$(uv python find 3.12)" /app/.venv) && \
        uv sync --frozen --no-dev; \
    fi

# ── CUDA wheel swap (cache mount ⇒ retries only re-download what failed) ─────
# torch cu126 pulls **all** nvidia-* runtime libs (cudnn/cublas/cufft/curand/
# nvjitlink/...) transitively — no need to install them explicitly.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$GPU" = "1" ]; then \
        echo '── swapping in CUDA torch ──' && \
        uv pip install --python /app/.venv/bin/python "torch==2.13.0" \
            --index-url https://download.pytorch.org/whl/cu126 --reinstall-package torch && \
        echo '── removing CPU inference wheels ──' && \
        (uv pip uninstall --python /app/.venv/bin/python onnxruntime onnxruntime-genai || true) && \
        echo '── installing GPU inference wheels ──' && \
        uv pip install --python /app/.venv/bin/python "onnxruntime-gpu==1.26.0" && \
        uv pip install --python /app/.venv/bin/python "onnxruntime-genai-cuda==0.14.1" && \
        echo '── forcing onnxruntime-gpu to win (genai-cuda may re-pull CPU ort) ──' && \
        (uv pip uninstall --python /app/.venv/bin/python onnxruntime || true) && \
        uv pip install --reinstall --python /app/.venv/bin/python "onnxruntime-gpu==1.26.0"; \
    fi

# ── Hard verification: build FAILS loudly if CUDA wheels didn't win ──────────
RUN if [ "$GPU" = "1" ]; then \
        /app/.venv/bin/python -c "import onnxruntime as o; p=o.get_available_providers(); print('ORT providers:', p); assert 'CUDAExecutionProvider' in p, 'FATAL: onnxruntime-gpu not active'"; \
        /app/.venv/bin/python -c "import torch; print('torch:', torch.__version__); assert 'cu126' in torch.__version__, 'FATAL: torch is not the CUDA build'"; \
        echo '── CUDA wheel swap VERIFIED ──'; \
    fi

ENV PATH="/app/.venv/bin:$PATH"

# application code (models/export excluded via .dockerignore — mounted at runtime)
COPY . .

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=240s --retries=3 \
    CMD curl -fsS http://localhost:8001/ || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
