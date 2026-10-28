# ══════════════════════════════════════════════════════════════════════════
# Football Analysis API — GPU build (single-stage, minimal)
#
# Decisions:
#   * GPU-only (compose requests `gpus: all`; host needs NVIDIA driver + toolkit)
#   * Single stage: wheels are installed into /app/.venv ONCE (no copy/remove
#     duplication that was doubling the image size)
#   * CUDA runtime comes from pip nvidia-* wheels (bundled via torch cu126 deps)
#     → base image stays python:3.12-slim (no multi-GB CUDA base needed);
#     GPU access is injected by nvidia-container-toolkit at runtime.
#   * uv HTTP retry/timeout hardening for the flaky pypi.nvidia.com index,
#     plus a cache mount so failed builds resume where they left off.
#   * Fails the build loudly if CUDA wheels didn't win.
# ══════════════════════════════════════════════════════════════════════════

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf_cache \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_HTTP_TIMEOUT=600 \
    UV_HTTP_RETRIES=10 \
    LD_LIBRARY_PATH=/app/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/app/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:/app/.venv/lib/python3.12/site-packages/nvidia/cufft/lib:/app/.venv/lib/python3.12/site-packages/nvidia/curand/lib:/app/.venv/lib/python3.12/site-packages/nvidia/nvjitlink/lib

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl tini ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Layer 1: base dependency set (cached; downloads land in the uv cache) ────
COPY pyproject.toml uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# ── Layer 2: CUDA torch swap (heavy ~2.5 GB) ─────────────────────────────────
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /app/.venv/bin/python "torch==2.13.0" \
        --index-url https://download.pytorch.org/whl/cu126 --reinstall-package torch

# ── Layer 3: GPU inference wheels (onnxruntime-gpu + genai-cuda) ─────────────
# Ordering matters: genai-cuda may re-declare a CPU `onnxruntime` dependency.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /app/.venv/bin/python "onnxruntime-gpu==1.26.0" && \
    uv pip install --python /app/.venv/bin/python "onnxruntime-genai-cuda==0.14.1" 


# ── Layer 4: hard verification (no silent CPU fallbacks) ─────────────────────
RUN /app/.venv/bin/python -c "import onnxruntime as o; p=o.get_available_providers(); \
    print('ORT providers:', p); assert 'CUDAExecutionProvider' in p, 'FATAL: onnxruntime-gpu not active'" \
 && /app/.venv/bin/python -c "import torch; print('torch:', torch.__version__); \
    assert 'cu126' in torch.__version__, 'FATAL: torch is not the CUDA build'"

# ── Application code (big artifacts excluded by .dockerignore) ────────────────
COPY . .
RUN sed -i 's/\r$//' /app/entrypoint.sh && chmod +x /app/entrypoint.sh

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=240s --retries=3 \
    CMD curl -fsS http://localhost:8001/ || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/app/entrypoint.sh"]
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
