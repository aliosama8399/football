"""
Langfuse Client & Tracing Helper (Phase 1)
===========================================
Traces RAG steps and logs evaluation scores to Langfuse (Cloud or self-hosted).

Supports Langfuse Python SDK v3 (start_as_current_span) and v4
(start_as_current_observation). Reads keys from the project .env:

    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_HOST=https://cloud.langfuse.com   # or https://us.cloud.langfuse.com
    LANGFUSE_ENABLED=true                      # set false to turn tracing off

When keys / package are missing every helper is a silent no-op, so the app
never breaks because of tracing. Use `langfuse_status()` to see why.
"""

import os
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).parent.parent / ".env")

_CLIENT = None
_INITIALIZED = False
_STATUS = "not initialized"


class _NoOpSpan:
    def update(self, *args, **kwargs):
        pass

    def score(self, *args, **kwargs):
        pass


_NOOP = _NoOpSpan()


def get_langfuse():
    """Return an authenticated Langfuse client, or None (see langfuse_status())."""
    global _CLIENT, _INITIALIZED, _STATUS
    if _INITIALIZED:
        return _CLIENT
    _INITIALIZED = True

    if os.getenv("LANGFUSE_ENABLED", "true").lower() in ("0", "false", "no"):
        _STATUS = "disabled by LANGFUSE_ENABLED=false"
        return None
    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        _STATUS = "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY missing in .env"
        return None

    host = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com"
    os.environ.setdefault("LANGFUSE_HOST", host)
    os.environ.setdefault("LANGFUSE_BASE_URL", host)

    try:
        from langfuse import get_client
    except ImportError:
        _STATUS = "package not installed (pip install langfuse)"
        return None

    try:
        client = get_client()
        if not client.auth_check():
            _STATUS = f"auth_check failed for {host} (wrong keys or region?)"
            return None
    except Exception as e:
        _STATUS = f"connection error to {host}: {e}"
        return None

    _CLIENT = client
    _STATUS = f"connected to {host}"
    return _CLIENT


def langfuse_status() -> str:
    get_langfuse()
    return _STATUS


@contextmanager
def trace_step(name: str, metadata: Optional[dict] = None, input=None, as_type: str = "span", model: Optional[str] = None):
    """Open a span or generation; nested calls become child observations of the current trace."""
    client = get_langfuse()
    if client is None:
        yield _NOOP
        return

    kwargs = {"name": name, "input": input, "metadata": metadata}
    if model:
        kwargs["model"] = model

    try:
        if hasattr(client, "start_as_current_observation"):
            cm = client.start_as_current_observation(as_type=as_type, **kwargs)
        else:
            cm = client.start_as_current_span(**kwargs)
    except Exception as e:
        logger.debug("Langfuse span '%s' not created: %s", name, e)
        yield _NOOP
        return

    with cm as span:
        yield span


def current_trace_id() -> Optional[str]:
    client = get_langfuse()
    if client is None:
        return None
    try:
        return client.get_current_trace_id()
    except Exception:
        return None


def log_score(name: str, value: float, comment: Optional[str] = None, trace_id: Optional[str] = None):
    """Attach a numeric score to a trace (or to the current trace if none given)."""
    client = get_langfuse()
    if client is None:
        return
    try:
        if trace_id:
            client.create_score(name=name, value=float(value), trace_id=trace_id, comment=comment)
        else:
            client.score_current_trace(name=name, value=float(value), comment=comment)
    except Exception as e:
        logger.debug("Failed to log score '%s': %s", name, e)


def flush_traces():
    client = get_langfuse()
    if client is not None:
        try:
            client.flush()
        except Exception:
            pass
