"""
FAISS Vector Provider
======================
Implements BaseVectorProvider using FAISS + JSON metadata sidecar.

Index file  : rag/vector_store/faiss.index
Metadata    : rag/vector_store/faiss_metadata.json
Embeddings  : sentence-transformers/all-MiniLM-L6-v2 (local, ~80MB)

Factory:
    from rag.providers.vector_provider import get_vector_provider
    vector = get_vector_provider("faiss")
    results = vector.search("Arsenal attacking tactics", k=5)
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional

import yaml

from rag.providers.base import BaseVectorProvider

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
_CFG_PATH = Path(__file__).parent.parent.parent / "models" / "llm_config.yaml"
_BASE_DIR  = Path(__file__).parent.parent


def _load_rag_cfg() -> dict:
    if not _CFG_PATH.exists():
        return {}
    with open(_CFG_PATH) as f:
        return yaml.safe_load(f) or {}


# ─────────────────────────────────────────────────────────────────────────────
# FAISS Provider
# ─────────────────────────────────────────────────────────────────────────────

class FAISSProvider(BaseVectorProvider):
    """
    Vector search provider using FAISS.
    Requires that build_faiss_index.py has been run first.
    """

    def __init__(self, index_path: str = None, metadata_path: str = None):
        cfg = _load_rag_cfg().get("rag", {})
        self.index_path    = Path(index_path    or cfg.get("faiss_index_path",    _BASE_DIR / "vector_store" / "faiss.index"))
        self.metadata_path = Path(metadata_path or cfg.get("faiss_metadata_path", _BASE_DIR / "vector_store" / "faiss_metadata.json"))
        self.embed_model   = cfg.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")

        self._index     = None
        self._metadata  = []   # list[dict] – one dict per vector
        self._embedder  = None

    # ── Load ───────────────────────────────────────────────────────────────
    def load(self) -> None:
        import faiss
        from sentence_transformers import SentenceTransformer

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {self.index_path}\n"
                "Run `python rag/build_faiss_index.py` first."
            )

        self._index    = faiss.read_index(str(self.index_path))
        self._embedder = SentenceTransformer(self.embed_model)

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)

        logger.info(
            "FAISS index loaded: %d vectors | embed_model=%s",
            self._index.ntotal, self.embed_model,
        )

    # ── Search ─────────────────────────────────────────────────────────────
    def search(self, query: str, k: int = 5, filter_meta: dict = None) -> list[dict]:
        """
        Semantic search against the FAISS index.

        Args:
            query       : Natural language query string.
            k           : Number of results to return.
            filter_meta : Optional dict for post-hoc metadata filtering
                          e.g. {"league": "Premier_League", "season": "2425"}

        Returns:
            List of dicts with keys: text, score, metadata.
        """
        if self._index is None:
            self.load()

        # Embed the query
        q_vec = self._embedder.encode([query], normalize_embeddings=True).astype("float32")

        # Retrieve more candidates if filtering is requested
        fetch_k = k * 5 if filter_meta else k
        fetch_k = min(fetch_k, self._index.ntotal)

        distances, indices = self._index.search(q_vec, fetch_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue

            meta = self._metadata[idx]

            # Post-hoc metadata filtering
            if filter_meta:
                if not all(meta.get(k) == v for k, v in filter_meta.items()):
                    continue

            results.append({
                "text":     meta.get("text", ""),
                "score":    float(dist),
                "metadata": {k: v for k, v in meta.items() if k != "text"},
            })

            if len(results) >= k:
                break

        return results

    def get_index_size(self) -> int:
        """Return number of vectors in the index."""
        if self._index is None:
            self.load()
        return self._index.ntotal


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

_VECTOR_REGISTRY = {
    "faiss": FAISSProvider,
}


def get_vector_provider(name: str = None) -> BaseVectorProvider:
    """
    Return a loaded vector provider.

    Args:
        name: "faiss". Defaults to llm_config.yaml rag.vector_provider.
    """
    if name is None:
        cfg  = _load_rag_cfg()
        name = cfg.get("rag", {}).get("vector_provider", "faiss")

    name = name.lower()
    if name not in _VECTOR_REGISTRY:
        raise ValueError(f"Unknown vector provider '{name}'. Choose from: {list(_VECTOR_REGISTRY)}")

    provider = _VECTOR_REGISTRY[name]()
    provider.load()
    return provider
