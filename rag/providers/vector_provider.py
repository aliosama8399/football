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
        self._nprobe       = int(cfg.get("faiss_nprobe", 16))

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
        # Post-hoc metadata filtering needs a broader candidate pool than the
        # default nprobe=1 cell, otherwise selective filters return nothing.
        if hasattr(self._index, "nprobe"):
            self._index.nprobe = self._nprobe
        self._embedder = SentenceTransformer(self.embed_model)

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)

        logger.info(
            "FAISS index loaded: %d vectors | embed_model=%s | nprobe=%d",
            self._index.ntotal, self.embed_model,
            getattr(self._index, "nprobe", 1),
        )

    # ── Search ─────────────────────────────────────────────────────────────
    def search(self, query: str, k: int = 5, filter_meta: dict = None,
               as_of: str = None) -> list[dict]:
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

        # Retrieve more candidates if filtering is requested (post-hoc filter
        # pool needs to be large enough to survive selective filters).
        filtered = bool(filter_meta or as_of)
        fetch_k = k * 50 if filtered else k
        fetch_k = max(fetch_k, 300) if filtered else fetch_k
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
            if as_of and meta.get("date") and str(meta["date"]) >= as_of:
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
# pgvector Provider (vectors live inside PostgreSQL)
# ─────────────────────────────────────────────────────────────────────────────

class PgVectorProvider(BaseVectorProvider):
    """
    Vector search using the pgvector extension (cosine distance, HNSW index).
    Requires that build_pgvector_index.py has been run (or startup seeding).
    """

    def __init__(self, dsn: str = None, table: str = None):
        import os
        cfg = _load_rag_cfg().get("rag", {})
        self.dsn = dsn or os.getenv("POSTGRES_DSN") or cfg.get(
            "postgres_dsn", "postgresql://postgres:123@localhost:5432/football_rag")
        self.table = table or cfg.get("pgvector_table", "kb_vectors")
        self.embed_model = cfg.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")
        self._conn = None
        self._embedder = None
        self._faiss_fallback = None

    def load(self) -> None:
        import psycopg2
        from sentence_transformers import SentenceTransformer

        self._conn = psycopg2.connect(self.dsn)
        self._conn.autocommit = True
        self._embedder = SentenceTransformer(self.embed_model)
        size = self.get_index_size()
        logger.info("pgvector connected: table=%s | vectors=%d | embed_model=%s",
                    self.table, size, self.embed_model)
        if size == 0:
            logger.warning(
                "pgvector table '%s' has 0 vectors! Run `python rag/build_pgvector_index.py` "
                "to populate it, or start the server to auto-seed. Using FAISS fallback.",
                self.table
            )

    def search(self, query: str, k: int = 5, filter_meta: dict = None,
               as_of: str = None) -> list[dict]:
        if self._conn is None:
            self.load()

        if self.get_index_size() == 0:
            if self._faiss_fallback is None:
                try:
                    self._faiss_fallback = FAISSProvider()
                    self._faiss_fallback.load()
                except Exception:
                    self._faiss_fallback = False
            if self._faiss_fallback:
                return self._faiss_fallback.search(query, k=k, filter_meta=filter_meta, as_of=as_of)

        q_vec = self._embedder.encode([query], normalize_embeddings=True)[0]
        literal = "[" + ",".join(f"{x:.6f}" for x in q_vec) + "]"

        where, params = [], []
        if filter_meta:
            where.append("metadata @> %s::jsonb")
            params.append(json.dumps(filter_meta))
        if as_of:
            where.append("(metadata->>'date') IS NULL OR (metadata->>'date') < %s")
            params.append(as_of)
        clause = ("WHERE " + " AND ".join(f"({w})" for w in where)) if where else ""

        sql = (f"SELECT text, metadata, embedding <=> %s::vector AS dist "
               f"FROM {self.table} {clause} ORDER BY dist LIMIT %s")
        with self._conn.cursor() as cur:
            cur.execute(sql, [literal, *params, k])
            rows = cur.fetchall()
        return [{"text": t, "score": float(d), "metadata": m} for t, m, d in rows]

    def get_index_size(self) -> int:
        if self._conn is None:
            self.load()
        with self._conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", (self.table,))
            if cur.fetchone()[0] is None:
                return 0
            cur.execute(f"SELECT COUNT(*) FROM {self.table}")
            return cur.fetchone()[0]


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

_VECTOR_REGISTRY = {
    "faiss": FAISSProvider,
    "pgvector": PgVectorProvider,
}


def get_vector_provider(name: str = None) -> BaseVectorProvider:
    """
    Return a loaded vector provider.

    Args:
        name: "faiss" | "pgvector". Defaults to llm_config.yaml rag.vector_provider.
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
