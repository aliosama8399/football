"""
Reranker Module (Phase 2)
==========================
Reranks initial vector/hybrid candidate documents using cross-encoder scoring
or token-overlap heuristic fallback.

Usage:
    from rag.retrieval.reranker import rerank_documents
    reranked = rerank_documents(query, candidates, top_k=5)
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_CROSS_ENCODER = None
_ATTEMPTED_LOAD = False


def _get_cross_encoder():
    global _CROSS_ENCODER, _ATTEMPTED_LOAD
    if _ATTEMPTED_LOAD:
        return _CROSS_ENCODER
    _ATTEMPTED_LOAD = True
    try:
        from sentence_transformers import CrossEncoder
        # Use small cross-encoder if installed/available locally
        _CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256)
        logger.info("CrossEncoder loaded for RAG reranking.")
    except Exception as e:
        logger.debug("CrossEncoder not available (%s); using heuristic reranker.", e)
        _CROSS_ENCODER = None
    return _CROSS_ENCODER


def rerank_documents(query: str, docs: list[dict], top_k: int = 5) -> list[dict]:
    """
    Reranks documents in place or returns top_k.
    """
    if not docs:
        return []

    model = _get_cross_encoder()
    if model:
        try:
            pairs = [[query, d.get("text", "")] for d in docs]
            scores = model.predict(pairs)
            for d, s in zip(docs, scores):
                d["rerank_score"] = float(s)
            docs.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            return docs[:top_k]
        except Exception as e:
            logger.warning("CrossEncoder prediction failed: %s; falling back", e)

    # Heuristic fallback: exact keyword overlap density
    q_words = set(query.lower().split())
    for d in docs:
        text = d.get("text", "").lower()
        overlap = sum(1 for w in q_words if w in text)
        d["rerank_score"] = overlap / max(1, len(q_words))

    docs.sort(key=lambda x: (x.get("rerank_score", 0), -x.get("score", 0)), reverse=True)
    return docs[:top_k]
