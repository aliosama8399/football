"""
Hybrid Retriever (BM25 + Dense Vector with Reciprocal Rank Fusion)
==================================================================
Combines keyword matching (BM25) with semantic embeddings (pgvector / FAISS)
to achieve high precision on exact entities while preserving semantic recall.

Usage:
    from rag.retrieval.hybrid import HybridRetriever
    retriever = HybridRetriever(vector_provider)
    results = retriever.search("Arsenal low block defense against counter-attacks", k=5, as_of="2024-08-01")
"""

import math
import logging
from typing import Optional, Any
from collections import Counter

logger = logging.getLogger(__name__)


def compute_rrf(rank_lists: list[list[dict]], k_constant: int = 60) -> list[dict]:
    """
    Reciprocal Rank Fusion (RRF) to merge multiple ranked result lists.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for ranked_docs in rank_lists:
        for rank, doc in enumerate(ranked_docs, 1):
            text = doc.get("text", "")
            if not text:
                continue
            doc_map[text] = doc
            scores[text] = scores.get(text, 0.0) + (1.0 / (k_constant + rank))

    sorted_texts = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)
    fused_results = []
    for text in sorted_texts:
        res = dict(doc_map[text])
        res["score"] = scores[text]
        fused_results.append(res)

    return fused_results


class SimpleBM25:
    """Lightweight in-memory BM25 scorer when rank_bm25 is not installed."""
    def __init__(self, corpus: list[str]):
        self.corpus = corpus
        self.doc_len = [len(d.lower().split()) for d in corpus]
        self.avg_dl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 1.0
        self.k1 = 1.5
        self.b = 0.75
        self.doc_freqs: dict[str, int] = Counter()
        for doc in corpus:
            tokens = set(doc.lower().split())
            for t in tokens:
                self.doc_freqs[t] += 1
        self.n_docs = len(corpus)

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        q_tokens = query.lower().split()
        scores = []
        for idx, doc in enumerate(self.corpus):
            d_tokens = Counter(doc.lower().split())
            score = 0.0
            dl = self.doc_len[idx]
            for t in q_tokens:
                if t in d_tokens:
                    df = self.doc_freqs.get(t, 0)
                    idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
                    tf = d_tokens[t]
                    num = tf * (self.k1 + 1)
                    denom = tf + self.k1 * (1 - self.b + self.b * (dl / self.avg_dl))
                    score += idf * (num / denom)
            if score > 0:
                scores.append((idx, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class HybridRetriever:
    """
    Hybrid retriever orchestrating Dense + Sparse retrieval.
    """
    def __init__(self, dense_provider: Any):
        self.dense = dense_provider

    def search(self, query: str, k: int = 5, as_of: Optional[str] = None, filter_meta: Optional[dict] = None) -> list[dict]:
        # 1. Dense vector search
        dense_results = []
        try:
            if hasattr(self.dense, "search"):
                dense_results = self.dense.search(query, k=k * 3, filter_meta=filter_meta, as_of=as_of)
        except Exception as e:
            logger.warning("Dense retrieval error in hybrid search: %s", e)

        if not dense_results:
            return []

        # 2. Sparse / BM25 re-ranking of dense candidates pool
        candidate_texts = [r.get("text", "") for r in dense_results]
        bm25 = SimpleBM25(candidate_texts)
        sparse_ranked_indices = bm25.search(query, top_k=len(candidate_texts))

        sparse_results = [dense_results[idx] for idx, _ in sparse_ranked_indices]

        # 3. Fuse with RRF
        fused = compute_rrf([dense_results, sparse_results])
        return fused[:k]
