"""
RAG Provider Package
====================
Provider-based architecture for KG and Vector retrieval.

Usage:
    from rag.providers import get_kg_provider, get_vector_provider

    kg     = get_kg_provider("neo4j")
    vector = get_vector_provider("faiss")
    kg     = get_kg_provider("postgres")  # Switch to Postgres
"""

from rag.providers.kg_provider import get_kg_provider
from rag.providers.vector_provider import get_vector_provider

__all__ = ["get_kg_provider", "get_vector_provider"]
