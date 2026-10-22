"""
LangChain Retriever Adapters (Phase 2, Task T2.3 Spike)
========================================================
Thin adapter module wrapping Football KG and Vector Providers as standard
LangChain BaseRetriever objects.

Enables standard LangChain / LangGraph chains or RAGAS evaluation pipelines
without altering native domain code. Easy to use or delete.

Usage:
    from rag.retrieval.langchain_adapters import FootballKGRetriever, FootballVectorRetriever
    kg_retriever = FootballKGRetriever()
    docs = kg_retriever.invoke("Arsenal")
"""

import logging
from typing import List, Any, Optional

logger = logging.getLogger(__name__)

# Resilient import of LangChain Document & BaseRetriever
try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False

    class Document:
        def __init__(self, page_content: str, metadata: Optional[dict] = None):
            self.page_content = page_content
            self.metadata = metadata or {}

    class BaseRetriever:
        def invoke(self, query: str) -> list[Document]:
            return self._get_relevant_documents(query)

    CallbackManagerForRetrieverRun = Any


class FootballKGRetriever(BaseRetriever):
    """
    LangChain BaseRetriever wrapping the active KG Provider (Neo4j or PostgreSQL).
    """
    kg_provider: Any = None

    def __init__(self, kg_provider: Any = None, **kwargs):
        super().__init__(**kwargs)
        if kg_provider is None:
            from rag.providers.kg_provider import get_kg_provider
            self.kg_provider = get_kg_provider()
        else:
            self.kg_provider = kg_provider

    def _get_relevant_documents(
        self, query: str, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        from rag.retrieval.team_linker import link_teams
        teams = link_teams(query)

        docs = []
        for team in teams:
            try:
                text = self.kg_provider.format_team_context(team)
                docs.append(Document(page_content=text, metadata={"source": "kg", "team": team}))
            except Exception as e:
                logger.debug("KG retrieval failed for team %s: %s", team, e)

        return docs


class FootballVectorRetriever(BaseRetriever):
    """
    LangChain BaseRetriever wrapping pgvector or FAISS vector search.
    """
    vector_provider: Any = None
    k: int = 5

    def __init__(self, vector_provider: Any = None, k: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        if vector_provider is None:
            from rag.providers.vector_provider import get_vector_provider
            self.vector_provider = get_vector_provider()
        else:
            self.vector_provider = vector_provider

    def _get_relevant_documents(
        self, query: str, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        docs = []
        try:
            results = self.vector_provider.search(query, k=self.k)
            for r in results:
                docs.append(Document(
                    page_content=r.get("text", ""),
                    metadata=r.get("metadata", {})
                ))
        except Exception as e:
            logger.debug("Vector search error in LangChain adapter: %s", e)

        return docs
