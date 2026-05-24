"""
Football RAG System
====================
Hybrid Knowledge Graph + Vector retrieval with provider-based architecture.

Quick start:
    from rag import FootballRAGSystem

    rag    = FootballRAGSystem()                          # reads llm_config.yaml
    rag_g  = FootballRAGSystem(llm="gemini")              # switch to Gemini
    rag_pg = FootballRAGSystem(kg="postgres", llm="ollama")  # PostgreSQL + Ollama

    answer = rag.query("How does Arsenal attack?")
    answer = rag.predict_match("Arsenal", "Chelsea")
    answer = rag.get_team_report("Liverpool")
    answer = rag.compare_teams("Barcelona", "Real Madrid")
"""

from rag.rag_orchestrator import FootballRAGSystem
from rag.providers         import get_kg_provider, get_vector_provider

__all__ = ["FootballRAGSystem", "get_kg_provider", "get_vector_provider"]
