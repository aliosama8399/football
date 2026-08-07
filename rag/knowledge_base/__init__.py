"""
Knowledge base package — provider-agnostic facts layer for the chat module.

    MatchDataStore     — lazy CSV (structured facts, standings, H2H, form)
    TeamProfileStore   — team info/stats (PostgreSQL via TeamGraphRepository,
                         CSV + tactics fallback, TTL cache)
    IntentClassifier   — rule-based intent/parameter routing
    ContextBundle      — retrieval contract (facts + tables + sources)

Consumers: chat service, REST /api/v1/kb/* and GraphQL kbRetrieve. The package
must stay importable WITHOUT an LLM, vector index, or running PostgreSQL —
degradation is handled per call.
"""

from rag.knowledge_base.config import kb_settings
from rag.knowledge_base.datastore import MatchDataStore
from rag.knowledge_base.team_store import TeamProfileStore
from rag.knowledge_base.intents import IntentClassifier, IntentResult
from rag.knowledge_base.context import ContextBundle, SourceRef

__all__ = [
    "kb_settings", "MatchDataStore", "TeamProfileStore",
    "IntentClassifier", "IntentResult", "ContextBundle", "SourceRef",
]
