"""KnowledgeBaseRepository — data access for the chat / KB feature.

Repository that encapsulates access to the KnowledgeBase singleton,
the same way TeamGraphRepository wraps the KG provider. The chat
service, the KB REST route and the GraphQL KB resolvers all go
through this repository instead of touching the concrete
KnowledgeBase class — so callers depend on a data-access interface
and the KBase can be swapped/mocked without touching consumers.
"""

from typing import List, Optional


class KnowledgeBaseRepository:
    """Wraps the global KnowledgeBase singleton behind a repo interface."""

    def __init__(self, knowledge_base=None):
        # Defaults to the singleton built in api.dependencies.init_knowledge_base
        if knowledge_base is None:
            from api.dependencies import get_knowledge_base
            knowledge_base = get_knowledge_base()
        self._kb = knowledge_base

    # ── Retrieval & Q&A ───────────────────────────────────────────────────────

    def retrieve(self, question: str, prefer_prediction: bool = False):
        """Retrieve the KB context bundle (facts/tables/sources) — no LLM."""
        return self._kb.retrieve(question, prefer_prediction=prefer_prediction)

    def ask(self, question: str, llm_name: Optional[str] = None,
            prefer_prediction: bool = False, memory: Optional[str] = None):
        """Ask the KB — structured answer (no LLM) or narrated answer."""
        return self._kb.ask(question, llm_name=llm_name,
                            prefer_prediction=prefer_prediction,
                            memory=memory)

    # ── Backing stores ────────────────────────────────────────────────────────

    def head_to_head(self, team_a: str, team_b: str, limit: int = 10,
                     league: Optional[str] = None,
                     season: Optional[str] = None) -> List[dict]:
        """Past head-to-head fixtures between two teams."""
        store = self._kb._store
        return store.head_to_head(team_a, team_b, limit=limit,
                                  league=league, season=season)
