
from api.repositories import KnowledgeBaseRepository
from api.repositories import Best11Repository
from api.services.best11_service import Best11ApiService
from api.services.scout_service import ScoutApiService
from api.repositories.scout_repo import ScoutRepository
from api.services.scout_service import ScoutApiService
from rag.knowledge_base.kb import KnowledgeBase
import logging
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends, HTTPException, Request

from api.database import AsyncSessionLocal
from rag.rag_orchestrator import FootballRAGSystem

from api.async_rag import AsyncRAGWrapper

logger = logging.getLogger(__name__)

# Global singleton RAG System instance (fallback / cli / scripts)
_rag_system: Optional[FootballRAGSystem] = None
_async_rag: Optional[AsyncRAGWrapper] = None
_kb: Optional["KnowledgeBase"] = None
_kb_repo: Optional["KnowledgeBaseRepository"] = None
_best11_repo: Optional["Best11Repository"] = None
_best11_api_service: Optional["Best11ApiService"] = None
_scout_repo: Optional["ScoutRepository"] = None
_scout_api_service: Optional["ScoutApiService"] = None

def init_rag_system() -> FootballRAGSystem:
    """
    Initialize the global singleton RAG System instance on app startup.
    Tolerates LLM-config failure by retrying with llm="none" so the API
    still starts (returns structured predictions without narrative).
    """
    global _rag_system
    if _rag_system is None:
        logger.info("Initializing global FootballRAGSystem...")
        try:
            _rag_system = FootballRAGSystem()
            logger.info("Global FootballRAGSystem initialized successfully.")
        except Exception as e:
            logger.warning("RAG init failed with default LLM (%s). Retrying with llm='none' (GNN only)...", e)
            try:
                _rag_system = FootballRAGSystem(llm="none")
                logger.info("FootballRAGSystem initialized with llm='none' (graceful degradation).")
            except Exception as e2:
                logger.critical("RAG System init failed even with LLM disabled: %s", e2, exc_info=True)
                raise
    return _rag_system

def get_rag_system(request: Request) -> FootballRAGSystem:
    """
    Dependency that returns the active FootballRAGSystem instance,
    preferring app.state when called inside a FastAPI request context.
    """
    if hasattr(request, "app") and hasattr(request.app.state, "rag_system"):
        return request.app.state.rag_system
    global _rag_system
    if _rag_system is None:
        return init_rag_system()
    return _rag_system

def get_async_rag(request: Request,
                  rag_sys: FootballRAGSystem = Depends(get_rag_system)) -> AsyncRAGWrapper:
    """
    Dependency that returns the ThreadPoolExecutor wrapped AsyncRAGWrapper,
    preferring app.state when available.
    """
    if hasattr(request, "app") and hasattr(request.app.state, "async_rag"):
        return request.app.state.async_rag
    global _async_rag
    if _async_rag is None:
        _async_rag = AsyncRAGWrapper(rag_sys)
    return _async_rag

# ── KnowledgeBase (chat-KB facade) ────────────────────────────────────────────

def init_knowledge_base() -> "KnowledgeBase":
    """
    Initialize the global KnowledgeBase singleton on app startup.
    Cheap: construction is lazy — CSV/Postgres/FAISS/GNN load only on first use.
    """
    global _kb
    if _kb is None:
        from rag.knowledge_base.kb import KnowledgeBase
        _kb = KnowledgeBase()
        logger.info("KnowledgeBase singleton initialized (lazy internals).")
    return _kb

def get_knowledge_base() -> "KnowledgeBase":
    """
    Dependency that returns the global KnowledgeBase instance.
    """
    global _kb
    if _kb is None:
        raise RuntimeError("KnowledgeBase not initialized. Call init_knowledge_base() on startup.")
    return _kb

def get_kb_repo() -> "KnowledgeBaseRepository":
    """
    Inject the KnowledgeBase repository singleton. Wraps the
    KnowledgeBase the same way TeamGraphRepository wraps the KG
    provider; consumers (chat service, KB route, GraphQL KB resolvers)
    depend on this repository instead of the concrete KBase class.
    No Depends args so the GraphQL resolver can call it directly too.
    """
    global _kb_repo
    if _kb_repo is None:
        from api.repositories.kb_repo import KnowledgeBaseRepository
        _kb_repo = KnowledgeBaseRepository()
    return _kb_repo

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that yields an async database session for PostgreSQL.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except HTTPException:
            await session.rollback()
            raise
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error, rolled back: {e}")
            raise
        finally:
            await session.close()

# ── Repository Dependency Injectors (Deferred Imports) ────────────────────────

def get_user_repo(db: AsyncSession = Depends(get_db)):
    """
    Inject UserRepository. Deferred import avoids early dependency cycle errors.
    """
    from api.repositories.user_repo import UserRepository
    return UserRepository(db)

def get_chat_repo(db: AsyncSession = Depends(get_db)):
    """
    Inject ChatRepository. Deferred import avoids early dependency cycle errors.
    """
    from api.repositories.chat_repo import ChatRepository
    return ChatRepository(db)

def get_feedback_repo(db: AsyncSession = Depends(get_db)):
    """
    Inject FeedbackRepository. Deferred import avoids early dependency cycle errors.
    """
    from api.repositories.feedback_repo import FeedbackRepository
    return FeedbackRepository(db)

def get_best11_repo() -> "Best11Repository":
    """
    Inject the Best-11 data repository singleton (collects squads, team
    totals and per-match player form from their sources).
    """
    global _best11_repo
    if _best11_repo is None:
        from api.repositories.best11_repo import Best11Repository
        _best11_repo = Best11Repository()
    return _best11_repo

# ── Service Dependency Injectors (Deferred Imports) ───────────────────────────

def get_prediction_service(
    feedback_repo = Depends(get_feedback_repo),
    rag_wrapper: AsyncRAGWrapper = Depends(get_async_rag)
):
    from api.services.prediction_service import PredictionService
    return PredictionService(feedback_repo, rag_wrapper)

def get_live_prediction_service(
    rag_wrapper: AsyncRAGWrapper = Depends(get_async_rag)
):
    from api.services.live_prediction_service import LivePredictionService
    return LivePredictionService(rag_wrapper)

def get_chat_service(
    chat_repo = Depends(get_chat_repo),
    kb_repo: "KnowledgeBaseRepository" = Depends(get_kb_repo)
):
    from api.services.chat_service import ChatService
    return ChatService(chat_repo, kb_repo)

def get_supervisor_service(
    db: AsyncSession = Depends(get_db),
    user_repo = Depends(get_user_repo),
    feedback_repo = Depends(get_feedback_repo)
):
    from api.services.supervisor_service import SupervisorService
    return SupervisorService(db, user_repo, feedback_repo)

def get_best11_api_service() -> "Best11ApiService":
    """
    Inject the Best-11 application service singleton. The service owns
    the domain Best11Service and wires it to the Best-11 repository.
    No Depends args so the GraphQL resolver can call it directly too.
    """
    global _best11_api_service
    if _best11_api_service is None:
        from api.services.best11_service import Best11ApiService
        _best11_api_service = Best11ApiService(get_best11_repo())
    return _best11_api_service

def get_scout_repo() -> "ScoutRepository":
    """
    Inject the Scout data repository singleton (collects identity pools,
    per-player stats and transfer info from their sources).
    """
    global _scout_repo
    if _scout_repo is None:
        from api.repositories.scout_repo import ScoutRepository
        _scout_repo = ScoutRepository()
    return _scout_repo

def get_scout_api_service() -> "ScoutApiService":
    """
    Inject the Scout application service singleton. The service owns
    the domain ScoutService and wires it to the Scout repository.
    No Depends args so the GraphQL resolver can call it directly too.
    """
    global _scout_api_service
    if _scout_api_service is None:
        from api.services.scout_service import ScoutApiService
        _scout_api_service = ScoutApiService(get_scout_repo())
    return _scout_api_service
