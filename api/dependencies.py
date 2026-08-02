import logging
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends, HTTPException

from api.database import AsyncSessionLocal
from rag.rag_orchestrator import FootballRAGSystem

from api.async_rag import AsyncRAGWrapper

logger = logging.getLogger(__name__)

# Global singleton RAG System instance
_rag_system: Optional[FootballRAGSystem] = None
_async_rag: Optional[AsyncRAGWrapper] = None

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

def get_rag_system() -> FootballRAGSystem:
    """
    Dependency that returns the active FootballRAGSystem instance.
    """
    global _rag_system
    if _rag_system is None:
        raise RuntimeError("RAG System has not been initialized. Call init_rag_system() on startup.")
    return _rag_system

def get_async_rag(rag_sys: FootballRAGSystem = Depends(get_rag_system)) -> AsyncRAGWrapper:
    """
    Dependency that returns the ThreadPoolExecutor wrapped AsyncRAGWrapper.
    """
    global _async_rag
    if _async_rag is None:
        _async_rag = AsyncRAGWrapper(rag_sys)
    return _async_rag

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

# ── Service Dependency Injectors (Deferred Imports) ───────────────────────────

def get_prediction_service(
    feedback_repo = Depends(get_feedback_repo),
    rag_wrapper: AsyncRAGWrapper = Depends(get_async_rag)
):
    from api.services.prediction_service import PredictionService
    return PredictionService(feedback_repo, rag_wrapper)

def get_chat_service(
    chat_repo = Depends(get_chat_repo),
    rag_wrapper: AsyncRAGWrapper = Depends(get_async_rag)
):
    from api.services.chat_service import ChatService
    return ChatService(chat_repo, rag_wrapper)

def get_supervisor_service(
    db: AsyncSession = Depends(get_db),
    user_repo = Depends(get_user_repo),
    feedback_repo = Depends(get_feedback_repo)
):
    from api.services.supervisor_service import SupervisorService
    return SupervisorService(db, user_repo, feedback_repo)
