from api.repositories import KnowledgeBaseRepository
from api.repositories import Best11Repository
from api.services.best11_service import Best11ApiService
from api.services.scout_service import ScoutApiService
from api.repositories.scout_repo import ScoutRepository
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


def init_rag_system(force_reload: bool = False) -> FootballRAGSystem:
    """
    Initialize the global singleton RAG System instance on app startup (or hot-reload).
    Tolerates LLM-config failure by retrying with llm="none" so the API
    still starts (returns structured predictions without narrative).
    """
    global _rag_system, _async_rag
    if force_reload and _rag_system is not None:
        logger.info("Closing previous FootballRAGSystem instance for hot-reload...")
        try:
            if hasattr(_rag_system, "close"):
                _rag_system.close()
        except Exception as e:
            logger.warning("Non-fatal error closing previous RAG system: %s", e)
        _rag_system = None
        _async_rag = None

    if _rag_system is None:
        logger.info("Initializing global FootballRAGSystem (force_reload=%s)...", force_reload)
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

        # ── Eagerly warm up models so the startup log shows the real backend ─
        import torch
        gpu_ok = torch.cuda.is_available()
        logger.info("torch.cuda.is_available()=%s", gpu_ok)
        if gpu_ok:
            dev = torch.cuda.get_device_name(0)
            logger.info("GPU detected: %s", dev)
        else:
            logger.warning("CUDA NOT visible to torch — running on CPU. (For GPU, ensure the image has nvidia wheels and the container was started with 'gpus: all')")

        # Force-load both experts so "[LLM] READY" / ONNX lines appear NOW, not on first request.
        try:
            llm = getattr(_rag_system, "llm", None)
            if llm is not None and hasattr(llm, "_load"):
                logger.info("Warming up LLM (Expert 2)...")
                llm._load()
                logger.info("LLM warm; backend=%s", getattr(llm, "_backend", "unknown"))
        except Exception as e:
            logger.warning("LLM eager load failed (%s); it will retry lazily on first request.", type(e).__name__)
        try:
            predictor = getattr(_rag_system, "predictor", None)
            if predictor is not None and hasattr(predictor, "load"):
                logger.info("Warming up GNN (Expert 1)...")
                predictor.load()
                logger.info("GNN warm; backend=%s", type(predictor).__name__)
        except Exception as e:
            logger.warning("GNN eager load failed (%s); it will retry lazily on first request.", type(e).__name__)
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

def init_knowledge_base(force_reload: bool = False) -> "KnowledgeBase":
    """
    Initialize the global KnowledgeBase singleton on app startup or hot-reload.
    Cheap: construction is lazy — CSV/Postgres/FAISS/GNN load only on first use.
    """
    global _kb, _kb_repo, _best11_repo, _best11_api_service, _scout_repo, _scout_api_service
    if force_reload:
        _kb = None
        _kb_repo = None
        _best11_repo = None
        _best11_api_service = None
        _scout_repo = None
        _scout_api_service = None

    if _kb is None:
        from rag.knowledge_base.kb import KnowledgeBase
        _kb = KnowledgeBase()
        logger.info("KnowledgeBase singleton initialized (force_reload=%s).", force_reload)
    return _kb


def reload_all_singletons(app=None) -> dict:
    """
    Hot-reload FootballRAGSystem, AsyncRAGWrapper, KnowledgeBase, Best11, and Scout
    singletons in-memory without restarting the FastAPI process.
    """
    rag_sys = init_rag_system(force_reload=True)
    async_rag = AsyncRAGWrapper(rag_sys)
    kb = init_knowledge_base(force_reload=True)

    global _async_rag
    _async_rag = async_rag

    if app is not None and hasattr(app, "state"):
        app.state.rag_system = rag_sys
        app.state.async_rag = async_rag
        app.state.knowledge_base = kb

    llm_backend = getattr(getattr(rag_sys, "llm", None), "_backend", None) or (
        "onnx" if getattr(rag_sys, "llm", None) is not None else "disabled"
    )
    gnn_backend = (
        type(getattr(rag_sys, "predictor", None)).__name__
        if getattr(rag_sys, "predictor", None)
        else "disabled"
    )
    return {
        "status": "reloaded",
        "gnn_backend": gnn_backend,
        "llm_backend": llm_backend,
        "knowledge_base": "reinitialized",
    }


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
