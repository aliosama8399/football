import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from strawberry.fastapi import GraphQLRouter


from api.config import settings
from api.database import init_db, AsyncSessionLocal
from api.graph_db import init_graph_db, close_graph_db
from api.dependencies import (
    init_rag_system,
    get_rag_system,
    init_knowledge_base,
    reload_all_singletons,
)
from api.auth import seed_supervisor_user
from api.routes import (
    auth_router,
    chat_router,
    predictions_router,
    live_predictions_router,
    feedback_router,
    supervisor_router,
    submissions_router,
    kb_router,
    best11_router,
    scout_router,
    jobs_router
)
from api.services.job_runner import job_manager
from api.graphql import schema

import os
from logging.handlers import RotatingFileHandler

# Configure logging — stdout + rotating file
_log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(_log_dir, exist_ok=True)
_log_file = os.path.join(_log_dir, "api.log")

_fmt = logging.Formatter("%(asctime)s %(levelname)-8s [%(name)s] %(message)s")

_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_fmt)

_file_handler = RotatingFileHandler(
    _log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
)
_file_handler.setFormatter(_fmt)

logging.basicConfig(level=logging.INFO, handlers=[_stream_handler, _file_handler])
logger = logging.getLogger(__name__)
logger.info("Logging to file: %s", os.path.abspath(_log_file))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles FastAPI application lifespan startup/shutdown hooks.
    Bootstraps resources (Postgres ORM schemas, KG connection pools, singleton RAG structures,
    and the autonomous MLOps background scheduler) and ensures clean teardowns on shutdown.
    """
    logger.info("Initializing Football RAG SaaS Service startup...")
    
    # 1. Bring Postgres schema to Alembic head (ORM tables only; matches/teams untouched)
    try:
        await init_db()
    except Exception as e:
        logger.critical(f"Database schema initialization failed: {e}")
        raise e
        
    # 2. Automatically seed default supervisor if DB is empty
    async with AsyncSessionLocal() as session:
        try:
            await seed_supervisor_user(session)
        except Exception as e:
            logger.error(f"Error seeding supervisor user: {e}")
            
    # 3. Connect the dynamic KG Provider pool (Neo4j or Postgres)
    try:
        init_graph_db()
    except Exception as e:
        logger.error(f"Error initializing KG database connection pool: {e}")
        
    # 4. Load Football RAG System singleton and stash on app.state
    #    Falls back to llm='none' if the configured LLM cannot be constructed (see dependencies.py).
    rag_status = "DOWN"
    llm_status = "-"
    gnn_status = "-"
    try:
        rag_sys = init_rag_system()
        app.state.rag_system = rag_sys
        from api.async_rag import AsyncRAGWrapper
        app.state.async_rag = AsyncRAGWrapper(rag_sys)
        rag_status = "OK"
        llm_status = getattr(getattr(rag_sys, "llm", None), "_backend", None) or \
            ("onnx" if getattr(rag_sys, "llm", None) is not None else "disabled")
        gnn_status = type(getattr(rag_sys, "predictor", None)).__name__ if \
            getattr(rag_sys, "predictor", None) else "disabled"
    except Exception as e:
        logger.error(f"Error loading RAG system structures: {e}")

    # 4.5. KnowledgeBase facade (chat-KB). Construction and pre-warm on app.state
    kb_status = "OK"
    try:
        kb = init_knowledge_base()
        app.state.knowledge_base = kb
    except Exception as e:
        logger.error(f"Error initializing KnowledgeBase: {e}")
        kb_status = "DOWN"

    # 4.8. Start Autonomous Background Scheduler & Self-Healing Bootstrap
    try:
        job_manager.start_auto_scheduler(app)
        sched_info = job_manager.get_scheduler_status()
        sched_status = f"ENABLED (every {sched_info['interval_hours']}h)" if sched_info["enabled"] else "PAUSED"
    except Exception as e:
        logger.error(f"Error starting autonomous scheduler: {e}")
        sched_status = "ERROR"

    logger.info("=" * 62)
    logger.info("STARTUP SUMMARY")
    logger.info(f"  DB (Postgres/Alembic) : OK")
    logger.info(f"  KG provider           : {settings.kg_provider}")
    logger.info(f"  Vector store          : {settings.vector_provider}")
    logger.info(f"  LLM (Expert 2)        : backend={llm_status}")
    logger.info(f"  GNN (Expert 1)        : {gnn_status}")
    logger.info(f"  RAG orchestrator      : {rag_status}")
    logger.info(f"  Knowledge base        : {kb_status}")
    logger.info(f"  Auto MLOps Scheduler  : {sched_status}")
    logger.info("=" * 62)
    logger.info("Application startup COMPLETE — serving on port 8001")

    yield
    
    logger.info("Executing teardown / shutdown cleanup...")
    try:
        job_manager.stop_auto_scheduler()
    except Exception:
        pass
    # 5. Clean up graph connections
    close_graph_db()
    # 6. Clean up RAG resources
    try:
        rag = getattr(app.state, "rag_system", None) or get_rag_system()
        if hasattr(rag, "close"):
            rag.close()
    except Exception as e:
        logger.error(f"Error closing RAG orchestrator resources: {e}")

app = FastAPI(
    title=settings.app_name,
    description="REST & GraphQL FastAPI interface for Football RAG GNN+LLM predictions & tactical chatbot.",
    version="1.0.0",
    lifespan=lifespan
)

# Setup CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register REST Routers
app.include_router(auth_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(predictions_router, prefix="/api/v1")
app.include_router(live_predictions_router, prefix="/api/v1")
app.include_router(feedback_router, prefix="/api/v1")
app.include_router(supervisor_router, prefix="/api/v1")
app.include_router(submissions_router, prefix="/api/v1")
app.include_router(kb_router, prefix="/api/v1")
app.include_router(best11_router, prefix="/api/v1")
app.include_router(scout_router, prefix="/api/v1")
app.include_router(jobs_router, prefix="/api/v1")

# Mount Strawberry GraphQL endpoint
graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")

@app.get("/", tags=["Health Check"])
@app.get("/health", tags=["Health Check"])
@app.get("/api/v1/health", tags=["Health Check"])
async def root():
    """Simple API health probe."""
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "kg_provider": settings.kg_provider,
        "version": "1.0.0"
    }


@app.post("/api/v1/admin/reload", tags=["Administration"])
async def reload_api_state(request: Request):
    """
    Hot-reload FootballRAGSystem, ONNX/PyTorch GNN graph snapshots, FAISS/pgvector
    handles, and KnowledgeBase caches after an automated scrape/retrain cycle.
    """
    try:
        info = reload_all_singletons(request.app)
        logger.info("Hot-reloaded API singletons: %s", info)
        return info
    except Exception as e:
        logger.error("Failed hot-reloading API singletons: %s", e, exc_info=True)
        return {"status": "error", "message": str(e)}


# ── Serve UI Assets ──────────────────────────────────────────────────────────

@app.get("/ui", response_class=HTMLResponse, tags=["User Interface"])
async def get_ui():
    """Serve the main Tactical Dashboard HTML interface."""
    import os
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(content=html)


@app.get("/ui/style.css", tags=["User Interface"])
async def get_ui_css():
    """Serve the Tactical Dashboard stylesheet."""
    import os
    css_path = os.path.join(os.path.dirname(__file__), "static", "style.css")
    with open(css_path, "r", encoding="utf-8") as f:
        return Response(content=f.read(), media_type="text/css")


@app.get("/ui/app.js", tags=["User Interface"])
async def get_ui_js(nocache: str = ""):
    """Serve the Tactical Dashboard client-side application logic."""
    import os
    js_path = os.path.join(os.path.dirname(__file__), "static", "app.js")
    with open(js_path, "r", encoding="utf-8") as f:
        return Response(content=f.read(), media_type="application/javascript")


@app.get("/metrics", tags=["Monitoring"])
async def metrics_endpoint():
    """Prometheus metrics scrape endpoint for Grafana and monitoring."""
    try:
        from monitoring.metrics_registry import (
            get_prometheus_metrics_bytes,
            load_cached_metrics,
            CONTENT_TYPE_LATEST,
        )
        load_cached_metrics()
        return Response(content=get_prometheus_metrics_bytes(), media_type=CONTENT_TYPE_LATEST)
    except Exception as e:
        logger.error("Prometheus metrics export error: %s", e)
        return Response(content=f"# Error exporting metrics: {e}\n", media_type="text/plain", status_code=500)


@app.post("/api/v1/monitoring/live-metrics", tags=["Monitoring"])
async def receive_live_metrics(payload: dict):
    """
    Ingest real-time evaluation metrics over HTTP from live test suites
    (run_rag_eval.py, run_prediction_eval.py) and update Prometheus Gauges directly in memory.
    """
    try:
        from monitoring.metrics_registry import (
            record_rag_evaluation_metrics,
            record_prediction_evaluation_metrics,
        )
        metric_type = payload.get("type")
        if metric_type == "rag":
            record_rag_evaluation_metrics(
                avg_latency=float(payload.get("avg_latency", 0.0)),
                avg_numeric=float(payload.get("avg_numeric", 0.0)),
                avg_grounding=float(payload.get("avg_grounding", 0.0)),
                ragas_scores=payload.get("ragas_scores"),
            )
        elif metric_type == "prediction":
            record_prediction_evaluation_metrics(
                gnn_acc=float(payload.get("gnn_accuracy", 0.0)),
                slm_acc=float(payload.get("slm_accuracy", 0.0)),
                ens_acc=float(payload.get("ensemble_accuracy", 0.0)),
                brier=float(payload.get("brier_score", 0.0)),
                log_loss=float(payload.get("log_loss", 0.0)),
                consensus=float(payload.get("consensus_rate", 0.0)),
                tactical=float(payload.get("tactical_grounding", 0.0)),
                latency=float(payload.get("latency", 0.0)),
            )
        elif metric_type in ("drift", "online"):
            from monitoring.metrics_registry import (
                record_online_rag_drift_metrics,
                record_online_prediction_drift_metrics,
            )
            if "evidence_score" in payload:
                record_online_rag_drift_metrics(
                    evidence_score=float(payload.get("evidence_score", 0.0)),
                    faithfulness_score=float(payload.get("faithfulness_score", 0.0)),
                    query_drift_score=float(payload.get("query_drift_score", 0.0)),
                    drift_type=payload.get("drift_type"),
                )
            if "entropy" in payload:
                record_online_prediction_drift_metrics(
                    entropy=float(payload.get("entropy", 0.0)),
                    drift_type=payload.get("drift_type"),
                )
        return {"status": "ok", "updated": metric_type}
    except Exception as e:
        logger.error("Failed processing live metric payload: %s", e)
        return {"status": "error", "message": str(e)}
