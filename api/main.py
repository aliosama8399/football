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
from api.dependencies import init_rag_system, get_rag_system, init_knowledge_base
from api.auth import seed_supervisor_user
from api.routes import (
    auth_router,
    chat_router,
    predictions_router,
    feedback_router,
    supervisor_router,
    submissions_router,
    kb_router,
    best11_router,
    scout_router
)
from api.graphql import schema

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles FastAPI application lifespan startup/shutdown hooks.
    Bootstraps resources (Postgres ORM schemas, KG connection pools, and singleton RAG structures)
    and ensures clean teardowns on shutdown.
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
        
    # 4. Load Football RAG System singleton (loads sentence-transformers and FAISS index).
    #    Falls back to llm='none' if the configured LLM cannot be constructed (see dependencies.py).
    try:
        init_rag_system()
    except Exception as e:
        logger.error(f"Error loading RAG system structures: {e}")
        raise

    # 4.5. KnowledgeBase facade (chat-KB). Construction is lazy — internals
    #      (CSV, Postgres, FAISS, GNN) load on first question; never fatal.
    try:
        init_knowledge_base()
    except Exception as e:
        logger.error(f"Error initializing KnowledgeBase: {e}")

    yield
    
    logger.info("Executing teardown / shutdown cleanup...")
    # 5. Clean up graph connections
    close_graph_db()
    # 6. Clean up RAG resources
    try:
        rag = get_rag_system()
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
app.include_router(feedback_router, prefix="/api/v1")
app.include_router(supervisor_router, prefix="/api/v1")
app.include_router(submissions_router, prefix="/api/v1")
app.include_router(kb_router, prefix="/api/v1")
app.include_router(best11_router, prefix="/api/v1")
app.include_router(scout_router, prefix="/api/v1")

# Mount Strawberry GraphQL endpoint
graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")

@app.get("/", tags=["Health Check"])
async def root():
    """Simple API health probe."""
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "kg_provider": settings.kg_provider,
        "version": "1.0.0"
    }

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


