import json
from fastapi import APIRouter, Depends, BackgroundTasks
from typing import List

from api.schemas import ConversationCreate, ConversationResponse, MessageCreate, MessageResponse
from api.dependencies import get_chat_service
from api.auth import get_current_user
from api.database import User

router = APIRouter(prefix="/chat", tags=["Conversations & Chatbot"])

@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
    payload: ConversationCreate,
    current_user: User = Depends(get_current_user),
    chat_service = Depends(get_chat_service)
):
    """Create a new general chat room or prediction analysis conversation."""
    return await chat_service.create_conversation(
        user_id=current_user.id,
        title=payload.title,
        mode=payload.mode
    )

@router.get("/conversations", response_model=List[ConversationResponse])
async def list_conversations(
    current_user: User = Depends(get_current_user),
    chat_service = Depends(get_chat_service)
):
    """Retrieve all conversations for the authenticated user."""
    return await chat_service.list_conversations(user_id=current_user.id)

@router.get("/conversations/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_messages(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    chat_service = Depends(get_chat_service)
):
    """Fetch chronological message history for a conversation."""
    return await chat_service.get_messages(conversation_id, user_id=current_user.id)

@router.post("/conversations/{conversation_id}/messages", response_model=MessageResponse)
async def post_message(
    conversation_id: int,
    payload: MessageCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    chat_service = Depends(get_chat_service)
):
    """
    Post a new user query to a conversation.
    Processes the request in either prediction/tactics mode (Model GNN+LLM prediction)
    or general conversational RAG mode, persisting both queries and replies.
    """
    resp = await chat_service.post_message(conversation_id, user_id=current_user.id, content=payload.content)
    try:
        from monitoring.metrics_registry import api_rag_queries_total
        api_rag_queries_total.labels(status="success").inc()
    except Exception:
        pass

    # Online 5-dimensional drift and evidence evaluation in background task (non-blocking)
    try:
        from api.services.drift_service import drift_evaluator
        sources_val = getattr(resp, "sources", None)
        if isinstance(sources_val, str):
            try:
                sources_val = json.loads(sources_val)
            except Exception:
                sources_val = []
        elif not isinstance(sources_val, list):
            sources_val = []

        background_tasks.add_task(
            drift_evaluator.evaluate_rag_turn,
            query=payload.content,
            answer=resp.content,
            sources=sources_val,
        )
    except Exception:
        pass

    return resp
