from fastapi import APIRouter, Depends
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
    current_user: User = Depends(get_current_user),
    chat_service = Depends(get_chat_service)
):
    """
    Post a new user query to a conversation.
    Processes the request in either prediction/tactics mode (Model GNN+LLM prediction)
    or general conversational RAG mode, persisting both queries and replies.
    """
    return await chat_service.post_message(conversation_id, user_id=current_user.id, content=payload.content)
