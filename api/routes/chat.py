from fastapi import APIRouter, Depends, HTTPException, status
from typing import List

from api.schemas import ConversationCreate, ConversationResponse, MessageCreate, MessageResponse
from api.dependencies import get_chat_repo, get_async_rag
from api.auth import get_current_user
from api.database import User

router = APIRouter(prefix="/chat", tags=["Conversations & Chatbot"])

@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
    payload: ConversationCreate,
    current_user: User = Depends(get_current_user),
    chat_repo = Depends(get_chat_repo)
):
    """Create a new general chat room or prediction analysis conversation."""
    if payload.mode not in ("prediction", "general"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid mode. Must be 'prediction' or 'general'."
        )
    return await chat_repo.create_conversation(
        user_id=current_user.id,
        title=payload.title,
        mode=payload.mode
    )

@router.get("/conversations", response_model=List[ConversationResponse])
async def list_conversations(
    current_user: User = Depends(get_current_user),
    chat_repo = Depends(get_chat_repo)
):
    """Retrieve all conversations for the authenticated user."""
    return await chat_repo.list_conversations(user_id=current_user.id)

@router.get("/conversations/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_messages(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    chat_repo = Depends(get_chat_repo)
):
    """Fetch chronological message history for a conversation."""
    conversation = await chat_repo.get_conversation(conversation_id, user_id=current_user.id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    return await chat_repo.get_messages(conversation_id=conversation_id)

@router.post("/conversations/{conversation_id}/messages", response_model=MessageResponse)
async def post_message(
    conversation_id: int,
    payload: MessageCreate,
    current_user: User = Depends(get_current_user),
    chat_repo = Depends(get_chat_repo),
    rag_wrapper = Depends(get_async_rag)
):
    """
    Post a new user query to a conversation. 
    Processes the request in either prediction/tactics mode (Model GNN+LLM prediction)
    or general conversational RAG mode, persisting both queries and replies.
    """
    conversation = await chat_repo.get_conversation(conversation_id, user_id=current_user.id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    # 1. Save user message to database
    user_msg = await chat_repo.save_message(
        conversation_id=conversation_id,
        sender="user",
        content=payload.content
    )

    # 2. Retrieve recent message history for memory/conversational context
    past_messages = await chat_repo.get_messages(conversation_id=conversation_id)
    
    # 3. Detect if match prediction flow is requested
    # If mode is prediction and we extract exactly two known teams from the text, run predict_match
    run_prediction = False
    detected_teams = []
    if conversation.mode == "prediction":
        known_teams = rag_wrapper.get_available_teams()
        q_lower = payload.content.lower()
        # Find exact matches for known team names
        for team in known_teams:
            if team.lower() in q_lower:
                detected_teams.append(team)
            if len(detected_teams) >= 2:
                break
        
        if len(detected_teams) == 2:
            run_prediction = True

    # 4. Invoke RAG wrapper asynchronously (runs via thread pool executor)
    if run_prediction:
        # Structured match prediction runs GNN Expert 1 + LLM Expert 2
        reply_content = await rag_wrapper.predict_match(
            home_team=detected_teams[0],
            away_team=detected_teams[1]
        )
    else:
        # Build prompt with conversational memory
        memory_lines = []
        # Gather up to 6 last messages (excluding the one just posted)
        for msg in past_messages[:-1][-6:]:
            memory_lines.append(f"{msg.sender.upper()}: {msg.content}")
        
        memory_context = "\n".join(memory_lines)
        
        if memory_context:
            full_prompt = (
                f"You are a football chatbot. Ground your response in the provided database. "
                f"Here is the recent conversation memory:\n{memory_context}\n\n"
                f"User's current question: {payload.content}"
            )
        else:
            full_prompt = payload.content

        reply_content = await rag_wrapper.query(question=full_prompt)

    # 5. Save and return assistant message reply
    assistant_msg = await chat_repo.save_message(
        conversation_id=conversation_id,
        sender="assistant",
        content=reply_content
    )
    return assistant_msg
