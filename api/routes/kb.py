import asyncio

from fastapi import APIRouter, Depends

from api.schemas import (
    KBRetrieveRequest,
    KBAskRequest,
    KBBundleResponse,
    KBAnswerResponse,
)
from api.dependencies import get_knowledge_base
from api.auth import get_current_user
from api.database import User

router = APIRouter(prefix="/kb", tags=["Knowledge Base"])


@router.post("/retrieve", response_model=KBBundleResponse)
async def kb_retrieve(
    payload: KBRetrieveRequest,
    current_user: User = Depends(get_current_user),
    knowledge_base = Depends(get_knowledge_base),
):
    """
    Retrieve the KB context bundle for a question — no LLM involved.
    Returns intent classification, facts, tables, semantic hits and sources.
    """
    bundle = await asyncio.to_thread(
        knowledge_base.retrieve,
        payload.question,
        prefer_prediction=payload.prefer_prediction,
    )
    return KBBundleResponse(**bundle.to_json())


@router.post("/ask", response_model=KBAnswerResponse)
async def kb_ask(
    payload: KBAskRequest,
    current_user: User = Depends(get_current_user),
    knowledge_base = Depends(get_knowledge_base),
):
    """
    Ask the KB. Pass llm_provider to get a narrated answer from that provider
    (falls back to a structured answer on provider failure). Omit it to get a
    structured answer without any LLM.
    """
    answer = await asyncio.to_thread(
        knowledge_base.ask,
        payload.question,
        llm_name=payload.llm_provider,
        prefer_prediction=payload.prefer_prediction,
    )
    return KBAnswerResponse(
        content=answer.content,
        provider=answer.provider,
        error=answer.error,
        sources=[s.to_json() for s in answer.bundle.sources],
    )
