import asyncio
import json

from fastapi import HTTPException, status

from api.utils import strip_markdown
from api.repositories.chat_repo import ChatRepository
from api.repositories.kb_repo import KnowledgeBaseRepository
from api.config import rag_cfg


class ChatService:
    """
    Business flow for conversations: create/list/get messages, plus KB-routed
    Q&A. Every message goes through the KnowledgeBaseRepository →
    KnowledgeBase.ask() — the intent classifier routes to the right resolver
    (prediction mode forces prediction intent). Conversational memory is
    injected into the LLM narration prompt; assistant replies persist their
    source citations (messages.sources JSON).

    The KBase is injected as a repository (KnowledgeBaseRepository) so the
    service depends on a data-access interface, not the concrete class.
    """

    def __init__(self, chat_repo: ChatRepository,
                 knowledge_base_repo: KnowledgeBaseRepository):
        self.chat_repo = chat_repo
        self.kb_repo = knowledge_base_repo
        # Same key the RAG orchestrator uses; missing/empty → none-safe
        # structured answers (no LLM).
        self._llm_name = (rag_cfg.get("llm_provider") or "").strip() or None

    async def create_conversation(self, user_id: int, title: str, mode: str):
        if mode not in ("prediction", "general"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid mode. Must be 'prediction' or 'general'."
            )
        return await self.chat_repo.create_conversation(user_id=user_id, title=title, mode=mode)

    async def list_conversations(self, user_id: int):
        return await self.chat_repo.list_conversations(user_id=user_id)

    async def get_messages(self, conversation_id: int, user_id: int):
        conversation = await self.chat_repo.get_conversation(conversation_id, user_id=user_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        return await self.chat_repo.get_messages(conversation_id=conversation_id)

    async def post_message(self, conversation_id: int, user_id: int, content: str):
        conversation = await self.chat_repo.get_conversation(conversation_id, user_id=user_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )

        await self.chat_repo.save_message(conversation_id=conversation_id, sender="user", content=content)

        past_messages = await self.chat_repo.get_messages(conversation_id=conversation_id)

        # Conversational memory: the last few turns before this question.
        memory_lines = []
        for msg in past_messages[:-1][-6:]:
            memory_lines.append(f"{msg.sender.upper()}: {msg.content}")
        memory_context = "\n".join(memory_lines) or None

        # KB ask runs blocking internals (CSV/Postgres/FAISS/GNN); offload it.
        answer = await asyncio.to_thread(
            self.kb_repo.ask,
            content,
            llm_name=self._llm_name,
            prefer_prediction=(conversation.mode == "prediction"),
            memory=memory_context,
        )

        reply_content = answer.content
        if answer.provider != "none" or not reply_content.lstrip().startswith("{"):
            reply_content = strip_markdown(reply_content)

        sources_json = None
        if answer.bundle.sources:
            sources_json = json.dumps(
                [s.to_json() for s in answer.bundle.sources],
                ensure_ascii=False,
            )

        return await self.chat_repo.save_message(
            conversation_id=conversation_id,
            sender="assistant",
            content=reply_content,
            sources=sources_json,
        )
