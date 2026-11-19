import asyncio
import json

from fastapi import HTTPException, status

from api.utils import strip_markdown
from api.repositories.chat_repo import ChatRepository
from api.repositories.kb_repo import KnowledgeBaseRepository
from api.config import rag_cfg
from models.llm_providers import resolve_default_provider


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
        raw_llm = str(rag_cfg.get("llm_provider") or "").strip()
        self._llm_name = None if raw_llm.lower() == "none" else (raw_llm or resolve_default_provider())

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

        # Conversational memory: fetch up to 6 recent messages before saving new turn
        past_messages = await self.chat_repo.get_recent_messages(conversation_id=conversation_id, limit=6)
        memory_lines = [f"{msg.sender.upper()}: {msg.content}" for msg in past_messages]
        memory_context = "\n".join(memory_lines) or None

        await self.chat_repo.save_message(conversation_id=conversation_id, sender="user", content=content)

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

    @staticmethod
    def _format_sse(event: str, payload: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    async def stream_post_message(self, conversation_id: int, user_id: int, content: str):
        """Stream pipeline stages, generated tactical tokens, citations, and completion as SSE."""
        conversation = await self.chat_repo.get_conversation(conversation_id, user_id=user_id)
        if not conversation:
            yield self._format_sse("error", {"detail": "Conversation not found", "status": 404})
            return

        past_messages = await self.chat_repo.get_recent_messages(conversation_id=conversation_id, limit=6)
        memory_lines = [f"{msg.sender.upper()}: {msg.content}" for msg in past_messages]
        memory_context = "\n".join(memory_lines) or None

        await self.chat_repo.save_message(conversation_id=conversation_id, sender="user", content=content)
        yield self._format_sse(
            "stage",
            {
                "stage": "context_loaded",
                "memory_turns": len(past_messages),
                "mode": conversation.mode,
            },
        )
        yield self._format_sse(
            "stage",
            {"stage": "hybrid_rag_and_gnn", "status": "executing"},
        )

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

        words = reply_content.split(" ")
        chunk_size = 4
        for idx in range(0, len(words), chunk_size):
            piece = " ".join(words[idx : idx + chunk_size])
            if idx + chunk_size < len(words):
                piece += " "
            yield self._format_sse("token", {"delta": piece})

        sources_list = []
        if answer.bundle and answer.bundle.sources:
            sources_list = [s.to_json() for s in answer.bundle.sources]
        sources_json = json.dumps(sources_list, ensure_ascii=False) if sources_list else None

        saved_msg = await self.chat_repo.save_message(
            conversation_id=conversation_id,
            sender="assistant",
            content=reply_content,
            sources=sources_json,
        )
        yield self._format_sse("citations", {"sources": sources_list})
        yield self._format_sse(
            "done",
            {
                "message_id": getattr(saved_msg, "id", None),
                "conversation_id": conversation_id,
                "provider": answer.provider,
            },
        )
