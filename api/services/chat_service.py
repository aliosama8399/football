from fastapi import HTTPException, status

from api.utils import strip_markdown
from api.repositories.chat_repo import ChatRepository
from api.async_rag import AsyncRAGWrapper


class ChatService:
    """
    Business flow for conversations: create/list/get messages, plus the
    prediction-mode team detection and general-RAG query routing with
    conversational memory. Model replies are stripped of markdown.
    """

    def __init__(self, chat_repo: ChatRepository, rag_wrapper: AsyncRAGWrapper):
        self.chat_repo = chat_repo
        self.rag_wrapper = rag_wrapper

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

        run_prediction = False
        detected_teams = []
        if conversation.mode == "prediction":
            known_teams = self.rag_wrapper.get_available_teams()
            q_lower = content.lower()
            for team in known_teams:
                if team.lower() in q_lower:
                    detected_teams.append(team)
                if len(detected_teams) >= 2:
                    break
            if len(detected_teams) == 2:
                run_prediction = True

        if run_prediction:
            reply_content = await self.rag_wrapper.predict_match(
                home_team=detected_teams[0],
                away_team=detected_teams[1]
            )
        else:
            memory_lines = []
            for msg in past_messages[:-1][-6:]:
                memory_lines.append(f"{msg.sender.upper()}: {msg.content}")
            memory_context = "\n".join(memory_lines)

            if memory_context:
                full_prompt = (
                    f"You are a football chatbot. Ground your response in the provided database. "
                    f"Here is the recent conversation memory:\n{memory_context}\n\n"
                    f"User's current question: {content}"
                )
            else:
                full_prompt = content

            reply_content = await self.rag_wrapper.query(question=full_prompt)

        reply_content = strip_markdown(reply_content)

        return await self.chat_repo.save_message(
            conversation_id=conversation_id,
            sender="assistant",
            content=reply_content
        )
