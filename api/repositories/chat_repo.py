from typing import Optional, List
from sqlalchemy.future import select
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from api.database import Conversation, Message

class ChatRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_conversation(self, user_id: int, title: str, mode: str) -> Conversation:
        """Create a new conversation room."""
        conversation = Conversation(
            user_id=user_id,
            title=title,
            mode=mode
        )
        self.db.add(conversation)
        await self.db.commit()
        await self.db.refresh(conversation)
        return conversation

    async def get_conversation(self, conversation_id: int, user_id: int) -> Optional[Conversation]:
        """Fetch a specific conversation owned by the user."""
        stmt = select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_conversations(self, user_id: int) -> List[Conversation]:
        """List all conversations owned by the user, sorted by recency."""
        stmt = select(Conversation).where(
            Conversation.user_id == user_id
        ).order_by(Conversation.updated_at.desc())
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_messages(self, conversation_id: int) -> List[Message]:
        """Fetch all messages for a specific conversation in chronological order."""
        stmt = select(Message).where(
            Message.conversation_id == conversation_id
        ).order_by(Message.created_at.asc())
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def save_message(self, conversation_id: int, sender: str, content: str) -> Message:
        """Save a new chat message and update the parent conversation's timestamp."""
        message = Message(
            conversation_id=conversation_id,
            sender=sender,
            content=content
        )
        self.db.add(message)
        
        # Touch conversation to update updated_at field
        stmt = select(Conversation).where(Conversation.id == conversation_id)
        res = await self.db.execute(stmt)
        conv = res.scalar_one_or_none()
        if conv:
            conv.updated_at = func.now()
            self.db.add(conv)
            
        await self.db.commit()
        await self.db.refresh(message)
        return message
