from typing import Optional, List
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from api.database import User

class UserRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_id(self, user_id: int) -> Optional[User]:
        stmt = select(User).where(User.id == user_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_username(self, username: str) -> Optional[User]:
        stmt = select(User).where(User.username == username)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> Optional[User]:
        stmt = select(User).where(User.email == email)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_username_or_email(self, identifier: str) -> Optional[User]:
        stmt = select(User).where((User.username == identifier) | (User.email == identifier))
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_activation_token(self, token: str) -> Optional[User]:
        stmt = select(User).where(User.activation_token == token)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def create_pending_user(self, username: str, email: str, role: str, activation_token: str) -> User:
        user = User(
            username=username,
            email=email,
            role=role,
            activation_token=activation_token,
            is_active=False
        )
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        return user

    async def activate_user(self, user: User, hashed_password: str) -> User:
        user.hashed_password = hashed_password
        user.activation_token = None
        user.is_active = True
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        return user

    async def update_password(self, user: User, hashed_password: str) -> User:
        user.hashed_password = hashed_password
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        return user

    async def list_users(self) -> List[User]:
        stmt = select(User).order_by(User.id)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
