import logging
from datetime import datetime, date
from typing import Optional, List
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Integer, Boolean, ForeignKey, Text, DateTime, Date, func

from api.config import settings

logger = logging.getLogger(__name__)

# Format DSN for asyncpg compatibility
db_url = settings.postgres_dsn
if db_url.startswith("postgresql://"):
    db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

# Create Async Engine
engine = create_async_engine(
    db_url,
    echo=settings.debug,
    future=True,
    pool_pre_ping=True
)

# Async Session Factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    hashed_password: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    role: Mapped[str] = mapped_column(String(20), default="user", nullable=False)  # "user", "supervisor"
    activation_token: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    conversations: Mapped[List["Conversation"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    feedbacks: Mapped[List["Feedback"]] = relationship(
        back_populates="user", 
        foreign_keys="[Feedback.user_id]",
        cascade="all, delete-orphan"
    )
    match_submissions: Mapped[List["MatchSubmission"]] = relationship(
        back_populates="user",
        foreign_keys="[MatchSubmission.user_id]",
        cascade="all, delete-orphan"
    )
    tactical_analyses: Mapped[List["TacticalAnalysis"]] = relationship(
        back_populates="user",
        foreign_keys="[TacticalAnalysis.user_id]",
        cascade="all, delete-orphan"
    )

class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title: Mapped[str] = mapped_column(String(255), default="General Chat", nullable=False)
    mode: Mapped[str] = mapped_column(String(50), default="general", nullable=False)  # "prediction", "general"
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="conversations")
    messages: Mapped[List["Message"]] = relationship(back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    sender: Mapped[str] = mapped_column(String(50), nullable=False)  # "user", "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)

    # Relationships
    conversation: Mapped["Conversation"] = relationship(back_populates="messages")

class PredictionOverride(Base):
    __tablename__ = "prediction_overrides"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    home_team: Mapped[str] = mapped_column(String(100), nullable=False)
    away_team: Mapped[str] = mapped_column(String(100), nullable=False)
    match_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    predicted_result: Mapped[str] = mapped_column(String(10), nullable=False)  # "H", "A", "D"
    predicted_home_goals: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    predicted_away_goals: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    tactical_analysis: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    created_by_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

class Feedback(Base):
    __tablename__ = "feedbacks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    type: Mapped[str] = mapped_column(String(50), nullable=False)  # "prediction_override", "tactic_modification", "team_profile_edit"
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)  # "pending", "approved", "rejected"

    # Prediction fields
    home_team: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    away_team: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    match_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    suggested_result: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    suggested_home_goals: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    suggested_away_goals: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    suggested_analysis: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Tactics fields
    team_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    suggested_attack_tactic: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    suggested_defense_tactic: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Extended team profile fields
    suggested_attack_headline: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    suggested_defense_headline: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    suggested_strengths: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    suggested_weaknesses: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    admin_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reviewed_by_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="feedbacks", foreign_keys=[user_id])
    reviewer: Mapped[Optional["User"]] = relationship(foreign_keys=[reviewed_by_id])


class MatchSubmission(Base):
    __tablename__ = "match_submissions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)  # "pending", "approved", "rejected"

    # Core identifiers
    home_team: Mapped[str] = mapped_column(String(100), nullable=False)
    away_team: Mapped[str] = mapped_column(String(100), nullable=False)
    match_date: Mapped[date] = mapped_column(Date, nullable=False)
    league: Mapped[str] = mapped_column(String(50), nullable=False)
    season: Mapped[str] = mapped_column(String(10), nullable=False)

    # Full-time result (raw — system computes H/A/D)
    home_goals: Mapped[int] = mapped_column(Integer, nullable=False)
    away_goals: Mapped[int] = mapped_column(Integer, nullable=False)

    # Half-time
    home_ht_goals: Mapped[int] = mapped_column(Integer, nullable=False)
    away_ht_goals: Mapped[int] = mapped_column(Integer, nullable=False)

    # Expected goals (optional)
    home_xg: Mapped[Optional[float]] = mapped_column(nullable=True)
    away_xg: Mapped[Optional[float]] = mapped_column(nullable=True)

    # Match statistics
    home_shots: Mapped[int] = mapped_column(Integer, nullable=False)
    away_shots: Mapped[int] = mapped_column(Integer, nullable=False)
    home_sot: Mapped[int] = mapped_column(Integer, nullable=False)
    away_sot: Mapped[int] = mapped_column(Integer, nullable=False)
    home_corners: Mapped[int] = mapped_column(Integer, nullable=False)
    away_corners: Mapped[int] = mapped_column(Integer, nullable=False)
    home_fouls: Mapped[int] = mapped_column(Integer, nullable=False)
    away_fouls: Mapped[int] = mapped_column(Integer, nullable=False)
    home_yellows: Mapped[int] = mapped_column(Integer, nullable=False)
    away_yellows: Mapped[int] = mapped_column(Integer, nullable=False)
    home_reds: Mapped[int] = mapped_column(Integer, nullable=False)
    away_reds: Mapped[int] = mapped_column(Integer, nullable=False)

    # Admin review
    admin_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reviewed_by_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="match_submissions", foreign_keys=[user_id])
    reviewer: Mapped[Optional["User"]] = relationship(foreign_keys=[reviewed_by_id])


class TacticalAnalysis(Base):
    __tablename__ = "tactical_analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)  # "pending", "approved", "rejected"

    home_team: Mapped[str] = mapped_column(String(100), nullable=False)
    away_team: Mapped[str] = mapped_column(String(100), nullable=False)
    match_date: Mapped[date] = mapped_column(Date, nullable=False)
    analysis_text: Mapped[str] = mapped_column(Text, nullable=False)

    admin_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reviewed_by_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="tactical_analyses", foreign_keys=[user_id])
    reviewer: Mapped[Optional["User"]] = relationship(foreign_keys=[reviewed_by_id])

async def init_db():
    """
    Bring the database schema to the Alembic head.
    Alembic is the single source of truth for the ORM tables
    (users, conversations, messages, feedbacks, match_submissions,
    tactical_analyses, prediction_overrides). Non-ORM tables
    (matches, teams) are excluded via include_object in alembic/env.py.
    """
    logger.info("Running Alembic migrations (upgrade head)...")
    from alembic import command
    from alembic.config import Config

    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", settings.postgres_dsn)
    command.upgrade(cfg, "head")
    logger.info("Database schema is up to date (Alembic head).")
