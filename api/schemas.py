from pydantic import BaseModel, EmailStr, Field
from datetime import datetime, date
from typing import Optional, List

# ── Auth Schemas ─────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    role: str = Field(default="user", description="Must be 'user' or 'supervisor'")

class UserActivate(BaseModel):
    token: str
    password: str = Field(..., min_length=6, max_length=100)

class UserLogin(BaseModel):
    username_or_email: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str
    is_active: bool
    activation_token: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

# ── Chat Schemas ─────────────────────────────────────────────────────────────

class ConversationCreate(BaseModel):
    title: Optional[str] = "General Chat"
    mode: str = Field(default="general", description="Must be 'prediction' or 'general'")

class ConversationResponse(BaseModel):
    id: int
    user_id: int
    title: str
    mode: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class MessageCreate(BaseModel):
    content: str = Field(..., min_length=1)

class MessageResponse(BaseModel):
    id: int
    conversation_id: int
    sender: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True

# ── Prediction Schemas ────────────────────────────────────────────────────────

class MatchPredictionRequest(BaseModel):
    home_team: str
    away_team: str
    match_date: Optional[date] = None

class MatchPredictionResponse(BaseModel):
    home_team: str
    away_team: str
    match_date: Optional[date] = None
    predicted_result: str
    predicted_home_goals: Optional[int] = None
    predicted_away_goals: Optional[int] = None
    tactical_analysis: Optional[str] = None
    source: str = Field(..., description="Either 'override' or 'live_model'")

# ── Feedback Schemas ──────────────────────────────────────────────────────────

class FeedbackCreate(BaseModel):
    type: str = Field(..., description="Either 'prediction_override' or 'tactic_modification'")
    
    # Optional prediction override fields
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    match_date: Optional[date] = None
    suggested_result: Optional[str] = None  # "H", "A", "D"
    suggested_home_goals: Optional[int] = None
    suggested_away_goals: Optional[int] = None
    suggested_analysis: Optional[str] = None

    # Optional tactic modification fields
    team_name: Optional[str] = None
    suggested_attack_tactic: Optional[str] = None
    suggested_defense_tactic: Optional[str] = None

class FeedbackReviewRequest(BaseModel):
    status: str = Field(..., description="Must be 'approved' or 'rejected'")
    admin_notes: Optional[str] = None

class FeedbackResponse(BaseModel):
    id: int
    user_id: int
    type: str
    status: str
    
    # Prediction fields
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    match_date: Optional[date] = None
    suggested_result: Optional[str] = None
    suggested_home_goals: Optional[int] = None
    suggested_away_goals: Optional[int] = None
    suggested_analysis: Optional[str] = None

    # Tactics fields
    team_name: Optional[str] = None
    suggested_attack_tactic: Optional[str] = None
    suggested_defense_tactic: Optional[str] = None

    admin_notes: Optional[str] = None
    reviewed_by_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
