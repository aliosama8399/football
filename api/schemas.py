import json

from pydantic import BaseModel, EmailStr, Field, field_validator
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

class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=100)
    role: str = Field(default="user", description="Must be 'user' or 'supervisor'")

class UserChangePassword(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=6, max_length=100)

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
    sources: Optional[List[dict]] = Field(
        default=None,
        description="KB source citations for assistant messages (JSON array of SourceRef dicts)."
    )

    class Config:
        from_attributes = True

    @field_validator("sources", mode="before")
    @classmethod
    def parse_sources(cls, v):
        """DB stores sources as JSON text; parse into a list of dicts."""
        if v is None or isinstance(v, list):
            return v
        if isinstance(v, str) and v.strip():
            try:
                return json.loads(v)
            except (ValueError, TypeError):
                return None
        return None

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
    probabilities: Optional[dict] = None
    analysis_breakdown: Optional[dict] = Field(
        default=None,
        description="Structured LLM analysis when the narrative is JSON "
                    "{prediction_verdict, confidence_rating, home_team_analysis, away_team_analysis, tactical_matchup_summary}",
    )

# ── Live Prediction Schemas ──────────────────────────────────────────────────

class LivePredictionRequest(BaseModel):
    home_team: str
    away_team: str
    minute: int = Field(..., ge=0, le=90, description="Current match minute (0-90)")
    home_goals: int = Field(default=0, ge=0)
    away_goals: int = Field(default=0, ge=0)

    # Cumulative live stats so far — None means "not provided" (skipped in pace math).
    home_xg: Optional[float] = None
    away_xg: Optional[float] = None
    home_shots: Optional[int] = None
    away_shots: Optional[int] = None
    home_sot: Optional[int] = None
    away_sot: Optional[int] = None
    home_corners: Optional[int] = None
    away_corners: Optional[int] = None
    home_fouls: Optional[int] = None
    away_fouls: Optional[int] = None
    home_yellows: Optional[int] = None
    away_yellows: Optional[int] = None
    home_reds: Optional[int] = Field(default=0, ge=0)
    away_reds: Optional[int] = Field(default=0, ge=0)

    explain: bool = Field(
        default=False,
        description="If true, call the LLM for a coach-actionable narrative (slower)."
    )

class LivePredictionResponse(BaseModel):
    home_team: str
    away_team: str
    minute: int
    home_goals: int
    away_goals: int
    predicted_result: str = Field(..., description="H | D | A")
    probabilities: dict = Field(..., description="Blended live H/D/A probabilities")
    pre_match_probabilities: dict = Field(..., description="TEA-GNN prior H/D/A probabilities")
    delta: dict = Field(..., description="live - pre probability deltas")
    expected_final_score: dict = Field(..., description="{'home': .., 'away': ..}")
    key_drivers: List[dict] = Field(default_factory=list, description="Ranked live-stat deviations")
    tactical_analysis: Optional[str] = None
    tactical_breakdown: Optional[dict] = Field(
        default=None,
        description="Structured coach-advisor JSON: {match_state, analysis: {who_controls_now, why, how_outlook_changed, coach_recommendations}}",
    )
    explain: bool = False
    source: str = Field(..., description="E.g. 'live_model' or 'live_model+llm'")

# ── Best-11 Schemas ───────────────────────────────────────────────────────────

class Best11Entry(BaseModel):
    slot: str = Field(..., description="Position bucket: GK | DF | MF | FW")
    name: str
    position: Optional[str] = None
    rating: float
    minutes: float
    flex: bool = Field(default=False, description="True when played out of position")
    top_shares: Optional[dict] = None
    season: Optional[dict] = Field(default=None, description="Season stat block: goals/assists/xg/xa/shots")
    h2h: Optional[dict] = Field(default=None, description="H2H stat block vs opponent (when requested)")

class Best11Sub(BaseModel):
    slot: str
    out: str
    in_: str = Field(alias="in", description="Player coming on")
    rating_delta: float
    reason: str

class Best11Bench(BaseModel):
    name: str
    position: Optional[str] = None
    rating: float
    minutes: float

class Best11Response(BaseModel):
    team: str
    league_code: str
    season: str
    formation: str
    captain: Optional[str] = None
    lineup: List[Best11Entry] = Field(default_factory=list)
    subs: List[Best11Sub] = Field(default_factory=list)
    bench: List[Best11Bench] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)

# ── Scouting Schemas ──────────────────────────────────────────────────────────

class ScoutStats(BaseModel):
    appearances: Optional[float] = None
    minutes: Optional[float] = None
    goals: Optional[float] = None
    assists: Optional[float] = None
    shots: Optional[float] = None
    shots_on_target: Optional[float] = None
    saves: Optional[float] = None
    goals_conceded: Optional[float] = None
    key_passes: Optional[float] = None
    tackles: Optional[float] = None
    interceptions: Optional[float] = None
    blocks: Optional[float] = None
    duels_won: Optional[float] = None
    dribbles_success: Optional[float] = None
    yellow_cards: Optional[float] = None
    red_cards: Optional[float] = None
    rating: Optional[float] = None

class ScoutTransfer(BaseModel):
    date: Optional[str] = None
    type: Optional[str] = None
    from_team: Optional[str] = None
    to_team: Optional[str] = None

class ScoutCandidate(BaseModel):
    rank: int
    name: str
    team: str
    league: Optional[str] = None
    position: Optional[str] = None
    age: Optional[int] = None
    nationality: Optional[str] = None
    shirt_number: Optional[float] = None
    stats: ScoutStats = Field(default_factory=ScoutStats)
    transfer: Optional[ScoutTransfer] = None
    score: float = Field(..., description="0..100 ranking score")
    score_breakdown: Optional[dict] = Field(default=None, description="Per-metric weighted contributions")

class ScoutResponse(BaseModel):
    season: str
    position: str
    youth: bool
    leagues: List[str] = Field(default_factory=list)
    pool_size: int = Field(0, description="Candidates that passed the position/age filter before scoring")
    top: int
    candidates: List[ScoutCandidate] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)

# ── Feedback Schemas ──────────────────────────────────────────────────────────

class FeedbackCreate(BaseModel):
    type: str = Field(..., description="Must be 'prediction_override', 'tactic_modification', or 'team_profile_edit'")
    
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
    
    # Extended team profile fields
    suggested_attack_headline: Optional[str] = None
    suggested_defense_headline: Optional[str] = None
    suggested_strengths: Optional[str] = None
    suggested_weaknesses: Optional[str] = None

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
    
    # Extended team profile fields
    suggested_attack_headline: Optional[str] = None
    suggested_defense_headline: Optional[str] = None
    suggested_strengths: Optional[str] = None
    suggested_weaknesses: Optional[str] = None

    admin_notes: Optional[str] = None
    reviewed_by_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# ── Match Submission Schemas ──────────────────────────────────────────────────

class MatchSubmissionCreate(BaseModel):
    home_team: str
    away_team: str
    match_date: date
    league: str
    season: str
    home_goals: int = Field(ge=0)
    away_goals: int = Field(ge=0)
    home_ht_goals: int = Field(ge=0)
    away_ht_goals: int = Field(ge=0)
    home_xg: Optional[float] = None
    away_xg: Optional[float] = None
    home_shots: int = Field(ge=0)
    away_shots: int = Field(ge=0)
    home_sot: int = Field(ge=0)
    away_sot: int = Field(ge=0)
    home_corners: int = Field(ge=0)
    away_corners: int = Field(ge=0)
    home_fouls: int = Field(ge=0)
    away_fouls: int = Field(ge=0)
    home_yellows: int = Field(ge=0)
    away_yellows: int = Field(ge=0)
    home_reds: int = Field(ge=0)
    away_reds: int = Field(ge=0)

class MatchSubmissionResponse(BaseModel):
    id: int
    user_id: int
    status: str
    home_team: str
    away_team: str
    match_date: date
    league: str
    season: str
    home_goals: int
    away_goals: int
    home_ht_goals: int
    away_ht_goals: int
    home_xg: Optional[float] = None
    away_xg: Optional[float] = None
    home_shots: int
    away_shots: int
    home_sot: int
    away_sot: int
    home_corners: int
    away_corners: int
    home_fouls: int
    away_fouls: int
    home_yellows: int
    away_yellows: int
    home_reds: int
    away_reds: int
    admin_notes: Optional[str] = None
    reviewed_by_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# ── Tactical Analysis Schemas ─────────────────────────────────────────────────

class TacticalAnalysisCreate(BaseModel):
    home_team: str
    away_team: str
    match_date: date
    analysis_text: str = Field(min_length=20, max_length=10000)

class TacticalAnalysisResponse(BaseModel):
    id: int
    user_id: int
    status: str
    home_team: str
    away_team: str
    match_date: date
    analysis_text: str
    admin_notes: Optional[str] = None
    reviewed_by_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# ── Team Profile Edit Schema ──────────────────────────────────────────────────

class TeamProfileEditCreate(BaseModel):
    team_name: str
    suggested_attack_tactic: Optional[str] = None
    suggested_defense_tactic: Optional[str] = None
    suggested_attack_headline: Optional[str] = None
    suggested_defense_headline: Optional[str] = None
    suggested_strengths: Optional[str] = None
    suggested_weaknesses: Optional[str] = None

# ── Unified Supervisor Schemas ─────────────────────────────────────────────────

class SubmissionReviewRequest(BaseModel):
    status: str = Field(..., description="Must be 'approved' or 'rejected'")
    admin_notes: Optional[str] = None

# ── Knowledge Base Schemas (external access) ───────────────────────────────────

class KBRetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1)
    prefer_prediction: bool = False

class KBAskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    llm_provider: Optional[str] = Field(
        default=None,
        description="None/empty → structured answer without any LLM. "
                    "Or a registered provider: ollama, openai, gemini, anthropic, huggingface."
    )
    prefer_prediction: bool = False

class KBBundleResponse(BaseModel):
    question: str
    intent: str
    teams: List[str] = Field(default_factory=list)
    league: Optional[str] = None
    season: Optional[str] = None
    facts: List[dict] = Field(default_factory=list)
    tables: List[dict] = Field(default_factory=list)
    vector_hits: List[dict] = Field(default_factory=list)
    sources: List[dict] = Field(default_factory=list)

class KBAnswerResponse(BaseModel):
    content: str
    provider: str
    error: Optional[str] = None
    sources: List[dict] = Field(default_factory=list)
