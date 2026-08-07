import strawberry
from typing import List, Optional

JSON = strawberry.scalars.JSON

# ── Knowledge Base types (external LLM integration) ───────────────────────────

@strawberry.type
class KBSource:
    ref: str
    title: str
    text: str
    source_type: str
    team: Optional[str] = None
    league: Optional[str] = None
    season: Optional[str] = None
    doc_id: Optional[str] = None

@strawberry.type
class KBBundle:
    question: str
    intent: str
    teams: List[str] = strawberry.field(default_factory=list)
    league: Optional[str] = None
    season: Optional[str] = None
    facts: List[JSON] = strawberry.field(default_factory=list)
    tables: List[JSON] = strawberry.field(default_factory=list)
    vector_hits: List[JSON] = strawberry.field(default_factory=list)
    sources: List[KBSource] = strawberry.field(default_factory=list)

@strawberry.type
class KBAnswer:
    content: str
    provider: str
    error: Optional[str] = None
    sources: List[KBSource] = strawberry.field(default_factory=list)

@strawberry.type
class TeamNode:
    name: str
    league: str
    total_matches: int
    win_rate: float
    draw_rate: float
    loss_rate: float
    clean_sheet_rate: float
    avg_goals_home: float
    avg_goals_away: float
    avg_xg: float
    avg_xga: float
    avg_shots: float
    avg_shots_against: float
    avg_sot: float
    avg_sot_against: float
    avg_corners: float
    avg_fouls: float
    avg_yellows: float
    attack_tactic: Optional[str] = None
    defense_tactic: Optional[str] = None
    attack_headline: Optional[str] = None
    defense_headline: Optional[str] = None
    strengths: List[str] = strawberry.field(default_factory=list)
    weaknesses: List[str] = strawberry.field(default_factory=list)

@strawberry.type
class MatchRelation:
    date: str
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    result: str
    home_xg: Optional[float] = None
    away_xg: Optional[float] = None
    league: str
    season: str
