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

# ── Best-11 (team-share ratings) types ────────────────────────────────────────

@strawberry.type
class Best11Season:
    goals: float = 0.0
    assists: float = 0.0
    xg: float = 0.0
    xa: float = 0.0
    shots: float = 0.0

@strawberry.type
class Best11H2H:
    matches: int = 0
    minutes: int = 0
    goals: int = 0
    assists: int = 0
    xg: float = 0.0
    xa: float = 0.0
    shots: int = 0

@strawberry.type
class Best11Entry:
    slot: str                       # GK | DF | MF | FW
    name: str
    position: Optional[str]         # provider position (GK/DF/MF/FW)
    rating: float                   # 0-100 within squad position bucket
    minutes: int
    flex: bool                      # true if promoted from another bucket
    top_shares: List[JSON] = strawberry.field(default_factory=list)
    season: Optional[Best11Season] = None   # full-season stats (verification)
    h2h: Optional[Best11H2H] = None         # stats vs the opponent (verification)

@strawberry.type
class Best11Sub:
    slot: str                       # position slot of the replacement
    out: str                        # starter replaced
    in_: str = strawberry.field(name="in")  # substitute brought on
    rating_delta: float
    reason: str

@strawberry.type
class Best11Result:
    team: str
    league_code: str
    season: str
    formation: str
    lineup: List[Best11Entry] = strawberry.field(default_factory=list)
    captain: Optional[str] = None
    subs: List[Best11Sub] = strawberry.field(default_factory=list)
    bench: List[JSON] = strawberry.field(default_factory=list)
    notes: List[str] = strawberry.field(default_factory=list)
    error: Optional[str] = None
