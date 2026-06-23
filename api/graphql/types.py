import strawberry
from typing import List, Optional

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
