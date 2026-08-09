"""
PlayerRecord — the provider-agnostic player schema for the best-11 feature.

Every player provider (FBRef, football-data.org, Understat) normalizes its
output into PlayerRecord so the rest of the pipeline (ratings, solver, KB)
never sees provider-specific shapes. Fields that a provider cannot fill are
left None / empty — the probe report measures that coverage.

Season format: 4-char football-data.co.uk style ('2425').
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class PlayerRecord:
    name: str
    team: str                       # canonical team name (team_registry style)
    league: str                     # 'Premier_League', 'La_Liga', ...
    season: str                     # '2425'
    source: str                     # 'fbref' | 'fod' | 'understat'

    # Identity / bio
    position: Optional[str] = None  # GK | DF | MF | FW (or raw provider value)
    age: Optional[float] = None
    nationality: Optional[str] = None

    # Playing time
    appearances: Optional[int] = None
    minutes: Optional[float] = None

    # Attack
    goals: Optional[float] = None
    assists: Optional[float] = None
    xg: Optional[float] = None
    xa: Optional[float] = None
    shots: Optional[float] = None
    shots_on_target: Optional[float] = None

    # Defense / GK
    tackles: Optional[float] = None
    interceptions: Optional[float] = None
    blocks: Optional[float] = None
    clearances: Optional[float] = None
    errors: Optional[float] = None
    saves: Optional[float] = None
    clean_sheets: Optional[float] = None
    goals_conceded: Optional[float] = None

    # Creation (FBRef passing table; understat reports key passes)
    key_passes: Optional[float] = None
    progressive_passes: Optional[float] = None

    # Discipline
    yellow_cards: Optional[int] = None
    red_cards: Optional[int] = None

    # Provider-specific extras (kept for diagnostics, never relied on)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
