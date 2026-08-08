"""
TeamTotals — season aggregates of the match-dataset features per team.

Reads data/processed/processed_matches.csv (the same dataset the ML models
train on) and collapses the per-match feature families — goals, xG, shots,
SoT, fouls, corners, cards, clean sheets — into per-team season totals.
These totals are the denominators for the player share ratings
(data/player_ratings.py): a player's xG/90 ÷ team xG/90 = "share of the
team's xG this player produced".

League codes are the config keys (E0, SP1, D1, I1, F1); team names are
canonicalized via team_registry.normalize_team_name so they match the
player providers' canonical names.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from data._config import get_leagues
from data.team_registry import normalize_team_name

logger = logging.getLogger(__name__)

_LEAGUE_NAME_TO_CODE = {info["name"]: code for code, info in get_leagues().items()}
_PROCESSED = Path("data/processed/processed_matches.csv")

_XG_COLS = ("xg", "xga")

# (team_feature_name, home_csv_col, away_csv_col)
_AGG_MAP = [
    ("gf", "FTHG", "FTAG"),
    ("ga", "FTAG", "FTHG"),
    ("xg", "Home_xG", "Away_xG"),
    ("xga", "Away_xG", "Home_xG"),
    ("shots", "HS", "AS"),
    ("shots_against", "AS", "HS"),
    ("sot", "HST", "AST"),
    ("sot_against", "AST", "HST"),
    ("fouls", "HF", "AF"),
    ("fouls_against", "AF", "HF"),
    ("corners", "HC", "AC"),
    ("corners_against", "AC", "HC"),
    ("yellows", "HY", "AY"),
    ("reds", "HR", "AR"),
]


@dataclass
class TeamTotals:
    team: str
    league_code: str
    season: str
    matches: int = 0
    gf: float = 0.0
    ga: float = 0.0
    xg: float = 0.0
    xga: float = 0.0
    shots: float = 0.0
    shots_against: float = 0.0
    sot: float = 0.0
    sot_against: float = 0.0
    fouls: float = 0.0
    fouls_against: float = 0.0
    corners: float = 0.0
    corners_against: float = 0.0
    yellows: float = 0.0
    reds: float = 0.0
    clean_sheets: int = 0

    def per90(self, stat: str) -> float:
        if not self.matches:
            return 0.0
        return getattr(self, stat, 0.0) / self.matches


def load_team_totals(league_code: str, season: str,
                     as_of: Optional[str] = None) -> Dict[str, TeamTotals]:
    """Aggregate one league-season from the processed match dataset.

    Returns {canonical_team_name: TeamTotals}. Missing/blank xG cells are
    ignored (their matches still count toward matches / clean sheets).

    With as_of (ISO date 'YYYY-MM-DD') only matches played on or before
    that date are included — used for through-the-season player ratings
    so share denominators match the player's cumulative stats.
    """
    league_name = get_leagues()[league_code]["name"]
    df = pd.read_csv(_PROCESSED)
    df = df[(df["League"] == league_name) & (df["Season"].astype(str) == str(season))]
    if as_of:
        df = df[pd.to_datetime(df["Date"], errors="coerce") <= pd.Timestamp(as_of)]
    if df.empty:
        logger.warning("team_totals: no %s %s rows in %s", league_name, season, _PROCESSED)
        return {}

    frames = []
    for side, team_col, opp_col in (("home", "HomeTeam", "AwayTeam"),
                                    ("away", "AwayTeam", "HomeTeam")):
        base = pd.DataFrame(index=df.index)
        base["side"] = side
        base["team_name"] = df[team_col].map(lambda t: normalize_team_name(str(t), league_code))
        for feat, home_col, away_col in _AGG_MAP:
            col = home_col if side == "home" else away_col
            base[feat] = pd.to_numeric(df[col], errors="coerce")
        conceded_col = "FTAG" if side == "home" else "FTHG"
        base["clean_sheet"] = (pd.to_numeric(df[conceded_col], errors="coerce") == 0)
        frames.append(base)

    combined = pd.concat(frames, ignore_index=True)
    totals: Dict[str, TeamTotals] = {}
    for team, g in combined.groupby("team_name"):
        tt = TeamTotals(team=team, league_code=league_code, season=str(season),
                        matches=int(len(g)))
        for feat, _, _ in _AGG_MAP:
            setattr(tt, feat, float(g[feat].sum()))
        tt.clean_sheets = int(g["clean_sheet"].sum())
        totals[team] = tt
    return totals


def load_all_totals(season: str) -> Dict[str, Dict[str, TeamTotals]]:
    """{league_code: {team: TeamTotals}} for every configured league."""
    return {code: load_team_totals(code, season) for code in _LEAGUE_NAME_TO_CODE.values()}
