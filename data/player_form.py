"""
Through-the-season player ratings (data/player_form.py).

Ratings are normally computed from *full-season* aggregates (squad cache +
team totals), which leaks future information when predicting a past
fixture: a match played in March would be rated with stats from April
and May. This module re-rates a squad using only per-match player stats
(understat getMatchData, collected by
data/collectors/player_match_stats.py) and team totals *up to* a given
date, so the XI for a specific match reflects each player's form and
minutes through that date.

Fusion:
  - attack + minutes (minutes, goals, assists, shots, xG, xA, key passes,
    cards): replaced by cumulative per-match values through the date;
  - defense / GK (tackles, interceptions, SoT, saves, clean sheets, GA):
    not in understat per-match data → season totals scaled by the share
    of minutes played before the date;
  - players missing from the per-match feed: season totals scaled by the
    team's match progress before the date (keeps them comparable).

Fallback: if per-match data is missing (not yet collected / other
season), the full-season ratings are returned unchanged with a flag.
"""

import logging
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from data.player_providers.schema import PlayerRecord
from data.player_ratings import PlayerRating, rate_player, normalize_ratings
from data.team_registry import normalize_team_name
from data.team_totals import load_team_totals

logger = logging.getLogger(__name__)

_PM_DIR = Path("data/raw/player_match")

_STAT_COLS = ["minutes", "goals", "assists", "shots", "xg", "xa",
              "key_passes", "yellow_cards", "red_cards"]
_SCALED_COLS = ["tackles", "interceptions", "blocks", "clearances", "errors",
                "progressive_passes", "shots_on_target", "saves",
                "clean_sheets", "goals_conceded"]


def load_match_player_stats(league_code: str, season: str) -> pd.DataFrame:
    """Per-match player rows for one league-season (empty if uncollected)."""
    path = _PM_DIR / f"player_match_{league_code}_{season}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    return df


def cumulative_player_stats(league_code: str, season: str, as_of: str,
                            team: str) -> Dict[str, Dict[str, float]]:
    """{player_name: {stat: total}} for one team, matches through as_of."""
    df = load_match_player_stats(league_code, season)
    if df.empty:
        return {}
    as_of_ts = pd.Timestamp(as_of)
    df = df[df.match_date <= as_of_ts]
    if df.empty:
        return {}
    canon = df["team"].map(lambda t: normalize_team_name(str(t), league_code))
    df = df[canon == team]
    if df.empty:
        return {}
    agg = df.groupby("player_name")[_STAT_COLS].sum()
    return agg.to_dict("index")


def latest_match_date(league_code: str, season: str) -> Optional[str]:
    """Latest match date in the per-match feed (ISO 'YYYY-MM-DD'), or None.

    Used to auto-switch best-11 to cumulative-through-date ratings while a
    season is in progress, so the lineup never leaks future matches.
    """
    df = load_match_player_stats(league_code, season)
    if df.empty or df["match_date"].isna().all():
        return None
    return df["match_date"].max().strftime("%Y-%m-%d")


def _norm_name(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return s.strip().lower().replace(" ", "")


def rate_squad_as_of(squad: List[PlayerRecord], as_of: str,
                     league_code: str, season: str
                     ) -> Tuple[List[PlayerRating], int, bool]:
    """Rate a squad with stats through `as_of`.

    Returns (ratings, players_matched, used_through). When per-match data
    is unavailable, falls back to full-season ratings with used_through
    False. Team totals are also filtered to matches before as_of, so
    share denominators stay consistent.
    """
    totals = load_team_totals(league_code, season, as_of=as_of)
    tt = totals.get(squad[0].team) if squad else None
    if tt is None or tt.matches == 0:
        logger.warning("rate_squad_as_of: no team totals through %s for %s",
                       as_of, squad[0].team if squad else "?")
        return rate_squad_full(squad, totals), 0, False

    team = squad[0].team
    season_full = load_team_totals(league_code, season)
    full_tt = season_full.get(team)
    progress = (tt.matches / full_tt.matches) if full_tt and full_tt.matches else 1.0

    per_player = cumulative_player_stats(league_code, season, as_of, team)
    norm_to_stats = {_norm_name(n): v for n, v in per_player.items()}

    adjusted: List[PlayerRecord] = []
    matched = 0
    for rec in squad:
        cum = per_player.get(rec.name) or norm_to_stats.get(_norm_name(rec.name or ""))
        if cum is None:
            ratio = progress
            scaled = {}
            for c in _STAT_COLS:
                if c == "key_passes":
                    continue
                scaled[c] = (getattr(rec, c, None) or 0.0) * ratio
            def_scale = {c: ((getattr(rec, c) or 0.0) * ratio) for c in _SCALED_COLS}
            extra = dict(rec.extra or {})
            extra["key_passes"] = (extra.get("key_passes") or 0.0) * ratio
            adjusted.append(replace(rec, **scaled, **def_scale, extra=extra))
            continue
        matched += 1
        min_ratio = min(cum.get("minutes", 0.0) / max(rec.minutes or 0.0, 1.0), 1.0)
        def_scale = {c: ((getattr(rec, c) or 0.0) * min_ratio) for c in _SCALED_COLS}
        adjusted.append(replace(
            rec,
            minutes=cum.get("minutes", 0.0),
            goals=cum.get("goals", 0.0),
            assists=cum.get("assists", 0.0),
            shots=cum.get("shots", 0.0),
            xg=cum.get("xg", 0.0),
            xa=cum.get("xa", 0.0),
            yellow_cards=int(cum.get("yellow_cards", 0)),
            red_cards=int(cum.get("red_cards", 0)),
            **def_scale,
        ))

    team_xa = sum((r.xa or 0.0) for r in adjusted) or None
    ratings = [rate_player(r, tt, team_xa) for r in adjusted]
    normalize_ratings(ratings)
    return ratings, matched, True


def rate_squad_full(squad: List[PlayerRecord],
                    totals: Dict[str, object]) -> List[PlayerRating]:
    """Full-season ratings (same as data.player_ratings.rate_squad)."""
    from data.player_ratings import rate_squad
    return rate_squad(squad, totals)


def h2h_player_stats(league_code: str, season: str, team: str, opponent: str
                     ) -> Tuple[Dict[str, Dict[str, float]], int]:
    """Aggregate per-player stats from matches between `team` and `opponent`.

    Returns ({player_name: {minutes, goals, assists, shots, xg, xa,
    key_passes}}, n_matches). Empty dict when the per-match feed for this
    league-season is missing or the pair never met.
    """
    df = load_match_player_stats(league_code, season)
    if df.empty:
        return {}, 0
    canon = lambda s: normalize_team_name(str(s), league_code)
    mask = (((df["home_team"].map(canon) == team) & (df["away_team"].map(canon) == opponent)) |
            ((df["home_team"].map(canon) == opponent) & (df["away_team"].map(canon) == team)))
    h2h = df[mask]
    if h2h.empty:
        return {}, 0
    # rows carry both teams' players → keep only the requesting team
    h2h = h2h[h2h["team"].map(canon) == team]
    if h2h.empty:
        return {}, 0
    n_matches = int(h2h["match_id"].nunique())
    out: Dict[str, Dict[str, float]] = {}
    for name, g in h2h.groupby("player_name"):
        out[str(name)] = {c: float(g[c].sum()) for c in
                          ("minutes", "goals", "assists", "shots", "xg", "xa",
                           "key_passes")}
    return out, n_matches
