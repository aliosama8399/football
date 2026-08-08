"""
PlayerRatings — team-share ratings for the best-11 feature.

Model: a player's season stats are expressed as a SHARE of the team's
match-dataset totals (data/processed/processed_matches.csv, aggregated by
data/team_totals.py). Shares are interpretable — "this player produced
9% of Arsenal's xG" — and comparable across teams of different volume.

Per position bucket the raw score is a weighted blend of the relevant
share features, then min-max rescaled to 0-100 within the squad's
position bucket (the solver only compares teammates) with a
minutes-credibility discount. Missing features simply drop out and the
remaining weights renormalize, so fused records with partial stats still
get a rating.

Weights are the defaults from the plan; they are module constants so the
KB can quote them. Position buckets: GK | DF | MF | FW.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from data.player_providers.schema import PlayerRecord
from data.team_totals import TeamTotals

logger = logging.getLogger(__name__)

MIN_MINUTES = 360.0     # below this a player cannot start (noise filter)
FULL_CREDIT_MINUTES = 900.0  # ~10 league games: beyond this, full credit
CARD_PENALTY = {"red": 0.10, "yellow": 0.03}

# position bucket → [(share feature, weight), ...]
_POSITION_WEIGHTS: Dict[str, List[tuple]] = {
    "FW": [("xg_share", 0.40), ("goals_share", 0.30), ("sot_share", 0.15),
           ("shots_share", 0.10), ("xa_share", 0.05)],
    "MF": [("xa_share", 0.30), ("build_share", 0.20), ("xg_share", 0.20),
           ("def_share", 0.20), ("goals_share", 0.10)],
    "DF": [("def_share", 0.40), ("def_volume_share", 0.20), ("xg_share", 0.15),
           ("cs_share", 0.10), ("xa_share", 0.10), ("goals_share", 0.05)],
    "GK": [("save_rate", 0.40), ("cs_rate", 0.25), ("ga_xg_ratio", 0.25),
           ("minutes_share", 0.10)],
}


@dataclass
class PlayerRating:
    name: str
    position: Optional[str]
    minutes: float
    rating: float          # 0-100, within-squad position bucket, minutes-discounted
    raw_score: float       # pre-normalization weighted share blend
    shares: dict           # share feature name → value (for KB quotes)
    per90: dict            # player per-90 stats
    pos_list: list = None  # all FBRef/understat positions, for lineup flexibility


def _per90(rec: PlayerRecord, field_name: str) -> Optional[float]:
    v = getattr(rec, field_name, None)
    if v is None or not rec.minutes:
        return None
    return v / (rec.minutes / 90.0)


def _share_t(player_total: Optional[float], team_total: float) -> Optional[float]:
    """player season total ÷ team season total — share of team output.

    Totals (not per-90) so shares close to 1 across the squad and stay
    interpretable ("this player produced 9% of the team's xG"). Minutes
    are handled by the credibility discount, not here.
    """
    if player_total is None or team_total <= 0:
        return None
    return min(player_total / team_total, 2.0)


def _build_shares(rec: PlayerRecord, tt: TeamTotals,
                  team_xa: Optional[float] = None) -> Dict[str, Optional[float]]:
    s = _share_t
    tackles_int = (rec.tackles or 0) + (rec.interceptions or 0)
    # xA denominator: squad-level player xA when available (the processed
    # match dataset has no xA), else team xG as the historic proxy.
    xa_denom = team_xa if team_xa is not None and team_xa > 0 else tt.xg
    return {
        "xg_share": s(rec.xg, tt.xg),
        "goals_share": s(rec.goals, tt.gf),
        "shots_share": s(rec.shots, tt.shots),
        "sot_share": s(rec.shots_on_target, tt.sot),
        "xa_share": s(rec.xa, xa_denom),
        "cs_share": s(rec.clean_sheets, tt.clean_sheets),
        "def_share": s(tackles_int, tt.sot_against),
        "def_volume_share": s(tackles_int, tt.sot_against + tt.shots_against),
        "build_share": _build_share(rec, tt),
        "save_rate": _save_rate(rec),
        "cs_rate": _cs_rate(rec, tt),
        "ga_xg_ratio": _ga_xg_ratio(rec, tt),
        "minutes_share": _minutes_share(rec, tt),
        "red_share": s(rec.red_cards, tt.reds),
        "yellow_share": s(rec.yellow_cards, tt.yellows),
    }


def _build_share(rec: PlayerRecord, tt: TeamTotals) -> Optional[float]:
    chain = (rec.extra or {}).get("xg_chain")
    buildup = (rec.extra or {}).get("xg_buildup")
    if chain is None or buildup is None or tt.xg <= 0:
        return None
    return min(((chain + buildup) / 2) / tt.xg, 2.0)


def _save_rate(rec: PlayerRecord) -> Optional[float]:
    if rec.saves is None or rec.goals_conceded is None:
        return None
    faced = rec.saves + rec.goals_conceded
    if faced <= 0:
        return None
    return min(rec.saves / faced, 1.0)


def _cs_rate(rec: PlayerRecord, tt: TeamTotals) -> Optional[float]:
    if rec.clean_sheets is None or tt.matches <= 0:
        return None
    return min(rec.clean_sheets / tt.matches, 1.0)


def _ga_xg_ratio(rec: PlayerRecord, tt: TeamTotals) -> Optional[float]:
    """Conceding vs expectation: team xGA/90 ÷ player GA/90 (>1 = better)."""
    ga90 = _per90(rec, "goals_conceded")
    xga90 = tt.per90("xga")
    if ga90 is None or xga90 <= 0 or ga90 <= 0:
        return None
    return min(xga90 / ga90, 2.0)


def _minutes_share(rec: PlayerRecord, tt: TeamTotals) -> Optional[float]:
    if not rec.minutes or tt.matches <= 0:
        return None
    return min(rec.minutes / (tt.matches * 90.0), 1.0)


def raw_score(rec: PlayerRecord, tt: TeamTotals,
              team_xa: Optional[float] = None) -> float:
    """Weighted blend of share features for the player's position bucket.

    Every bucket only uses the metrics that exist for that position
    (GK: saves/clean sheets/GA; DF: defense + clean sheets; MF: creation +
    buildup + defense; FW: xG/goals/shot volume), and per-player missing
    metrics (e.g. xA for a striker who never creates) drop out of the
    blend with the remaining weights renormalized — a player is never
    penalized for lacking a metric that isn't part of their position.
    """
    shares = _build_shares(rec, tt, team_xa)
    bucket = rec.position or "MF"
    weights = _POSITION_WEIGHTS.get(bucket, _POSITION_WEIGHTS["MF"])
    present = [(f, w) for f, w in weights if shares.get(f) is not None]
    if not present:
        return 0.0
    wsum = sum(w for _, w in present)
    score = sum(shares[f] * w for f, w in present) / wsum
    red = shares.get("red_share") or 0.0
    yellow = shares.get("yellow_share") or 0.0
    score -= CARD_PENALTY["red"] * red + CARD_PENALTY["yellow"] * yellow
    return score


def normalize_ratings(ratings: List[PlayerRating]) -> List[PlayerRating]:
    """Min-max 0-100 per position bucket, then minutes-credibility discount.

    Two buckets (e.g. two GKs) use their own min/max; a singleton bucket
    scores 100 unless discounted by minutes.
    """
    from collections import defaultdict

    buckets: Dict[str, List[PlayerRating]] = defaultdict(list)
    for r in ratings:
        buckets[r.position or "MF"].append(r)
    for group in buckets.values():
        lo = min(r.raw_score for r in group)
        hi = max(r.raw_score for r in group)
        span = hi - lo if hi > lo else 1.0
        for r in group:
            r.rating = 100.0 * (r.raw_score - lo) / span
            r.rating *= min(1.0, r.minutes / FULL_CREDIT_MINUTES)
    return ratings


def rate_player(rec: PlayerRecord, tt: Optional[TeamTotals],
                team_xa: Optional[float] = None) -> PlayerRating:
    shares = (_build_shares(rec, tt, team_xa) if tt is not None
              else {k: None for k in _build_shares(rec, tt, team_xa)})
    return PlayerRating(
        name=rec.name,
        position=rec.position,
        minutes=rec.minutes or 0.0,
        rating=0.0,
        raw_score=raw_score(rec, tt, team_xa) if tt is not None else 0.0,
        shares=shares,
        per90={f: _per90(rec, f) for f in
               ("goals", "assists", "xg", "xa", "shots", "shots_on_target",
                "tackles", "interceptions", "saves", "goals_conceded")},
        pos_list=list(rec.extra.get("pos_list", [])) if rec.extra else [],
    )


def _team_xa(squad: List[PlayerRecord]) -> Optional[float]:
    """Squad-level xA: sum of the players' reported xA (no team xA in the
    processed match dataset). None when nobody reports xA."""
    total = sum((rec.xa or 0.0) for rec in squad)
    return total if total > 0 else None


def rate_squad(squad: List[PlayerRecord], totals: Dict[str, TeamTotals]) -> List[PlayerRating]:
    """Rate a full squad; team totals keyed by canonical team name."""
    tt = totals.get(squad[0].team) if squad else None
    team_xa = _team_xa(squad) if squad else None
    ratings = [rate_player(rec, tt, team_xa) for rec in squad]
    return normalize_ratings(ratings)
