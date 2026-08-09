"""Rating strategies for the best-11 feature.

Strategy pattern: every rating mode implements RatingStrategy.rate()
returning a RatingOutcome (ratings + through-date metadata). The
H2HBlendDecorator is a RatingEnhancer applied on top of any strategy,
so match-specific lineups compose with either full-season or
through-date ratings.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from data.player_ratings import PlayerRating, rate_squad

from ..repository import (PlayerFormRepositoryABC, TotalsRepositoryABC,
                          get_player_form_repository, get_totals_repository)


@dataclass
class RatingOutcome:
    ratings: List[PlayerRating]
    matched: int = 0          # players matched by the per-match feed
    used_through: bool = False  # True when through-date stats were used


class RatingStrategy(ABC):
    """Base class for computing a squad's player ratings."""

    @abstractmethod
    def rate(self, squad, league_code: str, season: str,
             as_of: Optional[str]) -> RatingOutcome:
        """Rate a squad; as_of triggers through-date behavior per strategy."""


class SeasonRatingStrategy(RatingStrategy):
    """Full-season ratings from the processed match dataset."""

    def __init__(self, totals_repository: Optional[TotalsRepositoryABC] = None):
        self.totals_repository = totals_repository or get_totals_repository()

    def rate(self, squad, league_code: str, season: str,
             as_of: Optional[str]) -> RatingOutcome:
        totals = self.totals_repository.load_totals(league_code, season)
        ratings = rate_squad(squad, totals)
        return RatingOutcome(ratings=ratings, matched=0, used_through=False)


class ThroughDateRatingStrategy(RatingStrategy):
    """Cumulative-through-as_of ratings (no future leakage).

    Uses per-match player stats + team totals up to as_of; falls back to
    full-season ratings when the per-match feed is unavailable.
    """

    def __init__(self, form_repository: Optional[PlayerFormRepositoryABC] = None,
                 totals_repository: Optional[TotalsRepositoryABC] = None):
        self.form_repository = form_repository or get_player_form_repository()
        self.totals_repository = totals_repository or get_totals_repository()

    def rate(self, squad, league_code: str, season: str,
             as_of: Optional[str]) -> RatingOutcome:
        try:
            ratings, matched, used_through = self.form_repository.rate_squad_as_of(
                squad, as_of, league_code, season)
            return RatingOutcome(ratings=ratings, matched=matched,
                                 used_through=used_through)
        except Exception:
            # fall back to full-season ratings (per-match feed missing)
            return SeasonRatingStrategy(self.totals_repository).rate(
                squad, league_code, season, None)


class RatingEnhancer(ABC):
    """Optional post-processing applied to a rating outcome."""

    @abstractmethod
    def enhance(self, ratings: List[PlayerRating]) -> None: ...


class H2HBlendDecorator(RatingEnhancer):
    """Boost-only 70/30 season↔H2H blend against a specific opponent.

    H2H raw score uses the same share scheme as season ratings
    (xg .40, goals .30, shots .15, xa .15) over the meetings between the
    two teams, mapped to 0-100 on a fixed scale (raw .33 ≈ 100). Final
    rating = max(season, 0.7 × season + 0.3 × H2H rating), so players
    who produced against the opponent get a boost (cap +30 pts) and
    everyone else keeps their season rating — GKs, quiet defenders and
    bit-part players are never punished.
    """

    H2H_WEIGHTS = (("xg", 0.40), ("goals", 0.30), ("shots", 0.15), ("xa", 0.15))
    MIN_H2H_MINUTES = 90.0
    MIN_RAW = 0.03

    def __init__(self, h2h_stats: Dict[str, Dict[str, float]]):
        self.h2h_stats = h2h_stats

    def _totals(self) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        for v in self.h2h_stats.values():
            for f, _ in self.H2H_WEIGHTS:
                totals[f] = totals.get(f, 0.0) + (v.get(f) or 0.0)
        return totals

    def enhance(self, ratings: List[PlayerRating]) -> None:
        totals = self._totals()
        boosted: Dict[int, float] = {}
        for r in ratings:
            v = self.h2h_stats.get(r.name)
            if not v or (v.get("minutes") or 0) < self.MIN_H2H_MINUTES:
                continue
            score, n = 0.0, 0.0
            for f, w in self.H2H_WEIGHTS:
                t = totals.get(f) or 0.0
                p = v.get(f) or 0.0
                if t > 0:
                    score += min(p / t, 2.0) * w
                    n += w
            raw = (score / n) if n else 0.0
            if raw < self.MIN_RAW:  # no meaningful H2H output → keep rating
                continue
            h2h_rating = min(100.0, raw * 300.0)
            boosted[id(r)] = max(r.rating, 0.7 * r.rating + 0.3 * h2h_rating)
        for r in ratings:
            if id(r) in boosted:
                r.rating = boosted[id(r)]


def make_rating_strategy(as_of: Optional[str],
                         form_repository: Optional[PlayerFormRepositoryABC] = None,
                         totals_repository: Optional[TotalsRepositoryABC] = None) -> RatingStrategy:
    """Strategy factory: through-date when a date is given, else season."""
    if as_of:
        return ThroughDateRatingStrategy(form_repository, totals_repository)
    return SeasonRatingStrategy(totals_repository)
