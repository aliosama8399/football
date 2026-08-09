"""data/players — best-11 feature, structured with design patterns.

- Repository: SquadRepository / TotalsRepository / PlayerFormRepository
  hide all data access (squad cache, team totals, per-match form stats).
- Strategy: rating modes (season / through-date / H2H-blend enhancer),
  formation choice, and substitution rules are interchangeable.
- Facade: Best11Service.solve() is the single entry point; the legacy
  solve_best11() / _load_squad_cached() module functions are thin
  wrappers kept for backward compatibility.
"""

import logging
from typing import Dict, List, Optional

from data.player_providers.schema import PlayerRecord

from .repository import (PlayerFormRepository, PlayerFormRepositoryABC,
                         SquadRepository, SquadRepositoryABC, TotalsRepository,
                         TotalsRepositoryABC, get_player_form_repository,
                         get_squad_repository, get_totals_repository)
from .service import Best11Service
from .strategies.formations import FORMATIONS
from .strategies.ratings import (H2HBlendDecorator, RatingOutcome,
                                 RatingStrategy, SeasonRatingStrategy,
                                 ThroughDateRatingStrategy)
from .strategies.substitutions import RotationSubstitutionStrategy

__all__ = [
    "SquadRepository", "SquadRepositoryABC", "get_squad_repository",
    "TotalsRepository", "TotalsRepositoryABC", "get_totals_repository",
    "PlayerFormRepository", "PlayerFormRepositoryABC", "get_player_form_repository",
    "Best11Service",
    "H2HBlendDecorator", "RatingOutcome", "RatingStrategy",
    "SeasonRatingStrategy", "ThroughDateRatingStrategy",
    "RotationSubstitutionStrategy", "FORMATIONS",
    "solve_best11", "_load_squad_cached",
]

_default_service: Optional[Best11Service] = None


def get_best11_service() -> Best11Service:
    """Process-wide singleton service."""
    global _default_service
    if _default_service is None:
        _default_service = Best11Service()
    return _default_service


def solve_best11(team: str, league_code: str, season: str = "2425",
                 formation: str = "auto", provider: str = "all",
                 refresh: bool = False, as_of: Optional[str] = None,
                 opponent: Optional[str] = None) -> Dict:
    """Legacy module-level entry point (backward compatible)."""
    return get_best11_service().solve(team, league_code, season, formation,
                                      provider, refresh, as_of, opponent)


def _load_squad_cached(team: str, league_code: str, season: str,
                       provider: str = "all", refresh: bool = False) -> List[PlayerRecord]:
    """Legacy module-level repository access (backward compatible)."""
    return get_squad_repository().load_squad(team, league_code, season, provider, refresh)


logging.getLogger("best11").info(
    "data/players pattern package loaded (Repository + Strategy + Facade)")
