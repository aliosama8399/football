"""Best11ApiService — backend (HTTP-facing) application service for best-11.

SOLID layering of the best-11 feature:

    api/routes/best11.py             → HTTP concerns (query params, response)
    Best11ApiService                 → application use-case: league mapping,
                                       validation, blocking-work offload, error
                                       mapping to HTTP
    api/services/best11/service.py   → domain orchestration (Best11Service)
    api/repositories/best11_repo.py  → collects all data from sources
                                       (Repository pattern)
    data/                            → raw collection & scraping only:
                                       player_providers/, collectors/,
                                       player_form.py, team_totals.py

The GraphQL resolver and the REST route share this service, so league
resolution, validation and error semantics live in exactly one place
(Single Responsibility + DRY).
"""

import asyncio
import logging
from typing import Optional

from fastapi import HTTPException, status

from api.repositories.best11_repo import Best11Repository
from api.schemas import Best11Bench, Best11Entry, Best11Response, Best11Sub
from api.services.best11 import Best11Service as DomainBest11Service
from data._config import get_leagues

logger = logging.getLogger(__name__)


class Best11ApiService:
    """Application service exposing the best-11 use case to HTTP clients."""

    def __init__(self, repository: Optional[Best11Repository] = None,
                 domain_service: Optional[DomainBest11Service] = None):
        self.repository = repository or Best11Repository()
        # The domain service consumes the repository (the collector) and
        # the strategies (the processing layer) to produce the prediction.
        self.domain = domain_service or DomainBest11Service(self.repository)
        self._league_codes = {info["name"]: code
                              for code, info in get_leagues().items()}

    def resolve_league_code(self, league: str) -> Optional[str]:
        """Map a config league name ('La_Liga') to its code ('SP1')."""
        return self._league_codes.get(league)

    async def recommend(self, team: str, league: str, season: str = "2425",
                        formation: str = "auto", as_of: Optional[str] = None,
                        opponent: Optional[str] = None) -> Best11Response:
        """Compute and validate a best-11 recommendation.

        Raises HTTPException (404 unknown league, 422 domain error) so
        both the REST route and the GraphQL resolver can rely on it.
        The blocking domain solve runs off the event loop.
        """
        league_code = self.resolve_league_code(league)
        if league_code is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown league '{league}'. "
                       f"Known leagues: {', '.join(sorted(self._league_codes))}"
            )

        result = await asyncio.to_thread(
            self.domain.solve, team, league_code, season, formation,
            "all", False, as_of, opponent)

        if result.get("error"):
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                                detail=result["error"])

        return Best11Response(
            team=result.get("team", team),
            league_code=result.get("league_code", league_code),
            season=result.get("season", season),
            formation=result.get("formation", formation),
            captain=result.get("captain"),
            lineup=[Best11Entry(**e) for e in result.get("lineup", [])],
            subs=[Best11Sub(**s) for s in result.get("subs", [])],
            bench=[Best11Bench(**b) for b in result.get("bench", [])],
            notes=result.get("notes") or [],
        )
