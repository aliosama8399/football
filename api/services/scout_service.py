"""ScoutApiService — backend (HTTP-facing) application service for scouting.

SOLID layering of the scouting feature:

    api/routes/scout.py               → HTTP concerns (query params, response)
    ScoutApiService                   → application use-case: league mapping,
                                        validation, blocking-work offload, error
                                        mapping to HTTP
    api/services/scout/service.py     → domain orchestration (ScoutService)
    api/services/scout/strategies/    → the processing layer (position scoring)
    api/repositories/scout_repo.py    → collects all data from sources
                                        (Repository pattern)
    data/                             → raw collection & scraping only:
                                        player_providers/, team_totals.py, ...

The GraphQL resolver and the REST route share this service, so league
resolution, validation and error semantics live in exactly one place.
"""

import asyncio
import logging
from typing import Optional, Tuple

from fastapi import HTTPException, status

from api.repositories.scout_repo import SCOUT_LEAGUES, ScoutRepository
from api.schemas import ScoutCandidate, ScoutResponse
from api.services.scout import ScoutService as DomainScoutService

logger = logging.getLogger(__name__)

POSITIONS = ("GK", "DF", "MF", "FW")


class ScoutApiService:
    """Application service exposing the scouting use case to HTTP clients."""

    def __init__(self, repository: Optional[ScoutRepository] = None,
                 domain_service: Optional[DomainScoutService] = None):
        self.repository = repository or ScoutRepository()
        self.domain = domain_service or DomainScoutService(self.repository)

    def resolve_league_codes(self, league: Optional[str]) -> Tuple[str, ...]:
        """Map a league filter to codes. None/'all' → the full scouting scope
        (all configured leagues). 'E0'/'SP1'/... pass through."""
        if not league or league.lower() in ("all", "ALL"):
            return SCOUT_LEAGUES
        code = league.upper()
        if code in SCOUT_LEAGUES:
            return (code,)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown league '{league}'. Scouting scope: "
                   f"{', '.join(SCOUT_LEAGUES)} (or 'all').")

    async def scout(self, league: str, season: str, position: str,
                    youth: bool = False, top: int = 5,
                    team_needing: Optional[str] = None,
                    refresh: bool = False) -> ScoutResponse:
        """Scout and validate a top-N candidate list.

        Raises HTTPException (404 unknown league, 422 bad position) so
        both the REST route and the GraphQL resolver can rely on it.
        The blocking domain scout runs off the event loop.
        """
        league_codes = self.resolve_league_codes(league)

        position = position.upper()
        if position not in POSITIONS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Unsupported position '{position}'. "
                       f"Supported: {', '.join(POSITIONS)}.")

        result = await asyncio.to_thread(
            self.domain.scout, league_codes, season, position,
            top, youth, team_needing, refresh)

        return ScoutResponse(
            season=result.get("season", season),
            position=result.get("position", position),
            youth=result.get("youth", youth),
            leagues=result.get("leagues", list(league_codes)),
            pool_size=result.get("pool_size", 0),
            top=result.get("top", top),
            candidates=[ScoutCandidate(**c) for c in result.get("candidates", [])],
            notes=result.get("notes") or [],
        )
