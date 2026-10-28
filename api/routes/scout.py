"""REST route for the scouting feature (HTTP concerns only).

The business logic lives in ScoutApiService (application layer), which
delegates to the domain ScoutService + repositories — see
api/services/scout_service.py.
"""

from typing import Optional

from fastapi import APIRouter, Depends, Query

from api.dependencies import get_scout_api_service
from api.schemas import ScoutResponse
from api.services.scout_service import ScoutApiService
from data._config import resolve_default_player_season

router = APIRouter(prefix="/scout", tags=["Scouting"])


@router.get("", response_model=ScoutResponse,
            summary="Top-N scouting candidates to sign")
async def get_scout(
    position: str = Query(..., description="Position to sign: GK | DF | MF | FW"),
    league: str = Query("all", description="League code to scout: all | E0 | SP1 | D1 | I1 | F1"),
    season: Optional[str] = Query(
        default=None,
        description="Season code (defaults dynamically to latest completed/cached season, e.g. '2425')"),
    youth: bool = Query(False, description="Only players aged 19 or younger (academy targets)"),
    top: int = Query(5, ge=1, le=20, description="Number of candidates to return (1..20)"),
    team_needing: Optional[str] = Query(
        None, description="Your team name — excluded from the pool so you scout the competition"),
    refresh: bool = Query(False, description="Force-refresh cached stats from API-Football"),
    service: ScoutApiService = Depends(get_scout_api_service),
) -> ScoutResponse:
    """Return the top-N candidates to sign, ranked by a position-specific score."""
    resolved_season = season or resolve_default_player_season()
    return await service.scout(league, resolved_season, position, youth, top, team_needing, refresh)
