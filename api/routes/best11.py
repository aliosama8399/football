"""REST route for the best-11 feature (HTTP concerns only).

The business logic lives in Best11ApiService (application layer), which
delegates to the domain Best11Service + repositories — see
api/services/best11_service.py.
"""

from typing import Optional

from fastapi import APIRouter, Depends, Query

from api.dependencies import get_best11_api_service
from api.schemas import Best11Response
from api.services.best11_service import Best11ApiService

router = APIRouter(prefix="/best11", tags=["Best 11"])


@router.get("", response_model=Best11Response,
            summary="Best-11 lineup recommendation")
async def get_best11(
    team: str = Query(..., description="Team name, e.g. 'Barcelona'"),
    league: str = Query(..., description="League name, e.g. 'La_Liga'"),
    season: str = Query("2425", description="Season code, e.g. '2425'"),
    formation: str = Query("auto", description="'auto' or a shape (4-3-3, 4-4-2, 4-2-3-1, 3-5-2)"),
    date: Optional[str] = Query(
        None, description="Through-date (ISO YYYY-MM-DD): cumulative ratings with no future leakage"),
    opponent: Optional[str] = Query(
        None, description="Team to blend 70/30 H2H ratings against, e.g. 'Real Madrid'"),
    service: Best11ApiService = Depends(get_best11_api_service),
) -> Best11Response:
    """Recommend the best XI for a team, season and formation."""
    return await service.recommend(team, league, season, formation, date, opponent)
