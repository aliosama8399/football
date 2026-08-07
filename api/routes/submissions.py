from fastapi import APIRouter, Depends
from typing import List, Dict, Any

from api.schemas import (
    MatchSubmissionCreate, MatchSubmissionResponse,
    TacticalAnalysisCreate, TacticalAnalysisResponse,
    TeamProfileEditCreate, FeedbackResponse
)
from api.dependencies import get_supervisor_service
from api.auth import get_current_user
from api.database import User

router = APIRouter(prefix="/submissions", tags=["User Submissions"])


@router.post("/match", response_model=MatchSubmissionResponse)
async def submit_match_record(
    payload: MatchSubmissionCreate,
    current_user: User = Depends(get_current_user),
    supervisor_service = Depends(get_supervisor_service)
):
    return await supervisor_service.submit_match_record(user_id=current_user.id, payload=payload)


@router.post("/tactical-analysis", response_model=TacticalAnalysisResponse)
async def submit_tactical_analysis(
    payload: TacticalAnalysisCreate,
    current_user: User = Depends(get_current_user),
    supervisor_service = Depends(get_supervisor_service)
):
    return await supervisor_service.submit_tactical_analysis(user_id=current_user.id, payload=payload)


@router.post("/team-profile", response_model=FeedbackResponse)
async def submit_team_profile_edit(
    payload: TeamProfileEditCreate,
    current_user: User = Depends(get_current_user),
    supervisor_service = Depends(get_supervisor_service)
):
    return await supervisor_service.submit_team_profile_edit(user_id=current_user.id, payload=payload)


@router.get("/mine")
async def list_my_submissions(
    current_user: User = Depends(get_current_user),
    supervisor_service = Depends(get_supervisor_service)
) -> List[Dict[str, Any]]:
    return await supervisor_service.list_my_submissions(user_id=current_user.id)
