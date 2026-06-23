from fastapi import APIRouter, Depends, HTTPException, status
from typing import List

from api.schemas import FeedbackCreate, FeedbackResponse
from api.dependencies import get_feedback_repo
from api.auth import get_current_user
from api.database import User

router = APIRouter(prefix="/feedback", tags=["Feedback Submissions"])

@router.post("", response_model=FeedbackResponse)
async def submit_feedback(
    payload: FeedbackCreate,
    current_user: User = Depends(get_current_user),
    feedback_repo = Depends(get_feedback_repo)
):
    """
    Submit a tactic modification request or a prediction override request.
    Feedback enters a 'pending' state and awaits review by a supervisor.
    """
    if payload.type not in ("prediction_override", "tactic_modification"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid type. Must be 'prediction_override' or 'tactic_modification'."
        )

    # Perform structural validation based on feedback type
    if payload.type == "prediction_override":
        if not payload.home_team or not payload.away_team or not payload.suggested_result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Prediction overrides require home_team, away_team, and suggested_result."
            )
        if payload.suggested_result not in ("H", "A", "D"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="suggested_result must be 'H' (home win), 'A' (away win), or 'D' (draw)."
            )
    else:  # tactic_modification
        if not payload.team_name or (not payload.suggested_attack_tactic and not payload.suggested_defense_tactic):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tactic modifications require a team_name and at least one tactical suggestion."
            )

    return await feedback_repo.create_feedback(
        user_id=current_user.id,
        type=payload.type,
        home_team=payload.home_team,
        away_team=payload.away_team,
        match_date=payload.match_date,
        suggested_result=payload.suggested_result,
        suggested_home_goals=payload.suggested_home_goals,
        suggested_away_goals=payload.suggested_away_goals,
        suggested_analysis=payload.suggested_analysis,
        team_name=payload.team_name,
        suggested_attack_tactic=payload.suggested_attack_tactic,
        suggested_defense_tactic=payload.suggested_defense_tactic
    )

@router.get("", response_model=List[FeedbackResponse])
async def list_my_feedback(
    current_user: User = Depends(get_current_user),
    feedback_repo = Depends(get_feedback_repo)
):
    """List all feedbacks submitted by the logged in user."""
    return await feedback_repo.list_user_feedbacks(user_id=current_user.id)
