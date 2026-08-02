from fastapi import APIRouter, Depends
from typing import List

from api.schemas import FeedbackCreate, FeedbackResponse
from api.dependencies import get_supervisor_service
from api.auth import get_current_user
from api.database import User

router = APIRouter(prefix="/feedback", tags=["Feedback Submissions"])

@router.post("", response_model=FeedbackResponse)
async def submit_feedback(
    payload: FeedbackCreate,
    current_user: User = Depends(get_current_user),
    supervisor_service = Depends(get_supervisor_service)
):
    """
    Submit a tactic modification request or a prediction override request.
    Feedback enters a 'pending' state and awaits review by a supervisor.
    """
    return await supervisor_service.submit_feedback(user_id=current_user.id, payload=payload)

@router.get("", response_model=List[FeedbackResponse])
async def list_my_feedback(
    current_user: User = Depends(get_current_user),
    supervisor_service = Depends(get_supervisor_service)
):
    """List all feedbacks submitted by the logged in user."""
    return await supervisor_service.list_my_feedback(user_id=current_user.id)
