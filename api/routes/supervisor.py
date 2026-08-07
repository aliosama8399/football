from fastapi import APIRouter, Depends
from typing import List

from api.schemas import (
    UserCreate, UserResponse, FeedbackResponse, FeedbackReviewRequest,
    MatchSubmissionResponse, TacticalAnalysisResponse, SubmissionReviewRequest
)
from api.dependencies import get_supervisor_service
from api.auth import require_supervisor
from api.database import User

router = APIRouter(prefix="/supervisor", tags=["Supervisor Controls"])

@router.post("/users", response_model=UserResponse)
async def onboard_user(
    payload: UserCreate,
    current_supervisor: User = Depends(require_supervisor),
    supervisor_service = Depends(get_supervisor_service)
):
    """
    Onboard a new user profile by generating a unique activation token.
    The profile is generated in an inactive state until the user registers their password.
    Returns the created user response showing their activation token.
    """
    return await supervisor_service.onboard_user(payload)

@router.get("/feedback", response_model=List[FeedbackResponse])
async def list_pending_feedback(
    current_supervisor: User = Depends(require_supervisor),
    supervisor_service = Depends(get_supervisor_service)
):
    """Retrieve list of all pending user feedback submissions."""
    return await supervisor_service.list_pending_feedback()

@router.post("/feedback/{feedback_id}/review", response_model=FeedbackResponse)
async def review_feedback(
    feedback_id: int,
    payload: FeedbackReviewRequest,
    current_supervisor: User = Depends(require_supervisor),
    supervisor_service = Depends(get_supervisor_service)
):
    """
    Approve or reject pending feedback.
    Approvals automatically apply tactical changes or matching prediction overrides in the database.
    """
    return await supervisor_service.review_feedback(
        feedback_id, status_=payload.status, admin_notes=payload.admin_notes,
        reviewer_id=current_supervisor.id
    )


# ── Unified Submission Queue ──────────────────────────────────────────────────

@router.get("/submissions")
async def list_all_pending_submissions(
    current_supervisor: User = Depends(require_supervisor),
    supervisor_service = Depends(get_supervisor_service)
):
    """Retrieve a unified queue of all pending user submissions (matches, tactical analyses, team profiles)."""
    return await supervisor_service.list_all_pending_submissions()


@router.post("/submissions/{submission_id}/review")
async def review_submission(
    submission_id: int,
    payload: SubmissionReviewRequest,
    type: str,
    current_supervisor: User = Depends(require_supervisor),
    supervisor_service = Depends(get_supervisor_service)
):
    """
    Approve or reject any pending submission type.
    Query param `type` must be one of: match_submission, tactical_analysis, team_profile_edit
    """
    reviewed = await supervisor_service.review_submission(
        submission_id, type=type, status_=payload.status, admin_notes=payload.admin_notes,
        reviewer_id=current_supervisor.id
    )
    if type == "match_submission":
        return MatchSubmissionResponse.model_validate(reviewed)
    elif type == "tactical_analysis":
        return TacticalAnalysisResponse.model_validate(reviewed)
    else:  # team_profile_edit
        return FeedbackResponse.model_validate(reviewed)
